import copy
from queue import Queue # pt arc consistency

def get_value_from_index_state(state, row, col):
    if (row < 0 or row > 8 or col < 0 or col > 8):
        raise Exception("Invalid index")
    return state[row][col]

def calculate_domains(p_assignment):
    state, even_cells = p_assignment
    domains = []
    for i in range(9):
        for j in range(9):
            if state[i][j] != None:
                domains.append([state[i][j]])
            elif [i, j] in even_cells:
                domains.append([2, 4, 6, 8])
            else:
                domains.append([1, 2, 3, 4, 5, 6, 7, 8, 9])
    return domains

def get_domain_from_index(p_domains, row, col):
    if (row < 0 or row > 8 or col < 0 or col > 8):
        raise Exception("Invalid index")
    return p_domains[row * 9 + col]

def verify_matrix_aux(state, start_index_row, start_index_col, step=3):
    array = []
    for i in range(start_index_row, start_index_row + step):
        for j in range(start_index_col, start_index_col + step):
            array.append(get_value_from_index_state(state, i, j))
    for i in range(len(array)):
        for j in range(i + 1, len(array)):
            if array[i] != None and array[i] == array[j]:
                return False
    return True

def verify_restrictions(p_assignment):
    state,even_cells = p_assignment
    for i in range(9):
        for j in range(9):
            for k in range(9):
                if k != j and get_value_from_index_state(state, i, j) == get_value_from_index_state(state, i, k) and get_value_from_index_state(state, i, j) != None:
                    return False
                if k != i and get_value_from_index_state(state, i, j) == get_value_from_index_state(state, k, j) and get_value_from_index_state(state, i, j) != None:
                    return False

    indexes = [[0, 0], [0, 3], [0, 6], [3, 0], [3, 3], [3, 6], [6, 0], [6, 3], [6, 6]]

    for index in indexes:
        if not verify_matrix_aux(state, index[0], index[1]):
            return False

    for i in range(9):
        for j in range(9):
            if get_value_from_index_state(state, i, j) != None and get_value_from_index_state(state, i, j) % 2 != 0 and (i, j) in even_cells:
                return False

    return True

def isComplete(p_assignment):
    state, _ = p_assignment
    for row in state:
        if None in row:
            return False
    return True

def next_unassigned_variable(p_assignment):
    state, _ = p_assignment
    for i in range(9):
        for j in range(9):
            if state[i][j] is None:
                return (i, j)
    return None

def Domain(p_domains, var):
    i,j = var
    return p_domains[i*9+j]

def update_domains_FC(p_domains,var,value):
    i,j = var
    for k in range(9):
        if value in p_domains[i*9+k]:
            p_domains[i*9+k].remove(value)
        if value in p_domains[k*9+j]:
            p_domains[k*9+j].remove(value)

    indexes = [[0, 0], [0, 3], [0, 6], [3, 0], [3, 3], [3, 6], [6, 0], [6, 3], [6, 6]]

    for index in indexes:
        if i in range(index[0], index[0] + 3) and j in range(index[1], index[1] + 3):
            for k in range(index[0], index[0] + 3):
                for l in range(index[1], index[1] + 3):
                    if value in p_domains[k*9+l]:
                        p_domains[k*9+l].remove(value)


def verify_empty_domains(p_domains):
    for domain in p_domains:
        if len(domain) == 0:
            return True
    return False

def BKT_with_FC(p_assignment, p_domains):
    if isComplete(p_assignment):
        return p_assignment

    var = next_unassigned_variable(p_assignment)
    for value in Domain(p_domains, var):
        new_assignment = copy.deepcopy(p_assignment)
        new_assignment[0][var[0]][var[1]] = value
        new_domains = copy.deepcopy(p_domains)
        update_domains_FC(new_domains, var, value)
        if not verify_empty_domains(new_domains) and verify_restrictions(new_assignment):
            v_result = BKT_with_FC(new_assignment, p_domains)
            if v_result is not None:
                return v_result
    return None

#-----------------------------------------------------------------------------------------
def next_unassigned_variable_MRV(p_assignment, p_domains):
    state, _ = p_assignment
    min = 10
    min_index = None
    for i in range(9):
        for j in range(9):
            if len(Domain(p_domains, (i, j))) < min and state[i][j] is None:
                min = len(p_domains[i*9+j])
                min_index = (i, j)
    return min_index

def BKT_with_FC_and_MRV(p_assignment, p_domains):
    if isComplete(p_assignment):
        return p_assignment

    var = next_unassigned_variable_MRV(p_assignment, p_domains)
    for value in Domain(p_domains, var):
        new_assignment = copy.deepcopy(p_assignment)
        new_assignment[0][var[0]][var[1]] = value
        new_domains = copy.deepcopy(p_domains)
        update_domains_FC(new_domains, var, value)
        if not verify_empty_domains(new_domains) and verify_restrictions(new_assignment):
            v_result = BKT_with_FC_and_MRV(new_assignment, p_domains)
            if v_result is not None:
                return v_result
    return None

#-----------------------------------------------------------------------------------------
#Arc consistency
def get_neighbours(p_assignment, var):
    state, _ = p_assignment
    i, j = var
    neighbours = []
    for k in range(9):
        if k != i and state[k][j] is None:
            neighbours.append((k, j))
        if k != j and state[i][k] is None:
            neighbours.append((i, k))

    indexes = [[0, 0], [0, 3], [0, 6], [3, 0], [3, 3], [3, 6], [6, 0], [6, 3], [6, 6]]

    for index in indexes:
        if i in range(index[0], index[0] + 3) and j in range(index[1], index[1] + 3):
            for k in range(index[0], index[0] + 3):
                for l in range(index[1], index[1] + 3):
                    if k != i and l != j and state[k][l] is None:
                        neighbours.append((k, l))

    return neighbours

def AC3(p_assignment, p_domains):
    state, _ = p_assignment
    queue = Queue()
    for i in range(9):
        for j in range(9):
            if state[i][j] is not None:
                queue.put((i, j))

    while not queue.empty():
        var = queue.get()
        for neighbour in get_neighbours(p_assignment, var):
            if len(p_domains[neighbour[0] * 9 + neighbour[1]]) == 1:
                value = p_domains[neighbour[0] * 9 + neighbour[1]][0]
                if value in p_domains[var[0] * 9 + var[1]]:
                    p_domains[var[0] * 9 + var[1]].remove(value)
                    if len(p_domains[var[0] * 9 + var[1]]) == 0:
                        return False
                    queue.put(var)
    return True

def BKT_with_FC_and_MRV_and_AC3(p_assignment, p_domains):
    if isComplete(p_assignment):
        return p_assignment

    var = next_unassigned_variable_MRV(p_assignment, p_domains)
    for value in Domain(p_domains, var):
        new_assignment = copy.deepcopy(p_assignment)
        new_assignment[0][var[0]][var[1]] = value
        new_domains = copy.deepcopy(p_domains)
        update_domains_FC(new_domains, var, value)
        if AC3(new_assignment, new_domains) and verify_restrictions(new_assignment):
            v_result = BKT_with_FC_and_MRV_and_AC3(new_assignment, p_domains)
            if v_result is not None:
                return v_result
    return None

#-----------------------------------------------------------------------------------------

if __name__ == '__main__':
    initial_state = [
        [8, 4, None, None, 5, None, None, None, None],
        [3, None, None, 6, None, 8, None, 4, None],
        [None, None, None, 4, None, 9, None, None, None],
        [None, 2, 3, None, None, None, 9, 8, None],
        [1, None, None, None, None, None, None, None, 4],
        [None, 9, 8, None, None, None, 1, 6, None],
        [None, None, None, 5, None, 3, None, None, None],
        [None, 3, None, 1, None, 6, None, None, 7],
        [None, None, None, None, 2, None, None, 1, 3]
    ]

    even_cells = [[0, 6], [2, 2], [2, 8], [3, 4], [4, 3],
                  [4, 5], [5, 4], [6, 0], [6, 6], [8, 2]]

    assignment = (initial_state, even_cells)
    domains = calculate_domains(assignment)

    result = BKT_with_FC(assignment, domains)

    if result is not None:
        print("Solution:")
        for i in range(9):
            for j in range(9):
                print(result[0][i][j], end=" ")
                if j % 3 == 2:
                    print("|", end=" ")
            print()
            if i % 3 == 2:
                print("---------------------")
    else:
        print("No solution found.")

    print("------------------------------------------------------------------")
    result = BKT_with_FC_and_MRV(assignment, domains)

    if result is not None:
        print("Solution MRV:")
        for i in range(9):
            for j in range(9):
                print(result[0][i][j], end=" ")
                if j % 3 == 2:
                    print("|", end=" ")
            print()
            if i % 3 == 2:
                print("---------------------")
    else:
        print("No solution found.")

    print("------------------------------------------------------------------")
    result = BKT_with_FC_and_MRV_and_AC3(assignment, domains)

    if result is not None:
        print("Solution MRV + AC3:")
        for i in range(9):
            for j in range(9):
                print(result[0][i][j], end=" ")
                if j % 3 == 2:
                    print("|", end=" ")
            print()
            if i % 3 == 2:
                print("---------------------")
    else:
        print("No solution found.")



