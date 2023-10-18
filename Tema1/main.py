from math import sqrt
import time

# GENERATE INIT STATE
def from_vector_to_matrix_init(vec):
    n = len(vec)
    m = int(sqrt(n))
    mat = [[0 for i in range(m)] for j in range(m)]
    for i in range(m):
        for j in range(m):
            mat[i][j] = vec[i * m + j]
    # last_moved_Cell retain the last moved cell and the value from the cell (10, 10, 10) - initial value
    last_moved_cell = (-1, -1, -1)
    return mat, last_moved_cell


# VERIFY IS FINAL STATE OR NOT - FINAL STATE IS THE MATRIX SORTED ASCENDING, excepting 0
def is_final_state(mat):
    n = len(mat)
    # init vector from mat
    vec = []
    for i in range(n):
        for j in range(n):
            if(mat[i][j] != 0):
                vec.append(mat[i][j])
    # check if vector is sorted

    for i in range(0, (n * n) - 2):
        if vec[i] > vec[i + 1]:
            return False
    return True


# GENERATE FINAL STATES - FINAL STATE IS THE MATRIX SORTED ASCENDING, excepting 0 as a value from matrix, return a list of matrix
def move_zero(vec):
    n = len(vec)
    for i in range(n):
        if vec[i] == 0:
            if i == n - 1:
                return vec
            else:
                vec[i] = vec[i + 1]
                vec[i + 1] = 0
                return vec
    return vec


def generate_final_states():
    vec = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    final_states = []
    for i in range(len(vec)):
        mat, last_moved_cell = from_vector_to_matrix_init(vec)
        final_states.append(mat)
        vec = move_zero(vec)
    return final_states


# ------------------------------------------------------------------------------------------------------------------------------

# MAKE A FUNCTION THAT TAKES AS PARAMETER A MATRIX, A LAST MOVE CELL AND A DIRECTION(STRING UP,DOWN,RIGHT,LEFT) (TO MOVE THE CELL 0 IN THAT DIRECTION)
# AND RETURN THE NEW MATRIX AND THE NEW LAST MOVE CELL, VERIFYING IF THE MOVE IS POSSIBLE (A CELL CAN BE MOVED JUST IN THE POSITION WHERE IS 0 )

def verify_transition(matrix, last_move_cell, cell_to_move, direction):
    cell_to_move_row, cell_to_move_col = cell_to_move

    if(matrix[cell_to_move_row][cell_to_move_col] != 0):
        # Găsim poziția elementului 0 în matrice
        for i in range(3):
            for j in range(3):
                if matrix[i][j] == 0:
                    zero_row, zero_col = i, j

        last_row, last_col, last_value = last_move_cell

        # Verificăm direcția și calculăm noua poziție a elementului 0
        if direction == "UP":
            new_row, new_col = cell_to_move_row - 1, cell_to_move_col
        elif direction == "DOWN":
            new_row, new_col = cell_to_move_row + 1, cell_to_move_col
        elif direction == "LEFT":
            new_row, new_col = cell_to_move_row, cell_to_move_col - 1
        elif direction == "RIGHT":
            new_row, new_col = cell_to_move_row, cell_to_move_col + 1
        else:
            return False

        # Verificăm dacă noua poziție este în interiorul matricei
        if (0 <= new_row < 3 and 0 <= new_col < 3 and (cell_to_move_row, cell_to_move_col) != (last_row, last_col)
                and (new_row, new_col) == (zero_row, zero_col)):
            return True
        else:
            return False
    else:
        last_row, last_col, last_value = last_move_cell

        # Verificăm direcția și calculăm noua poziție a elementului 0
        if direction == "UP":
            new_row, new_col = cell_to_move_row - 1, cell_to_move_col
        elif direction == "DOWN":
            new_row, new_col = cell_to_move_row + 1, cell_to_move_col
        elif direction == "LEFT":
            new_row, new_col = cell_to_move_row, cell_to_move_col - 1
        elif direction == "RIGHT":
            new_row, new_col = cell_to_move_row, cell_to_move_col + 1
        else:
            return False

        # Verificăm dacă noua poziție este în interiorul matricei
        if (0 <= new_row < 3 and 0 <= new_col < 3 and (new_row, new_col) != (last_row, last_col)):
            return True
        else:
            return False


def transition(matrix, last_move_cell, cell_to_move, direction):
    verify_move_ok = verify_transition(matrix, last_move_cell, cell_to_move, direction)
    if verify_move_ok == True:
        cell_to_move_row, cell_to_move_col = cell_to_move
        # Găsim poziția elementului 0 în matrice
        if (matrix[cell_to_move_row][cell_to_move_col] != 0):
            for i in range(3):
                for j in range(3):
                    if matrix[i][j] == 0:
                        zero_row, zero_col = i, j

            matrix[zero_row][zero_col] = matrix[cell_to_move_row][cell_to_move_col]
            matrix[cell_to_move_row][cell_to_move_col] = 0
            last_move_cell = (zero_row, zero_col, matrix[zero_row][zero_col])
            return matrix, last_move_cell
        else:
            if direction == "UP":
                new_row, new_col = cell_to_move_row - 1, cell_to_move_col
            elif direction == "DOWN":
                new_row, new_col = cell_to_move_row + 1, cell_to_move_col
            elif direction == "LEFT":
                new_row, new_col = cell_to_move_row, cell_to_move_col - 1
            elif direction == "RIGHT":
                new_row, new_col = cell_to_move_row, cell_to_move_col + 1

            matrix[cell_to_move_row][cell_to_move_col] = matrix[new_row][new_col]
            matrix[new_row][new_col] = 0
            last_move_cell = (cell_to_move_row, cell_to_move_col, matrix[cell_to_move_row][cell_to_move_col])
            return matrix, last_move_cell
    else:
        return matrix, last_move_cell

def IDDFS(init_state, max_depth):
    for depth in range(max_depth + 1):
        visited = set()
        solution = depth_limited_DFS(init_state, depth, visited)
        if solution is not None:
            return solution
    return None

def depth_limited_DFS(state, depth, visited):
    if is_final_state(state[0]):  # Check if the matrix is in the final state
        return state
    if depth == 0:
        return None

    visited.add(tuple(map(tuple, state[0])))

    # for state 0 print each line


    for i in range(3):
        for j in range(3):
            if verify_transition(state[0], state[1], (i, j), "UP"):
                new_state, new_last_moved_cell = transition(state[0], state[1], (i, j), "UP")
                if tuple(map(tuple, new_state)) not in visited:
                    result = depth_limited_DFS((new_state, new_last_moved_cell), depth - 1, visited)
                    if result is not None:
                        return result

            if verify_transition(state[0], state[1], (i, j), "DOWN"):
                new_state, new_last_moved_cell = transition(state[0], state[1], (i, j), "DOWN")
                if tuple(map(tuple, new_state)) not in visited:
                    result = depth_limited_DFS((new_state, new_last_moved_cell), depth - 1, visited)
                    if result is not None:
                        return result

            if verify_transition(state[0], state[1], (i, j), "LEFT"):
                new_state, new_last_moved_cell = transition(state[0], state[1], (i, j), "LEFT")
                if tuple(map(tuple, new_state)) not in visited:
                    result = depth_limited_DFS((new_state, new_last_moved_cell), depth - 1, visited)
                    if result is not None:
                        return result

            if verify_transition(state[0], state[1], (i, j), "RIGHT"):
                new_state, new_last_moved_cell = transition(state[0], state[1], (i, j), "RIGHT")
                if tuple(map(tuple, new_state)) not in visited:
                    result = depth_limited_DFS((new_state, new_last_moved_cell), depth - 1, visited)
                    if result is not None:
                        return result
    return None


if __name__ == '__main__':
    #calculate the time, take init time
    start_time = time.time()
    x = [2, 5, 3, 1, 0, 6, 4, 7, 8]
    init_state, last_moved_cell = from_vector_to_matrix_init(x)
    max_depth = 15
    solution = IDDFS((init_state, last_moved_cell), max_depth)

    if solution is not None:
        finish_time = time.time()
        print("Time: ", finish_time - start_time)
        print("Solution found:")
        for row in solution[0]:
            print(row)
    else:
        print("No solution found within the maximum depth.")

    start_time = time.time()
    x = [8, 6, 7, 2, 5, 4, 0, 3, 1]
    init_state, last_moved_cell = from_vector_to_matrix_init(x)
    solution = IDDFS((init_state, last_moved_cell), max_depth)

    if solution is not None:
        finish_time = time.time()
        print("Time: ", finish_time - start_time)
        print("Solution found:")
        for row in solution[0]:
            print(row)
    else:
        print("No solution found within the maximum depth.")

    start_time = time.time()
    x = [2, 7, 5, 0, 8, 4, 3, 1, 6]
    init_state, last_moved_cell = from_vector_to_matrix_init(x)
    solution = IDDFS((init_state, last_moved_cell), max_depth)

    if solution is not None:
        finish_time = time.time()
        print("Time: ", finish_time - start_time)
        print("Solution found:")
        for row in solution[0]:
            print(row)
    else:
        print("No solution found within the maximum depth.")