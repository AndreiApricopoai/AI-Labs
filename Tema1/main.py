from math import sqrt
from queue import PriorityQueue
from datetime import datetime
import copy

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
        # Create a deep copy of the matrix
        new_matrix = copy.deepcopy(matrix)

        cell_to_move_row, cell_to_move_col = cell_to_move

        if new_matrix[cell_to_move_row][cell_to_move_col] != 0:
            for i in range(3):
                for j in range(3):
                    if new_matrix[i][j] == 0:
                        zero_row, zero_col = i, j

            new_matrix[zero_row][zero_col] = new_matrix[cell_to_move_row][cell_to_move_col]
            new_matrix[cell_to_move_row][cell_to_move_col] = 0
            last_move_cell = (zero_row, zero_col, new_matrix[zero_row][zero_col])

            return new_matrix, last_move_cell
        else:
            if direction == "UP":
                new_row, new_col = cell_to_move_row - 1, cell_to_move_col
            elif direction == "DOWN":
                new_row, new_col = cell_to_move_row + 1, cell_to_move_col
            elif direction == "LEFT":
                new_row, new_col = cell_to_move_row, cell_to_move_col - 1
            elif direction == "RIGHT":
                new_row, new_col = cell_to_move_row, cell_to_move_col + 1

            new_matrix[cell_to_move_row][cell_to_move_col] = new_matrix[new_row][new_col]
            new_matrix[new_row][new_col] = 0
            last_move_cell = (cell_to_move_row, cell_to_move_col, new_matrix[cell_to_move_row][cell_to_move_col])

            return new_matrix, last_move_cell
    else:
        return matrix, last_move_cell

def IDDFS(init_state, max_depth):
    for depth in range(max_depth + 1):
        visited = set()
        moves_count = 0
        solution = depth_limited_DFS(init_state, depth, visited, moves_count)
        if solution is not None:
            return solution
    return None, None

def depth_limited_DFS(state, depth, visited, moves_count):
    if is_final_state(state[0]):  # Check if the matrix is in the final state
        return state, moves_count
    if depth == 0:
        return None

    visited.add(tuple(map(tuple, state[0])))

    for i in range(3):
        for j in range(3):
            if verify_transition(state[0], state[1], (i, j), "UP"):
                new_state, new_last_moved_cell = transition(state[0], state[1], (i, j), "UP")
                if tuple(map(tuple, new_state)) not in visited:
                    moves_count += 1
                    result = depth_limited_DFS((new_state, new_last_moved_cell), depth - 1, visited, moves_count)
                    if result is not None:
                        return result

            if verify_transition(state[0], state[1], (i, j), "DOWN"):
                new_state, new_last_moved_cell = transition(state[0], state[1], (i, j), "DOWN")
                if tuple(map(tuple, new_state)) not in visited:
                    moves_count += 1
                    result = depth_limited_DFS((new_state, new_last_moved_cell), depth - 1, visited, moves_count)
                    if result is not None:
                        return result

            if verify_transition(state[0], state[1], (i, j), "LEFT"):
                new_state, new_last_moved_cell = transition(state[0], state[1], (i, j), "LEFT")
                if tuple(map(tuple, new_state)) not in visited:
                    moves_count += 1
                    result = depth_limited_DFS((new_state, new_last_moved_cell), depth - 1, visited, moves_count)
                    if result is not None:
                        return result

            if verify_transition(state[0], state[1], (i, j), "RIGHT"):
                new_state, new_last_moved_cell = transition(state[0], state[1], (i, j), "RIGHT")
                if tuple(map(tuple, new_state)) not in visited:
                    moves_count += 1
                    result = depth_limited_DFS((new_state, new_last_moved_cell), depth - 1, visited, moves_count)
                    if result is not None:
                        return result

    return None

# implement manhattan heuristic, use the final states
def manhattan_heuristic(state):
    final_states = generate_final_states()
    min = 10000000
    for i in range(len(final_states)):
        sum = 0
        for j in range(3):
            for k in range(3):
                if state[j][k] != 0:
                    for l in range(3):
                        for m in range(3):
                            if state[j][k] == final_states[i][l][m]:
                                sum += abs(j - l) + abs(k - m)
        if sum < min:
            min = sum
    return min

#implement hamming heuristic, use the final states
def hamming_heuristic(state):
    final_state = generate_final_states()[0]  # Assume we're working with the first final state

    # Flatten the current state and final state matrices
    flat_state = [value for row in state for value in row]
    flat_final_state = [value for row in final_state for value in row]

    # Count the number of misplaced tiles
    misplaced_count = sum(1 for s, f in zip(flat_state, flat_final_state) if s != f and s != 0)

    return misplaced_count


#implement CHEBYSHEV DISTANCE heuristic, use the final states
def chebyshev_heuristic(state):
    final_states = generate_final_states()
    min = 10000000
    for i in range(len(final_states)):
        sum = 0
        for j in range(3):
            for k in range(3):
                if state[j][k] != 0:
                    for l in range(3):
                        for m in range(3):
                            if state[j][k] == final_states[i][l][m]:
                                sum += max(abs(j - l), abs(k - m))
        if sum < min:
            min = sum
    return min

#implement greedy search for a state + heuristic
def greedy_search(initial_state, heuristic_fn):
    # Initialize a priority queue (heap)
    priority_queue = PriorityQueue()
    # Create a set to keep track of visited states
    visited = set()
    # Push the initial state with its heuristic value to the queue
    priority_queue.put((heuristic_fn(initial_state[0]), initial_state))

    iterations = 0

    while not priority_queue.empty():
        # Get the state with the bigest heuristic value

        '''
        print("plm")
        priority_queue_aux = PriorityQueue()
        while not priority_queue.empty():
            item = priority_queue.get()
            priority_queue_aux.put((item[0], item[1]))
            print("Priority:", item[0], "Item:", item[1])
        while not priority_queue_aux.empty():
            item = priority_queue_aux.get()
            priority_queue.put((item[0], item[1]))
        '''
        current_state_1 = priority_queue.get()

        #print (current_state_1)

        current_state = current_state_1[1]

        # Check if the current state is the goal state
        if is_final_state(current_state[0]):
            return current_state, iterations

        iterations += 1

        # Add the current state to the visited set
        visited.add(tuple(map(tuple, current_state[0])))

        # Generate and check all possible next states
        for i in range(3):
            for j in range(3):
                if verify_transition(current_state[0], current_state[1], (i, j), "UP"):
                    new_state1, new_last_moved_cell1 = transition(current_state[0], current_state[1], (i, j), "UP")
                    state_tuple = tuple(map(tuple, new_state1))
                    if state_tuple not in visited:
                        visited.add(state_tuple)
                        priority_queue.put((heuristic_fn(new_state1), (new_state1, new_last_moved_cell1)))

                if verify_transition(current_state[0], current_state[1], (i, j), "DOWN"):
                    new_state2, new_last_moved_cell2 = transition(current_state[0], current_state[1], (i, j), "DOWN")
                    state_tuple = tuple(map(tuple, new_state2))
                    if state_tuple not in visited:
                        visited.add(state_tuple)
                        priority_queue.put((heuristic_fn(new_state2), (new_state2, new_last_moved_cell2)))

                if verify_transition(current_state[0], current_state[1], (i, j), "LEFT"):
                    new_state3, new_last_moved_cell3 = transition(current_state[0], current_state[1], (i, j), "LEFT")
                    state_tuple = tuple(map(tuple, new_state3))
                    if state_tuple not in visited:
                        visited.add(state_tuple)
                        priority_queue.put((heuristic_fn(new_state3), (new_state3, new_last_moved_cell3)))

                if verify_transition(current_state[0], current_state[1], (i, j), "RIGHT"):
                    new_state4, new_last_moved_cell4 = transition(current_state[0], current_state[1], (i, j), "RIGHT")
                    state_tuple = tuple(map(tuple, new_state4))
                    if state_tuple not in visited:
                        visited.add(state_tuple)
                        priority_queue.put((heuristic_fn(new_state4), (new_state4, new_last_moved_cell4)))

    return None

# Helper function to format and print results
def print_results(strategy_name, instance_index, result, time_elapsed, moves_count=0):
    if result:
        state, moves = result
        print(f"Strategy: {strategy_name}")
        print(f"Instance {instance_index + 1}:")
        print("Solution:")
        for row in state:
            print(row)
        print(f"Last move cell: {moves}")
        print(f"Number of moves: {moves_count}")
        print(f"Execution time: {time_elapsed:.6f} seconds")
        print("-" * 30)
    else:
        print(f"Strategy: {strategy_name}")
        print(f"Instance {instance_index + 1}: No solution found")
        print("-" * 30)


if __name__ == "__main__":
    # Define the initial puzzle instances
    initial_instances = [
        [8, 6, 7, 2, 5, 4, 0, 3, 1],
        [2, 5, 3, 1, 0, 6, 4, 7, 8],
        [2, 7, 5, 0, 8, 4, 3, 1, 6]
    ]
# Run strategies for all instances
    for instance_index, initial_state in enumerate(initial_instances):
        print(f"Initial State for Instance {instance_index + 1}:")
        for row in from_vector_to_matrix_init(initial_state)[0]:
            print(row)
        print("-" * 30)

        # IDDFS
        start_time = datetime.now()
        iddfs_result, depth = IDDFS(from_vector_to_matrix_init(initial_state), 45)  # You can adjust the max depth
        end_time = datetime.now()
        iddfs_time_elapsed = (end_time - start_time).total_seconds()
        print_results("IDDFS", instance_index, iddfs_result, iddfs_time_elapsed, depth)

        # Greedy Search with Manhattan Heuristic
        start_time = datetime.now()
        greedy_manhattan_result, iterations = greedy_search(from_vector_to_matrix_init(initial_state), manhattan_heuristic)
        end_time = datetime.now()
        greedy_manhattan_time_elapsed = (end_time - start_time).total_seconds()
        print_results("Greedy (Manhattan)", instance_index, greedy_manhattan_result, greedy_manhattan_time_elapsed, iterations)

        # Greedy Search with Hamming Heuristic
        start_time = datetime.now()
        greedy_hamming_result, iterations = greedy_search(from_vector_to_matrix_init(initial_state), hamming_heuristic)
        end_time = datetime.now()
        greedy_hamming_time_elapsed = (end_time - start_time).total_seconds()
        print_results("Greedy (Hamming)", instance_index, greedy_hamming_result, greedy_hamming_time_elapsed, iterations)

        # Greedy Search with Chebyshev Heuristic
        start_time = datetime.now()
        greedy_chebyshev_result, iterations = greedy_search(from_vector_to_matrix_init(initial_state), chebyshev_heuristic)
        end_time = datetime.now()
        greedy_chebyshev_time_elapsed = (end_time - start_time).total_seconds()
        print_results("Greedy (Chebyshev)", instance_index, greedy_chebyshev_result, greedy_chebyshev_time_elapsed, iterations)



