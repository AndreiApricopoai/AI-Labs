class Game:

    def __init__(self):
        self.matrix = [[(None, None) for i in range(3)] for j in range(3)]
        self.player_turn = "KeyboardPlayer"
        self.player = "BotAI"

    def find_first_empty_cell(self):
        for i in range(3):
            for j in range(3):
                if self.matrix[i][j][0] == None:
                    return (i, j)
        return None

    def check_for_winner(self):

        sumAI = 0
        sumKeyboard = 0
        countAI = self.count_player_moves("BotAI")
        countKeyboard = self.count_player_moves("KeyboardPlayer")

        for i in range(3):
            for j in range(3):
                if self.matrix[i][j][1] == "BotAI":
                    sumAI += self.matrix[i][j][0]
                elif self.matrix[i][j][1] == "KeyboardPlayer":
                    sumKeyboard += self.matrix[i][j][0]
        if sumAI == 15 and countAI <= 3:
            return "BotAI"
        elif sumKeyboard == 15 and countKeyboard <= 3:
            return "KeyboardPlayer"
        else:
            return None

    def count_player_moves(self, player):
        sum = 0
        for i in range(3):
            for j in range(3):
                if self.matrix[i][j][1] == player:
                    sum += 1
        return sum

    def check_for_draw(self):
        # if there are no empty cells or there is no player with under 15, then draw
        countAI = self.count_player_moves("BotAI")
        countKeyboard = self.count_player_moves("KeyboardPlayer")

        if countAI >= 3 and countKeyboard >= 3:
            return True

        sumAI = 0
        sumKeyboard = 0
        for i in range(3):
            for j in range(3):
                if self.matrix[i][j][1] == "BotAI":
                    sumAI += self.matrix[i][j][0]
                elif self.matrix[i][j][1] == "KeyboardPlayer":
                    sumKeyboard += self.matrix[i][j][0]
        if (sumAI >= 15 and sumKeyboard >= 15):
            return True
        else:
            return False

    def valide_move(self, value):
        if value < 1 or value > 9:
            return False
        for i in range(3):
            for j in range(3):
                if self.matrix[i][j][0] == value:
                    return False
        return True

    def find_best_move(self):
        sum_keyboard = 0

        for i in range(3):
            for j in range(3):
                if self.matrix[i][j][1] == "KeyboardPlayer":
                    sum_keyboard += self.matrix[i][j][0]

        if (sum_keyboard >= 15):
            sum_ai = 0
            for i in range(3):
                for j in range(3):
                    if self.matrix[i][j][1] == "BotAI":
                        sum_ai += self.matrix[i][j][0]
            potential_move = 15 - sum_ai
        else:
            potential_move = min(15 - sum_keyboard, 9)

        if self.valide_move(potential_move) == True:
            return potential_move
        else:
            initial_potential_move = potential_move
            while self.valide_move(potential_move) == False:
                potential_move -= 1
                if potential_move < 1:
                    break
            if potential_move < 1:
                potential_move = initial_potential_move
                while self.valide_move(potential_move) == False:
                    potential_move += 1
            return potential_move

    def print_special(self):
        # print matrix (value and player)
        for i in range(3):
            for j in range(3):
                if self.matrix[i][j][0] == None:
                    print("(NA/NA)", end=" ")
                else:
                    print("(" + str(self.matrix[i][j][0]) + "/" + str(self.matrix[i][j][1]) + ")", end=" ")
            print("\n")

    def make_move(self):
        cell = self.find_first_empty_cell()
        if cell == None:
            print("No more empty cells - draw")
            return False

        if self.player_turn == "KeyboardPlayer":
            value = int(input("KeyboardPlayer turn: "))
            while self.valide_move(value) == False:
                value = int(input("KeyboardPlayer turn(please input a valid move): "))
            self.matrix[cell[0]][cell[1]] = (value, "KeyboardPlayer")
            self.player_turn = "BotAI"
        else:
            value = self.find_best_move()
            self.matrix[cell[0]][cell[1]] = (value, "BotAI")
            self.player_turn = "KeyboardPlayer"

        print("Current state: ")
        self.print_special()

        check_win = self.check_for_winner()
        check_draw = self.check_for_draw()

        if check_win == "KeyboardPlayer" or check_win == "BotAI":
            print("The winner is: " + check_win)
            return False
        elif check_draw == True:
            print("Draw")
            return False
        else:
            print("The game continues...")

        return True

    def heuristic(self, player):
        # Define a heuristic function to evaluate the game state.
        # Assign a higher score for states where the player is closer to winning.
        sum_player = 0
        for i in range(3):
            for j in range(3):
                if self.matrix[i][j][1] == player:
                    sum_player += self.matrix[i][j][0]

        if sum_player >= 15:
            return 100  # AI player wins
        return sum_player  # Return the sum as a heuristic score

    def minimax(self, depth, is_max_player):
        if depth == 0 or self.check_for_winner():
            # Terminal state or depth limit reached, return the heuristic value
            return self.heuristic("BotAI")

        if is_max_player:
            max_eval = -float('inf')
            for i in range(3):
                for j in range(3):
                    if self.matrix[i][j][0] is None:
                        self.matrix[i][j] = (self.find_best_move(), "BotAI")  # AI makes a move
                        eval = self.minimax(depth - 1, False)
                        self.matrix[i][j] = (None, None)  # Undo the move
                        max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for i in range(3):
                for j in range(3):
                    if self.matrix[i][j][0] is None:
                        self.matrix[i][j] = (self.find_best_move(), "KeyboardPlayer")  # Opponent makes a move
                        eval = self.minimax(depth - 1, True)
                        self.matrix[i][j] = (None, None)  # Undo the move
                        min_eval = min(min_eval, eval)
            return min_eval

    def find_best_move_with_minimax(self):
        best_move = None
        best_eval = -float('inf')
        for i in range(3):
            for j in range(3):
                if self.matrix[i][j][0] is None:
                    self.matrix[i][j] = (self.find_best_move(), "BotAI")  # AI makes a move
                    eval = self.minimax(2, False)  # Look ahead two moves
                    self.matrix[i][j] = (None, None)  # Undo the move
                    if eval > best_eval:
                        best_eval = eval
                        best_move = (i, j)
        return best_move

    def make_move_with_minimax(self):
        cell = self.find_first_empty_cell()
        if cell is None:
            print("No more empty cells - draw")
            return False

        if self.player_turn == "KeyboardPlayer":
            value = int(input("KeyboardPlayer turn: "))
            while not self.valide_move(value):
                value = int(input("KeyboardPlayer turn(please input a valid move): "))
            self.matrix[cell[0]][cell[1]] = (value, "KeyboardPlayer")
            self.player_turn = "BotAI"
        else:
            best_move = self.find_best_move_with_minimax()
            if best_move:
                self.matrix[best_move[0]][best_move[1]] = (self.find_best_move(), "BotAI")
                self.player_turn = "KeyboardPlayer"

        print("Current state:")
        self.print_special()

        check_win = self.check_for_winner()
        check_draw = self.check_for_draw()

        if check_win == "KeyboardPlayer" or check_win == "BotAI":
            print("The winner is: " + check_win)
            return False
        elif check_draw:
            print("Draw")
            return False
        else:
            print("The game continues...")

        return True

    # minmax


'''
def minimax(state, depth, is_max_player):

    if (depth == 0 || is_final(state)):
        return h(state)

    if (is_max_player):
        value = - INF
        for each valid child of state:
        value = max(value, minimax(child, depth-1, False))
    else:
    value = + INF
    for each valid child of state:
        value = min(value, minimax(child, depth-1, True))

    return value
'''

if __name__ == '__main__':
    game = Game()

    while game.make_move_with_minimax() == True:
        pass