import random
import time

class Board:
    # creates a board (state is stored as a base-3 integer)
    # each cell is 3^i (i is the cell number 0-8)
    # to get the state of the board just see if each place from 3^(1 through 8) is a:
    # 0 - empty
    # 1 - X
    # 2 - O
    # Memory efficient storage
    def __init__(self):
        self.state = 0
    
    # gets the number of the cell 0-2
    def _get_cell(self, row, col):
        index = row * 3 + col
        return (self.state // (3 ** index)) % 3

    # sets the cell number 0-2
    def _set_cell(self, row, col, value):
        index = row * 3 + col
        current_value = self._get_cell(row, col)
        self.state -= current_value * (3 ** index)
        self.state += value * (3 ** index)

    # converts the base-3 int to the string version
    # prints it to the console with row and column numbers
    def display(self):
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        print("   0   1   2")
        for r in range(3):
            row = [symbols[self._get_cell(r, c)] for c in range(3)]
            print(f"{r}  " + " | ".join(row))
            if r < 2:
                print("  " + "-" * 11)

    # checks if the square is empty or not
    def is_valid_move(self, row, col):
        return 0 <= row < 3 and 0 <= col < 3 and self._get_cell(row, col) == 0

    # gets all empty squares
    def get_valid_moves(self):
        valid_moves = []
        for i in range(9):
            row, col = divmod(i, 3)
            if self.is_valid_move(row, col):
                valid_moves.append(i)
        return valid_moves

    # makes the move given cpu input which is a number from 0-8
    # for players the row is multipled by 3 and the column num is added to get the cell number
    # row, col converesion to 0-8 is done in the main game loop
    def make_move_cpu(self, player_num, move):
        if 0 <= move < 9:
            row, col = divmod(move, 3)
            if self.is_valid_move(row, col):
                self._set_cell(row, col, player_num)
                return True
        return False

    # creates a list of all possible winning lines (vertical, horizontal, both diagonals)
    # checks if any of the lists are 3 of a player's symbol
    # returns: -1 = draw, 0 = still going, 1 = player1 wins, 2 = player2 wins
    def check_winner(self):
        lines = []
        for i in range(3):
            lines.append([self._get_cell(i, j) for j in range(3)])
            lines.append([self._get_cell(j, i) for j in range(3)])
        lines.append([self._get_cell(i, i) for i in range(3)])
        lines.append([self._get_cell(i, 2 - i) for i in range(3)])
        for line in lines:
            if line.count(line[0]) == 3 and line[0] != 0:
                return line[0]
        if all(self._get_cell(r, c) != 0 for r in range(3) for c in range(3)):
            return -1
        return 0

class Player:
    # player gets a name and a marker (X or O)
    def __init__(self, name, marker):
        self.name = name
        self.marker = marker
    
    # asks player for input via console
    def get_move(self, board):
        while True:
            try:
                row = int(input(f"\n{self.name}, enter row (0-2): "))
                col = int(input(f"{self.name}, enter col (0-2): "))
                print()
                if board.is_valid_move(row, col):
                    return row, col
            except:
                pass

class CPU(Player):
    # same as player but it also stores:
    # q_values - each state, action mapped to the total_reward, times_reached
    # moves - the current state-action pairs played by the cpu in the game
    def __init__(self, name, marker):
        super().__init__(name, marker)
        self.q_values = {}
        self.moves = []

    # gets the move and saves the state action pair in self.moves
    # move played depends on the gen:
    # gen 1 - random moves
    # gen 2 - better, smarter moves
    # gen 3 - best possible player (never loses if gen1 and gen2 ran for enough games)
    def get_and_save_move(self, board: Board, gen=1, q_values=None, epsilon=0.1):
        # gets all valid moves on the board
        moves = board.get_valid_moves()
        # if gen1, then play a random move and save the state-action pair
        if gen == 1 or q_values is None or random.random() < epsilon:
            move = random.choice(moves)
        # if not gen 1, then make moves based on the q_values
        else:
            # initialize a list storing the (average_reward, move) pairs for all valid moves
            q_avgs = []
            # go through each possible move
            for m in moves:
                # get the q_value stats (state, action) from the dictionary
                total, count = q_values.get((board.state, m), (0, 0))
                # calculate the average reward based on the total_reward and times_reached
                avg = total / count if count > 0 else 0
                # append the tuple
                q_avgs.append((avg, m))
            # get the highest average in the list (the best move based on previous games and data)    
            max_avg = max(q_avgs, key=lambda x: x[0])[0]
            # handle ties between moves if the average reward is the same
            best_moves = [m for avg, m in q_avgs if avg == max_avg]
            # pick a random move out of the best move (only 1) or tied best moves (multiple)
            move = random.choice(best_moves)
        # add the move to the self.moves list (moves played in the current game)
        self.moves.append((board.state, move))
        # then return the move
        return move

# simulates games between two cpu objects:
# one is X, the other is O
# they play moves based on the generation
# same as a regular game but the moves are not printed to the board for efficiency
# trains the cpu to find the best moves based on reward
def simulate_game(cpu1: CPU, cpu2: CPU, q_values: dict, stats: dict, gen=1, epsilon=0.1):
    board = Board()
    cpu1.moves = []
    cpu2.moves = []
    current_player, other_player = cpu1, cpu2

    # simulates a full game in the while loop
    while True:
        # gets the move
        move = current_player.get_and_save_move(board, gen=gen, q_values=q_values, epsilon=epsilon)
        # plays the move
        board.make_move_cpu(current_player.marker, move)
        # checks the result
        result = board.check_winner()
        # if the result is not a draw or no one has won yet
        if result != 0:
            # if the result is -1 (draw): neither player gains any reward
            if result == -1:
                reward1 = 0
                reward2 = 0
                stats['draws'] += 1

            # if someone won:
            # winner gets reward = 1
            # loser gets reward = -1
            elif result == cpu1.marker:
                reward1 = 1
                reward2 = -1
                stats['win_loss'] += 1
            else:
                reward1 = -1
                reward2 = 1
                stats['win_loss'] += 1
            
            # assigns the respective reward to all of the state-action pairs played by each cpu
            for state, action in cpu1.moves:
                total_reward, times = q_values.get((state, action), (0, 0))         # first it gets the current total reward and times reached
                q_values[(state, action)] = (total_reward + reward1, times + 1)     # then it adds the reward for this game and it adds 1 to the number of times reached
            for state, action in cpu2.moves:
                total_reward, times = q_values.get((state, action), (0, 0))         # same as first cpu
                q_values[(state, action)] = (total_reward + reward2, times + 1)

            # ends the while loop (game finished)
            break
        
        # otherwise game has not finished so switch the turn
        current_player, other_player = other_player, current_player

# trains the cpu given a number of games (iterations) for the first and second gen
# first gen = random moves
# second gen = moves based on reward with a 0.1 epsilon (epsilon = chance to try a random move)
# epsilon allows the cpu to keep learning (otherwise the reward gap between good and bad moves would just increase
# but it wouldn't chance much about the moves the cpu plays)
# *** Recommended: First Gen is atleast 10,000-100,000 games for exploration, Second Gen - around half of the first gen
def train_cpus(first_gen_games=0, second_gen_games=0, test_games=0):
    start_time = time.time()
    q_values = {}   # stores all the state-action pairs to reward-times pairs (both X and O to allow a single CPU object to be good at both)
    cpu1 = CPU("CPU 1", 1)
    cpu2 = CPU("CPU 2", 2)

    # Gen 1 training
    stats = {'win_loss': 0, 'draws': 0}
    for _ in range(first_gen_games):
        simulate_game(cpu1, cpu2, q_values, stats, gen=1)
    print(f"\nThe gen 1 CPU played itself {first_gen_games} times.\nThe games ended in a win/loss {stats['win_loss']} times.\nThe games ended in a draw {stats['draws']} times.\n")

    # Gen 2 training
    stats = {'win_loss': 0, 'draws': 0}
    for _ in range(second_gen_games):
        simulate_game(cpu1, cpu2, q_values, stats, gen=2, epsilon=0.1)
    print(f"The gen 2 CPU played itself {second_gen_games} times.\nThe games ended in a win/loss {stats['win_loss']} times.\nThe games ended in a draw {stats['draws']} times.\n")

    # Gen 3 testing
    # Tests the greediest version
    # only plays the moves it thinks are the best
    # never plays any random moves (epsilon=0)
    if test_games > 0:
        stats = {'win_loss': 0, 'draws': 0}
        for _ in range(test_games):
            simulate_game(cpu1, cpu2, q_values, stats, gen=3, epsilon=0)
        print(f"The gen 3 CPU played itself {test_games} times.\nThe games ended in a win/loss {stats['win_loss']} times.\nThe games ended in a draw {stats['draws']} times.\n")
    end_time = time.time()
    dif = end_time-start_time
    return q_values, dif

# player vs. cpu
# actually prints the board out to the screen
def play_against_cpu(q_values, cpu_marker=2):
    human_marker = 2 if cpu_marker == 1 else 1
    cpu = CPU("CPU", cpu_marker)
    cpu.q_values = q_values
    human = Player("Human", human_marker)
    board = Board()
    current_player = cpu if cpu_marker == 1 else human
    while True:
        board.display()
        if current_player == cpu:
            move = cpu.get_and_save_move(board, gen=3, q_values=q_values, epsilon=0)
            board.make_move_cpu(cpu.marker, move)
            row, col = divmod(move, 3)
            print(f"\nCPU played at row {row}, col {col}\n")
        else:
            row, col = human.get_move(board)
            board.make_move_cpu(human.marker, row*3 + col)
        result = board.check_winner()
        if result != 0:
            board.display()
            if result == -1:
                print("\nIt's a draw!\n")
            elif result == cpu.marker:
                print("\nCPU wins!\n")
            else:
                print("\nYou win!\n")
            break
        current_player = human if current_player == cpu else cpu

# cpu vs. cpu
# player gets to watch the CPUs play against each other
# requires seconds as parameter for delay between moves
def play_cpu_vs_cpu(q_values, cpu1_marker=1, cpu2_marker=2, seconds=1):
    cpu1 = CPU("CPU 1", cpu1_marker)
    cpu2 = CPU("CPU 2", cpu2_marker)
    cpu1.q_values = q_values
    cpu2.q_values = q_values
    board = Board()
    current_player, other_player = (cpu1, cpu2) if cpu1_marker == 1 else (cpu2, cpu1)

    board.display()
    time.sleep(seconds)
    while True:
        move = current_player.get_and_save_move(board, gen=3, q_values=q_values, epsilon=0)
        board.make_move_cpu(current_player.marker, move)
        row, col = divmod(move, 3)
        print(f"\n{current_player.name} played at row {row}, col {col}\n")
        board.display()
        time.sleep(seconds)

        result = board.check_winner()
        if result != 0:
            if result == -1:
                print("\nIt's a draw!\n")
            elif result == current_player.marker:
                print(f"\n{current_player.name} wins!\n")
            else:
                print(f"\n{other_player.name} wins!\n")
            break

        current_player, other_player = other_player, current_player

if __name__ == "__main__":
    while True:
        match_up = input("\nChoose an option:\n1 = Player vs. CPU\n2 = Watch CPU vs. CPU\n3 = Test CPU playing itself\nAnything Else = Exit\nEnter number: ").strip()
        if match_up not in {'1','2','3'}:
            print("\nQuitting Program.\n")
            break

        first_gen = int(input("\nEnter number of first generation (random) games to simulate: "))
        second_gen = int(input("\nEnter number of second generation (epsilon-greedy) games to simulate: "))


        if match_up == '3':
            test_games = int(input("\nEnter number of test games (gen 3) to simulate: "))
            q_vals, dif = train_cpus(first_gen, second_gen, test_games=test_games)
            print("Testing complete!\n")
            print(f"Total time spent training was {dif} seconds.\n")
        else:
            q_vals, dif = train_cpus(first_gen, second_gen, test_games=0)
            print(f"Total time spent training was {dif} seconds.\n")

        if match_up == '1':
            x_o = int(input("Do you want to play as X or O?\n1 = X\n2 = O\nEnter a number: "))
            if x_o == 2:
                play_against_cpu(q_vals, cpu_marker=1)
            else:    
                play_against_cpu(q_vals)
        elif match_up == '2':
            secs = float(input("Enter the delay between CPU moves in seconds (ex: 1.5): "))
            play_cpu_vs_cpu(q_vals, seconds=secs)