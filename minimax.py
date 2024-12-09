# Minimax Algorithm with Interactive Tic-Tac-Toe Game
def print_board(board):
    for row in board:
        print(' | '.join(row))
        print("-" * 5)

# Function to evaluate the board
def evaluate(board):
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] != ' ':
            return 10 if board[row][0] == 'X' else -10
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != ' ':
            return 10 if board[0][col] == 'X' else -10
    if board[0][0] == board[1][1] == board[2][2] != ' ':
        return 10 if board[0][0] == 'X' else -10
    if board[0][2] == board[1][1] == board[2][0] != ' ':
        return 10 if board[0][2] == 'X' else -10
    return 0

# Minimax function
def minimax(board, depth, is_max):
    score = evaluate(board)
    if score != 0: return score
    if all(board[i][j] != ' ' for i in range(3) for j in range(3)): return 0
    
    if is_max:
        best = -float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    best = max(best, minimax(board, depth + 1, False))
                    board[i][j] = ' '
        return best
    else:
        best = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    best = min(best, minimax(board, depth + 1, True))
                    board[i][j] = ' '
        return best

# Function to find the best move for the computer
def find_best_move(board):
    best_val = -float('inf')
    best_move = (-1, -1)
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'X'
                move_val = minimax(board, 0, False)
                board[i][j] = ' '
                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val
    return best_move

# Main game loop
def tic_tac_toe():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    print("Welcome to Tic-Tac-Toe! You are 'O', Computer is 'X'.")
    print_board(board)
    
    for _ in range(9):
        # Player's move
        row, col = map(int, input("Enter your move (row and column: 0, 1, 2): ").split())
        if board[row][col] == ' ':
            board[row][col] = 'O'
        else:
            print("Invalid move! Try again.")
            continue
        
        print_board(board)
        
        # Check for player win or draw
        if evaluate(board) == -10:
            print("You win!")
            break
        if all(board[i][j] != ' ' for i in range(3) for j in range(3)):
            print("It's a draw!")
            break
        
        # Computer's move
        print("Computer's move:")
        best_move = find_best_move(board)
        board[best_move[0]][best_move[1]] = 'X'
        print_board(board)
        
        # Check for computer win
        if evaluate(board) == 10:
            print("Computer wins!")
            break
    else:
        print("It's a draw!")

# Run the game
tic_tac_toe()