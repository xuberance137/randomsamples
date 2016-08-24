
# From the twitch interview - 23AUG2016

import numpy as np

board = [
  [1, 10, 3, 8],
  [12, 2, 9, 6],
  [5, 7, 4, 11],
  [3, 7, 16, 5]
]

def max_gifts(board):
    
    sum_board = [[0 for col in range(len(board))] for row in range(len(board[0]))]
    path = []
    #initializing values
    sum_board[0] = board[0] 
    path.append((0,0))
    for row in range(len(board)-1):
        sum_board[row+1][0] = sum_board[row][0] + board[row+1][0]
    for col in range(len(board[0]) -1):
        sum_board[0][col+1] = sum_board[0][col] + board[0][col+1]
  
    for row in range(len(board)-1):
        for col in range(len(board[0]) -1):
            if sum_board[row+1][col] > sum_board[row][col+1] :
              path.append((row+1, col))
            elif sum_board[row][col+1] > sum_board[row+1][col]:
              path.append((row, col+1))
            else:
              path.append((row+1, col)) #go row major on path if symmetric top left submatrix
            sum_board[row+1][col+1] = board[row+1][col+1] + max(sum_board[row+1][col], sum_board[row][col+1])

    print  
    print(sum_board)
    print
    print(sum_board[len(sum_board) - 1][len(sum_board[0]) - 1])
    print 
    print(path)


N = 15
# X = np.random.random((N,N))
X = np.random.randint(1000, size=(N,N))
print X

max_gifts(X)

max_gifts(board)


