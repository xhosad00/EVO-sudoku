from sudoku import Sudoku
import random
import numpy as np
from typing import List

SUDOKU_SEED = 100
RND_SEED = 42
random.seed(RND_SEED) 
# puzzle = Sudoku(3,3).difficulty(0.9999)
# puzzle.show()
# print(puzzle.solve())

VERBOSE = True
BLOCK_SIZE = 3

def prettyPrint2d(matrix):
    for row in matrix:
        print(row)

TASK_BOARD = [
    [0,0,7,0,4,0,0,0,0],
    [0,0,0,0,0,8,0,0,6],
    [0,4,1,0,0,0,9,0,0],
    [0,0,0,0,0,0,1,7,0],
    [0,0,0,0,0,6,0,0,0],
    [0,0,8,7,0,0,2,0,0],
    [3,0,0,0,0,0,0,0,0],
    [0,0,0,1,2,0,0,0,0],
    [8,6,0,0,7,6,0,0,5]
]
# prettyPrint2d(board)

puzzle = Sudoku(3, 3, board = TASK_BOARD)
puzzle.show()



# prettyPrint2d(sl)

def fillBlock(TASK_BLOCK):
    # print("Filling block:")
    # prettyPrint2d(TASK_BLOCK)
    size = len(TASK_BLOCK)
    allNumbers = list(range(1, size**2 + 1)) # list of numbers [1..size+1]
    # remove numbers that are in the task
    for row in TASK_BLOCK:
        for i in row:
            if i in allNumbers:
                allNumbers.remove(i)
    # shuffle the numbers 
    random.shuffle(allNumbers)

    # create and empty block with zeroes
    board = [[0] * size for _ in range(size)]
    # insert the numbers them into the block
    numIdx = 0
    for j in range(size):
        for k in range(size):
            if TASK_BLOCK[j][k] == 0 or TASK_BLOCK[j][k] == None: # TODO delete one based on final matrix format (if 0 or None is used)
                board[j][k] = allNumbers[numIdx]
                numIdx += 1
    
    # print("Finished:")
    # prettyPrint2d(board)
    return board


def createSolutionBlock(TASK_BOARD: List[List[int]]):
    board = []
    for row in range(3):
        rowOfBlocks = [[],[],[]]
        for col in range(3):
            colIdx = col * BLOCK_SIZE
            rowIdx = row * BLOCK_SIZE
            sl = [Row[colIdx:colIdx + BLOCK_SIZE] for Row in TASK_BOARD[rowIdx:rowIdx + BLOCK_SIZE]] # slice one block
            solutionBlock = fillBlock(sl)
            for i in range (BLOCK_SIZE):
                rowOfBlocks[i].extend(solutionBlock[i])
        board.extend(rowOfBlocks)
    return board


board = createSolutionBlock(TASK_BOARD)

print(np.shape(board))
print(np.shape(TASK_BOARD))
prettyPrint2d(board)
print()
prettyPrint2d(TASK_BOARD)


def join_matrices(matrix1, matrix2):
    result = [[matrix2[i][j] if matrix1[i][j] == 0 else matrix1[i][j] for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
    return result

# WORKED WITH THE IDEO OF TWO MATRIXES