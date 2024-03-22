from sudoku import Sudoku
import random
import numpy as np
from typing import List
import sys
from collections import Counter


#*****CONSTANST*****
# for program
BLOCK_SIZE = 3
WIDHT = 3
HEIGHT = 3

#for random
SUDOKU_SEED = 100
RND_SEED = 42
#*****CONSTANST*****

random.seed(RND_SEED) 


# TODO delete
# puzzle = Sudoku(3,3).difficulty(0.5)
# # puzzle.show()
# print(puzzle.solve())
# print(puzzle.solve().board)
# print(puzzle.solve())
# arr = [[3, 6, 7, None, None, 4, 5, 1, None], [2, 9, 4, None, 1, None, 8, None, 3], [1, None, 8, None, None, None, 6, None, None], [None, None, None, 8, None, 9, 2, 3, None], [5, 7, 3, None, None, None, None, 6, None], [8, None, 9, None, 7, 3, 1, None, None], [7, None, None, 4, None, None, 3, 9, None], [None, 4, 2, 9, 3, 8, 7, None, 1], [None, 3, None, None, None, 1, None, None, None]]
# for i in range(len(arr)):
#     for j in range(len(arr[i])):
#         if arr[i][j] is None:
#             arr[i][j] = 0
# b = puzzle.solve().board
# for i in b:
#     print(i)

def prettyPrint2d(matrix):
    for row in matrix:
        print(row)

        #Original taken from Sudoku lib, changed indexing from self.
def printBoard(TASK_BOARD):
        width = 3 # todo get dimensions from TASK_BOARD
        height = 3
        cell_length = 1
        format_int = '{0:0' + str(cell_length) + 'd}'
        for i, row in enumerate(TASK_BOARD):
            if i == 0:
                print(('+-' + '-' * (cell_length + 1) *width) * height + '+')
            print((('| ' + '{} ' * width) * height + '|').format(*[format_int.format(x) if x != 0 else ' ' * cell_length for x in row]))
            if i == width * height - 1 or i % height == height - 1:
                print(('+-' + '-' * (cell_length + 1) * width) * height + '+')
board = [
    [6, 8, 9, 2, 3, 1, 7, 4, 5],
    [2, 5, 1, 4, 6, 7, 9, 3, 8],
    [3, 7, 4, 9, 5, 8, 1, 2, 6],
    [9, 3, 8, 5, 7, 6, 2, 1, 4],
    [1, 6, 2, 8, 4, 9, 3, 5, 7],
    [5, 4, 7, 1, 2, 3, 8, 6, 9],
    [8, 1, 3, 6, 9, 4, 5, 7, 2],
    [4, 9, 5, 7, 1, 2, 6, 8, 3],
    [7, 2, 6, 3, 8, 5, 4, 9, 1]
]

# hard_board = [
#     [0,0,7,0,4,0,0,0,0],
#     [0,0,0,0,0,8,0,0,6],
#     [0,4,1,0,0,0,9,0,0],
#     [0,0,0,0,0,0,1,7,0],
#     [0,0,0,0,0,6,0,0,0],
#     [0,0,8,7,0,0,2,0,0],
#     [3,0,0,0,0,0,0,0,0],
#     [0,0,0,1,2,0,0,0,0],
#     [8,6,0,0,7,6,0,0,5]
# ]
med_board = [
    [3, 6, 7, 0, 0, 4, 5, 1, 0],
    [2, 9, 4, 0, 1, 0, 8, 0, 3],
    [1, 0, 8, 0, 0, 0, 6, 0, 0],
    [0, 0, 0, 8, 0, 9, 2, 3, 0],
    [5, 7, 3, 0, 0, 0, 0, 6, 0],
    [8, 0, 9, 0, 7, 3, 1, 0, 0],
    [7, 0, 0, 4, 0, 0, 3, 9, 0],
    [0, 4, 2, 9, 3, 8, 7, 0, 1],
    [0, 3, 0, 0, 0, 1, 0, 0, 0],
]


def fillBlock(taskBlock):
    size = len(taskBlock)
    allNumbers = list(range(1, size**2 + 1)) # list of numbers [1..size+1]
    # remove numbers that are in the task
    for row in taskBlock:
        for i in row:
            if i in allNumbers:
                allNumbers.remove(i)
    # shuffle the numbers 
    random.shuffle(allNumbers)

    # create and empty block with zeroes
    oneHotBlock = [[0] * size for _ in range(size)] # one hot encoding if a field is Task or Solution
    # insert the numbers them into the block
    numIdx = 0
    for j in range(size):
        for k in range(size):
            if taskBlock[j][k] == 0 or taskBlock[j][k] == None: # TODO delete one based on final matrix format (if 0 or None is used)
                taskBlock[j][k] = allNumbers[numIdx]
                oneHotBlock[j][k] = 1
                numIdx += 1
    return taskBlock, oneHotBlock


def createSolutionBoard(TASK_BOARD: List[List[int]]):
    board = []
    oneHot = []
    for row in range(0, BLOCK_SIZE * WIDHT, BLOCK_SIZE):
        rowOfBlocks = [[] for _ in range(WIDHT)]
        rowOfoneHot = [[] for _ in range(WIDHT)]
        for col in range(0, BLOCK_SIZE * HEIGHT, BLOCK_SIZE):
            sl = [Row[col:col + BLOCK_SIZE] for Row in TASK_BOARD[row:row + BLOCK_SIZE]] # slice one block
            solutionBlock = fillBlock(sl)
            for i in range (BLOCK_SIZE):
                rowOfBlocks[i].extend(solutionBlock[0][i])
                rowOfoneHot[i].extend(solutionBlock[1][i])
        board.extend(rowOfBlocks)
        oneHot.extend(rowOfoneHot)
    return board, oneHot



# block indexes start at the leftmost highest item of the block (Block 2 wloud have indexes [0,3] (first row fourth column of the metrix))
def getBlockIdxs(blockIdx:int) -> tuple[int, int]:
    rowIdx = (blockIdx // BLOCK_SIZE) * BLOCK_SIZE #  double slash is integer division
    colIdx = (blockIdx % BLOCK_SIZE) * BLOCK_SIZE
    return rowIdx, colIdx

# get cnt of how many cells are movable (not part of Task matrix) in each given block
# also get the indexes of movable cells (in relation to board)
def movable(oneHot: List[List[int]]) -> tuple[List[int], List[tuple[int, int]]]:    
    movableCnt = []
    movableIdx = []
    for row in range(0, BLOCK_SIZE * WIDHT, BLOCK_SIZE):
        for col in range(0, BLOCK_SIZE * HEIGHT, BLOCK_SIZE):
            # for each block
            #  count movable
            movableCntBlock = 0
            #  and get indexes of movables
            indexes = []
            for j in range(BLOCK_SIZE):
                for k in range(BLOCK_SIZE):
                    if oneHot[row + j][col + k] == 1:
                        movableCntBlock += 1
                        indexes.append((j + row, k + col))
            movableCnt.append(movableCntBlock)
            movableIdx.append(indexes)
    return movableCnt, movableIdx

def switch(board: List[List[int]], fst:tuple[int, int], snd:tuple[int, int]):
    tmp = board [fst[0]][fst[1]]
    board [fst[0]][fst[1]] = board [snd[0]][snd[1]]
    board [snd[0]][snd[1]] = tmp
    


def evalBoard(board: List[List[int]], oneHot: List[List[int]]): # TODO will oneHot be used in eval func?
    print("---EVAL---")
    conflictCnt = 0
    rowLen = BLOCK_SIZE * WIDHT
    colLen = BLOCK_SIZE * HEIGHT
    for r in board:
        counter = Counter(r)
        conflictCnt += rowLen - len(counter)

    # print([board[i][0:colLen] for i in range(0,3)])
    for colIdx in range(colLen):
        col = [row[colIdx:colIdx + 1][0] for row in board]
        counter = Counter(col)
        conflictCnt += colLen - len(counter)
        # print(col)

    print("Eval: ", conflictCnt)
    return conflictCnt


task_board = board
board, oneHot = createSolutionBoard(task_board)

printTables = False
if printTables:
    print("TASK_BOARD")
    printBoard(task_board)
    print("solution board")
    printBoard(board)
    print("One Hot Encoding")
    printBoard(oneHot)



ITER = 1
temp = 0.5

printBoard(board)
m1 = ()
switch(board, )
oneHot[0][5] = 1

printBoard(board)
bestScore = evalBoard(board, oneHot)
score = bestScore
exit()
#**** SA alg ******
# goal is to have eval == 0
GOAL = 0
BLOCK_CNT = 9

for i in range(ITER):

    stuckCnt = 0
    #* SAstep
    foundStep = False
    blocksSearched = []
    while not foundStep:
        #select random block that was not yet searched in this step
        blockIdx = random.choice([num for num in range(0, BLOCK_CNT) if num not in blocksSearched])
        # printBoard(oneHot)
        print(blockIdx + 1)
        
        rowIdx, colIdx = getBlockIdxs(blockIdx)
        movables = movableIdx[blockIdx]
        print(movables)
        # keep randomly selecting movables until all were searched
        while len(movables) > 1:
            m = random.choice(movables)
            movables.remove(m)
            # select another and try to switch them
            neighbour = random.choice(movables)
            print(m , "  ", neighbour)
            # printBoard(board)
            switch(board, m, neighbour)
            # print("-------SWITCH-------")
            eval = evalBoard(board, oneHot)
            if eval < score:
                foundStep = True
                break
            elif temp >= 1: #TODO temp
                foundStep = True
                break
            # switch back and continue search                
            # printBoard(board)
            switch(board, m, neighbour)
            # print("-------SWITCH2-------")
            # printBoard(board)

        # did not find step in this block
        if len(movables) <= 1:            
            print("could not find")
            blocksSearched.append(blockIdx)
            if(len(blocksSearched) == BLOCK_CNT):
                print("GOT STUCK!")
                print(blocksSearched)
                exit()



        # exit()
    # found step
    print("Found step")
    score = evalBoard(board, oneHot)
    if score == GOAL:
        print("Goal reached, TODO") # todo
        exit()

        
    

