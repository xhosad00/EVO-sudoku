from sudoku import Sudoku
import random
import numpy as np
from typing import List
import sys
from collections import Counter
import matplotlib.pyplot as plt
from enum import Enum


#*****CONSTANST*****
# for program
BLOCK_SIZE = 3
WIDHT = 3
HEIGHT = 3

#for random
SUDOKU_SEED = 100
RND_SEED = 42
#*****CONSTANST*****

# random.seed(RND_SEED) 


def replaceNoneWithZeros(board: List[List[int]]):
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] is None:
                board[i][j] = 0

#Original taken from Sudoku lib, changed indexing from self.
def printBoard(TASK_BOARD):
        width = 3 # todo get dimensions from TASK_BOARD
        height = 3
        cell_length = 1
        format_int = '{0:0' + str(cell_length) + 'd}'
        for i, row in enumerate(TASK_BOARD):
            if i == 0:
                print(('+-' + '-' * (cell_length + 1) *width) * height + '+')

            print((('| ' + '{} ' * width) * height + '|').format(*[format_int.format(x) if x != 0  else ' ' * cell_length for x in row]))
            if i == width * height - 1 or i % height == height - 1:
                print(('+-' + '-' * (cell_length + 1) * width) * height + '+')


filled_board = [
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

easy_board = [
    [4, 7, 8, 5, 0, 6, 1, 2, 3],
    [2, 5, 1, 0, 7, 8, 4, 0, 9],
    [3, 0, 6, 1, 2, 4, 5, 8, 7],
    [5, 3, 2, 0, 0, 1, 7, 4, 8],
    [7, 6, 4, 8, 5, 2, 3, 9, 1],
    [1, 8, 9, 0, 0, 0, 2, 5, 6],
    [6, 0, 3, 0, 0, 9, 0, 7, 5],
    [8, 2, 5, 0, 3, 7, 9, 1, 4],
    [9, 0, 7, 4, 8, 5, 6, 3, 0]
]

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

hard_board = [
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



# TODo unused
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
    conflictCnt = 0
    rowLen = BLOCK_SIZE * WIDHT
    colLen = BLOCK_SIZE * HEIGHT
    if True:
        for r in board:
            counter = Counter(r)
            conflictCnt += rowLen - len(counter)

        for colIdx in range(colLen):
            col = [row[colIdx:colIdx + 1][0] for row in board]
            counter = Counter(col)
            conflictCnt += colLen - len(counter)

        # print("Eval: ", conflictCnt)
        return conflictCnt
    CONFLICT_WITH_TASK_CELL_COST = 1
    for r in range(len(board)): # each row
        for c in range(len(board[0])):
            for second in range(c + 1, len(board[0]) - 1):
                if board[r][c] == board[r][second]: #Conflict
                    if oneHot[r][c] == oneHot[r][second]: # both are movable
                        conflictCnt += 1
                    else:
                        conflictCnt += 1 * CONFLICT_WITH_TASK_CELL_COST

                
    for c in range(len(board[0])):
        for r in range(len(board)):
            for second in range(r + 1, len(board) - 1):
                if board[r][c] == board[second][c]: #Conflict
                    if oneHot[r][c] == oneHot[second][c]: # both are movable
                        conflictCnt += 1
                    else:
                        conflictCnt += 1 * CONFLICT_WITH_TASK_CELL_COST

    # print("Eval: ", conflictCnt)
    return conflictCnt
    




def plot(samples: List[int])-> None:
    print("Plotting graph")
    x = list(range(1, len(samples) + 1))

    plt.plot(x, samples, marker='o', linestyle='-')

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Evaluaiton')
    plt.title('2D Graph of energy function')
    plt.show()


def tempChange(temp:float):
    # Linear
    # newTemp = temp - TEMP_LOSS

    newTemp = temp * TEMP_LOSS
    return newTemp if (newTemp > 0) else 0



# printTables = False
# if printTables:
#     print("TASK_BOARD")
#     printBoard(task_board)
#     print("solution board")
#     printBoard(board)
#     print("One Hot Encoding")
#     printBoard(oneHot)

class SAReturn(Enum):
    REACHED_GOAL = 1
    STUCK = 2
    MAX_ITER = 3
    


#**** SA alg ******
def SAsudoku (task_board: List[List[int]], ITER:int, temp)-> tuple[List[List[int]], List[int], bool]:
    # fill board with numbers
    board, oneHot = createSolutionBoard(task_board)
    print("Filling cells:")
    printBoard(board)

    bestScore = evalBoard(board, oneHot)
    scores = [bestScore]
    #start SA
    for i in range(ITER):
        foundStep = False
        blocksSearched = []
        while not foundStep:
            #select random block that was not yet searched in this step
            blockIdx = random.choice([num for num in range(0, BLOCK_CNT) if num not in blocksSearched])
            # printBoard(oneHot)
            movableCnt, movableIdx = movable(oneHot) #todo movableCnt            
            movables = movableIdx[blockIdx]
            # keep randomly selecting movables until all were searched
            while len(movables) > 1:
                m = random.choice(movables)
                movables.remove(m)
                neighbour = random.choice(movables)
                switch(board, m, neighbour)
                eval = evalBoard(board, oneHot)
                if eval < bestScore: #best score
                    foundStep = True
                else:
                    diff = -float(eval - bestScore)
                    c = np.exp((diff)/(temp))
                    rnd = random.random()
                    if c > rnd: 
                        # print("Selected with Temp ", c, rnd)
                        foundStep = True

                if foundStep:# found step, exit searching
                    scores.append(eval)
                    if eval < bestScore:
                        print(f"{i:3d}.new best score: {eval:3d}")
                        bestScore = eval
                    break

                switch(board, m, neighbour)
            
            temp = tempChange(temp) #update temperutre
            if foundStep:   # found step
                # print("Found step")
                if scores[-1] == GOAL: # hit GOAL
                    return board, scores, bestScore, SAReturn.REACHED_GOAL
                
            # did not find step in this block
            elif len(movables) <= 1:            
                # print("could not find")
                blocksSearched.append(blockIdx)
                if(len(blocksSearched) == BLOCK_CNT):
                    print(f"{i:3d} GOT STUCK!")
                    return board, scores, bestScore, SAReturn.STUCK
                    
    print("Reached maximum number of Iterations:", ITER)
    return board, scores, bestScore, SAReturn.MAX_ITER

######RUN########
GENERATE_BOARD = False
DIFFICULTY = 0.3
task_board = easy_board 
if GENERATE_BOARD:
    task_board = Sudoku(3,3).difficulty(DIFFICULTY).board    
replaceNoneWithZeros(task_board)


ITER = 100
temp = 5
GOAL = 0 # goal is to have eval == 0
BLOCK_CNT = 9
TEMP_LOSS = 0.95


print("Solving sudoku:")
printBoard(task_board)

board, scores, bestScore, foundSolution = SAsudoku(task_board,  ITER, temp)

match foundSolution:
    case SAReturn.REACHED_GOAL:    
        print("Goal reached! showing solution") # todo
        printBoard(board)
    case SAReturn.MAX_ITER:
        print("Goal not reached due to max iterations:", ITER) 
    case SAReturn.STUCK:
        print("Goal not reached due beeing stuck") 
        printBoard(board)
# plot(scores)
exit(0)