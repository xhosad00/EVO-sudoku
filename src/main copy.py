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
WIDTH = 3
HEIGHT = 3
BLOCK_CNT = WIDTH * HEIGHT
PRINT_DECIMAL_LEN = 1

#for random
SUDOKU_SEED = 100
RND_SEED = 40
#*****CONSTANST*****

# random.seed(RND_SEED) 


def replaceNoneWithZeros(board: List[List[int]]):
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] is None:
                board[i][j] = 0

#Original taken from Sudoku lib, changed indexing from self.
def printBoard(TASK_BOARD):
    width = WIDTH # todo get dimensions from TASK_BOARD
    height = HEIGHT
    cell_length = 1
    if(HEIGHT > 3 or WIDTH > 3):
        cell_length = 2
    format_int = '{0:0' + str(cell_length) + 'd}'
    for i, row in enumerate(TASK_BOARD):
        if i == 0:
            print(('+-' + '-' * (cell_length + 1) *width) * height + '+')

        print((('| ' + '{} ' * width) * height + '|').format(*[format_int.format(x) if x != 0  else ' ' * cell_length for x in row]))
        if i == width * height - 1 or i % height == height - 1:
            print(('+-' + '-' * (cell_length + 1) * width) * height + '+')


# filled_board = [
#     [6, 8, 9, 2, 3, 1, 7, 4, 5],
#     [2, 5, 1, 4, 6, 7, 9, 3, 8],
#     [3, 7, 4, 9, 5, 8, 1, 2, 6],
#     [9, 3, 8, 5, 7, 6, 2, 1, 4],
#     [1, 6, 2, 8, 4, 9, 3, 5, 7],
#     [5, 4, 7, 1, 2, 3, 8, 6, 9],
#     [8, 1, 3, 6, 9, 4, 5, 7, 2],
#     [4, 9, 5, 7, 1, 2, 6, 8, 3],
#     [7, 2, 6, 3, 8, 5, 4, 9, 1]
# ]

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

# med_board = [
#     [3, 6, 7, 0, 0, 4, 5, 1, 0],
#     [2, 9, 4, 0, 1, 0, 8, 0, 3],
#     [1, 0, 8, 0, 0, 0, 6, 0, 0],
#     [0, 0, 0, 8, 0, 9, 2, 3, 0],
#     [5, 7, 3, 0, 0, 0, 0, 6, 0],
#     [8, 0, 9, 0, 7, 3, 1, 0, 0],
#     [7, 0, 0, 4, 0, 0, 3, 9, 0],
#     [0, 4, 2, 9, 3, 8, 7, 0, 1],
#     [0, 3, 0, 0, 0, 1, 0, 0, 0],
# ]

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
    for j in range(HEIGHT):
        for k in range(WIDTH):
            if taskBlock[j][k] == 0: # TODO delete one based on final matrix format (if 0 or None is used)
                taskBlock[j][k] = allNumbers[numIdx]
                oneHotBlock[j][k] = 1
                numIdx += 1
    return taskBlock, oneHotBlock


def createSolutionBoard(TASK_BOARD):
    board = []
    oneHot = []
    for row in range(0, HEIGHT * HEIGHT, HEIGHT):
        rowOfBlocks = [[] for _ in range(WIDTH)]
        rowOfoneHot = [[] for _ in range(WIDTH)]
        for col in range(0, WIDTH * WIDTH, WIDTH):
            sl = [Row[col:(col + WIDTH)] for Row in TASK_BOARD[row:(row + HEIGHT)]] # slice one block
            solutionBlock = fillBlock(sl)
            for i in range (WIDTH):
                rowOfBlocks[i].extend(solutionBlock[0][i])
                rowOfoneHot[i].extend(solutionBlock[1][i])
        board.extend(rowOfBlocks)
        oneHot.extend(rowOfoneHot)
    return np.array(board), np.array(oneHot)




# get cnt of how many cells are movable (not part of Task matrix) in each given block
# also get the indexes of movable cells (in relation to board)
def movable(oneHot: List[List[int]]) -> tuple[List[int], List[tuple[int, int]]]:    
    movableCnt = []
    movableIdx = []
    for row in range(0, HEIGHT * HEIGHT, HEIGHT):
        for col in range(0, WIDTH * WIDTH, WIDTH):
            # for each block
            #  count movable
            movableCntBlock = 0
            #  and get indexes of movables
            indexes = []
            for j in range(WIDTH):
                for k in range(HEIGHT):
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
    rowLen = WIDTH * WIDTH
    colLen = HEIGHT * HEIGHT
    if False:
        for r in board:
            counter = Counter(r)
            conflictCnt += rowLen - len(counter)

        for colIdx in range(colLen):
            col = [row[colIdx:colIdx + 1][0] for row in board]
            counter = Counter(col)
            conflictCnt += colLen - len(counter)

        # print("Eval: ", conflictCnt)
        return conflictCnt
    CONFLICT_WITH_TASK_CELL_COST = 2
    for r in range(rowLen): # each row
        for c in range(colLen - 1): #all but last
            for second in range(c + 1, colLen): # compare with all following this [r][c] in row
                if board[r][c] == board[r][second]: #Conflict
                    if oneHot[r][c] == oneHot[r][second]: # both are movable
                        conflictCnt += 1
                    else:
                        conflictCnt += 1 * CONFLICT_WITH_TASK_CELL_COST

                
    for c in range(colLen):
        for r in range(rowLen - 1):
            for second in range(r + 1, rowLen): # compare with all following this [r][c] in column
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

    plt.plot(x, samples)

    # Add labels and title
    plt.xlabel('New best eval')
    plt.ylabel('Evaluaiton')
    plt.title('2D Graph of energy function')
    plt.show()


def tempChange(temp:float):
    # Linear
    # newTemp = temp - TEMP_LOSS
    # return newTemp if (newTemp > 0) else 0

    return temp * TEMP_LOSS



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
    
TEMP_LOW_THR = 1e-3
VERBOSE = False

#**** SA alg ******
def SAsudoku (task_board, ITER:int, temp)-> tuple[List[List[int]], List[int], bool]:
    # fill board with numbers
    board, oneHot = createSolutionBoard(task_board)
    return
    if VERBOSE:
        print("Filling cells:")
        printBoard(board)

    bestScore = evalBoard(board, oneHot)
    scores = [bestScore]
    stuckCnt = 0
    #start SA
    for i in range(ITER):
        foundStep = False

        #select random block that was not yet searched in this step
        blockIdx = random.choice([num for num in range(0, BLOCK_CNT)])
        # printBoard(oneHot)
        movableCnt, movableIdx = movable(oneHot) #todo movableCnt            
        movables = movableIdx[blockIdx]
        if len(movables) <= 1: # only one movable in block
            i -= 1
            continue

        m = random.choice(movables)
        movables.remove(m)
        neighbour = random.choice(movables)
        switch(board, m, neighbour)
        eval = evalBoard(board, oneHot)
        if eval <= bestScore: #best score
            foundStep = True
        else:
            diff = -float(eval - bestScore)
            c = np.exp((diff)/(temp))
            rnd = random.random()
            if c > rnd: 
                # print("Selected with Temp ", c, rnd)
                foundStep = True

        temp = tempChange(temp) #update temperutre
        if foundStep:# found step, exit searching
            # if eval != scores[-1]: # TODO delete
            scores.append(eval)
            if eval < bestScore:
                if VERBOSE:
                    print(f"{i:3d}.new best score: {eval:3d}")
                bestScore = eval
            if scores[-1] == GOAL: # hit GOAL
                print(f"{i:3d}.Found goal")
                return board, scores, bestScore, SAReturn.REACHED_GOAL
            continue

        # did not find solution
        switch(board, m, neighbour)
        if temp <= TEMP_LOW_THR:
            if VERBOSE:
                print("Small temp:", temp, "  reheating")
            temp = 0.6 # TODO

        stuckCnt += 1
        if stuckCnt == 10000: # TODO
            if VERBOSE:
                print("Stuck, restarting..")
            board, oneHot = createSolutionBoard(task_board)
            # print("Filling cells:")
            # printBoard(board)

            bestScore = evalBoard(board, oneHot)
            scores = [bestScore]
            temp = 0.2 # TODO
            stuckCnt = 0
        
                            
    print(temp)       
    print("Reached maximum number of Iterations:", ITER)
    return board, scores, bestScore, SAReturn.MAX_ITER

######RUN########
GENERATE_BOARD = True
DIFFICULTY = 0.2
WIDTH = 2
HEIGHT= 4
task_board = np.array(easy_board)
if GENERATE_BOARD:
    puzzle = Sudoku(WIDTH,HEIGHT).difficulty(DIFFICULTY)
    puzzle.show()
    replaceNoneWithZeros(puzzle.board)
    task_board = np.array(puzzle.board)


ITER = 50000
temp = 0.4
GOAL = 0 # goal is to have eval == 0
TEMP_LOSS = 0.999


print("Solving sudoku:")
printBoard(task_board)

board, scores, bestScore, foundSolution = SAsudoku(task_board,  ITER, temp)

# match foundSolution:
#     case SAReturn.REACHED_GOAL:    
#         print("Goal reached! showing solution") # todo
#         # printBoard(board)
#     case SAReturn.MAX_ITER:
#         print("Goal not reached due to max iterations:", ITER) 
#         print("Final Eval:", scores[-1]) 
#     case SAReturn.STUCK:
#         print("Goal not reached due beeing stuck") 
#         printBoard(board)


# printBoard(board)
# plot(scores)
exit(0)