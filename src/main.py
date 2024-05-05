from sudoku import Sudoku
import random
import numpy as np
from typing import List
import sys
from collections import Counter
import matplotlib.pyplot as plt
from enum import Enum
import os


#*****CONSTANST*****
# for program
MAX_STUCK_COUNT = 1000
TASK_ONE_HOT = 1
BOARD_ONE_HOT = 0
VERBOSE = False
COOLING_LINEAR = 0
COOLING_GEOMETRIC = 1
GOAL = 0 # goal is to have eval == 0

#for random
RND_SEED = 40
#*****CONSTANST*****

#****GLOBAL VARIABLes***
######RUN########
DIFFICULTY = 0.4
WIDTH = 4
HEIGHT= 4
ITER = 5000
CONFLICT_WITH_TASK_CELL_COST = 10
TEMP_INITIAL = 5
TEMP_LOSS = 0.9
TEMP_REHEAT = 2
TEMP_LOW_THR = 0.5
# random.seed(RND_SEED) 
# np.random.seed(RND_SEED)


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
        print((('| ' + '{} ' * WIDTH) * HEIGHT + '|').format(*[format_int.format(x) if x != 0  else ' ' * cell_length for x in row]))
        if i == width * height - 1 or i % height == height - 1:
            print(('+-' + '-' * (cell_length + 1) * width) * height + '+')


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
    oneHot = np.zeros_like(TASK_BOARD)
    oneHot[TASK_BOARD != 0] = TASK_ONE_HOT # if number is not zero, set ohe hot to one
    boardSize = HEIGHT * WIDTH
    numbersInBlock = HEIGHT * WIDTH # because of sudoku rules, boardSize == numbersInBlock

    allNumbers = np.array(range(1, numbersInBlock + 1)) # list of numbers in one cell

    board = np.zeros((boardSize, boardSize), dtype=np.int32)
    # indexing may seem weird, but it is the way tha pysudoku generates boards
    for row in range(0, boardSize, HEIGHT):  # step size of rows is equal to HEIGHT
        for col in range(0, boardSize, WIDTH): # step size of columns is equal to WIDTH
            taskBlock = TASK_BOARD[row:(row + HEIGHT), col:(col + WIDTH)]
            remainingNumbers = np.setdiff1d(allNumbers, taskBlock.flatten())
            np.random.shuffle(remainingNumbers)
            for rowBlock in range (HEIGHT):
                for colBlock in range (WIDTH):
                    if (oneHot[row + rowBlock, col + colBlock] != 0): # copy TASK value
                        # board[row + rowBlock, col + colBlock] = taskBlock [rowBlock, colBlock]
                        x = 0
                    else: #set value to one of remaining numbers
                        board[row + rowBlock, col + colBlock] = remainingNumbers[0]
                        remainingNumbers = remainingNumbers [1:]
    return np.array(board), np.array(oneHot)




# get cnt of how many cells are movable (not part of Task matrix) in each given block
# also get the indexes of movable cells (in relation to board)
def movable(oneHot: np.ndarray) -> tuple[List[int], List[tuple[int, int]]]:
    movableBlockIdx = []
    movableElementIdx = []
    boardSize = HEIGHT * WIDTH

    for row in range(0, WIDTH): 
        for col in range(0, HEIGHT):
            blockRowIdx = row * HEIGHT
            blockColIdx = col * WIDTH
            block = oneHot[blockRowIdx:(blockRowIdx + HEIGHT), blockColIdx:(blockColIdx + WIDTH)]
            movableInBlockCnt = block[block == BOARD_ONE_HOT].size
            if movableInBlockCnt >= 2:
                movableBlockIdx.append(row * HEIGHT + col)
                blockIndexes = np.argwhere(block == BOARD_ONE_HOT)
                indexOffset = [blockRowIdx, blockColIdx]
                blockIndexes = blockIndexes + indexOffset
                movableElementIdx.append(blockIndexes)
            else:
                movableElementIdx.append([])

    return movableBlockIdx, movableElementIdx

def switch(board: np.ndarray, fst:tuple[int, int], snd:tuple[int, int]):
    tmp = board [fst[0],fst[1]]
    board [fst[0],fst[1]] = board [snd[0],snd[1]]
    board [snd[0],snd[1]] = tmp
    


def evalBoard(board: np.ndarray, taskBoard: np.ndarray): 
    boardSize = HEIGHT * WIDTH
    totalConflicts = 0
    for i in range(boardSize):
        taskRow = taskBoard[i][taskBoard[i] != 0]
        boarRow = board[i][board[i] != 0]
        rowConflicts = arrayConflicts(boarRow, taskRow)
        
        taskCol = taskBoard[:, i][taskBoard[:, i] != 0]
        boardCol = board[:, i][board[:, i] != 0]
        colConflicts = arrayConflicts(taskCol, boardCol)
        totalConflicts += rowConflicts + colConflicts    
    return totalConflicts
    




def plot(samples: List[int])-> None:
    print("Plotting graph")
    x = list(range(1, len(samples) + 1))

    plt.plot(x, samples)

    # Add labels and title
    plt.xlabel('New best eval')
    plt.ylabel('Evaluaiton')
    plt.title('2D Graph of energy function')
    plt.show()


def tempChange(temp:float, coolingType):
    global COOLING_LINEAR
    global COOLING_GEOMETRIC
    match coolingType:
        case 0:
            new = temp - TEMP_LOSS
            if (new < 0):
                new = 0.002
            return new
        case 1:
            return temp * TEMP_LOSS 
    # Linear
    # newTemp = temp - TEMP_LOSS
    # return newTemp if (newTemp > 0) else 0

    return temp * TEMP_LOSS    


def arrayConflicts(boardRow: np.ndarray, taskRow: np.ndarray) -> int:
    # board X task conflicts
    z = np.isin(boardRow, taskRow) # if board element is in task row
    taskConflicts = z.sum().item() * CONFLICT_WITH_TASK_CELL_COST

    # board conflicts
    unique = np.unique(boardRow)
    a = boardRow.size
    boardConflicts = boardRow.size - unique.size
    return taskConflicts + boardConflicts

#**** SA alg ******
def SAsudoku (taskBoard: np.ndarray, ITER:int, temp, COOLING_TYPE)-> tuple[np.ndarray, List[int], bool]:
    # fill board with numbers
    board, oneHot = createSolutionBoard(taskBoard)
    boardSize = HEIGHT * WIDTH
    if VERBOSE:
        print("Filling cells:")
        printBoard(board)
    reheat = TEMP_REHEAT

    bestScore = evalBoard(board, taskBoard)
    scores = [bestScore]
    stuckCnt = 0
    movableBlockIdx, movableElementIdx = movable(oneHot) #todo movableCnt

    #start SA
    for i in range(ITER):
        foundStep = False

        #select random block 
        blockIdx = random.choice(movableBlockIdx) # select one of movable blocks
        movables = movableElementIdx[blockIdx].tolist()

        m = random.choice(movables)
        movables.remove(m)
        neighbour = random.choice(movables)
        switch(board, m, neighbour)
        eval = evalBoard(board, taskBoard)
        if eval <= bestScore: #best score
            foundStep = True
        else:
            diff = -float(eval - bestScore)
            c = np.exp((diff)/(temp))
            rnd = random.random()
            if c > rnd: 
                if (VERBOSE):
                    print(i,": Selected with Temp ", c, rnd)
                foundStep = True

        temp = tempChange(temp, COOLING_TYPE) #update temperutre
        if foundStep:# found step, exit searching
            if eval < bestScore:
                if VERBOSE:
                    print(f"{i:3d}.new best score: {eval:3d}")
                bestScore = eval
            if eval == GOAL: # hit GOAL
                scores.append(bestScore)
                if VERBOSE:
                    print(f"{i:3d}.Found goal")
                return board, scores, True
            continue

        #append to scores
        scores.append(bestScore)
        if VERBOSE and (i % 1000) == 0:
            print(i,": current eval:", eval)

        # did not find solution
        switch(board, m, neighbour)
        if temp <= TEMP_LOW_THR:
            if VERBOSE:
                print("Small temp:", temp, "  reheating")
            temp = reheat

        stuckCnt += 1
        if stuckCnt == MAX_STUCK_COUNT: # TODO
            if VERBOSE:
                print("Stuck, restarting with added heat..")
            reheat += reheat
            if (reheat > TEMP_INITIAL):
                reheat = TEMP_INITIAL / 2
            temp = reheat
            stuckCnt = 0
        
    if VERBOSE:                            
        print(temp)       
        print("Reached maximum number of Iterations:", ITER)
    return board, scores, False

def getConfigString(COOLING_TYPE, testCnt):
    colling = 'Linear' if COOLING_TYPE == 0 else 'Geometric'
    configString = f'''DIFFICULTY = {DIFFICULTY}
WIDTH = {WIDTH}
HEIGHT = {HEIGHT}
ITER = {ITER}
CONFLICT_WITH_TASK_CELL_COST = {CONFLICT_WITH_TASK_CELL_COST}
COOLING_TYPE = {colling}
TEMP_INITIAL = {TEMP_INITIAL}
TEMP_LOSS = {TEMP_LOSS}
TEMP_REHEAT = {TEMP_REHEAT}
TEMP_LOW_THR = {TEMP_LOW_THR}
TEST_COUNT = {testCnt}
'''
    return configString

def runSAsudoku(testCount, COOLING_TYPE) -> tuple[List[int]]:
    # puzzle = Sudoku(WIDTH,HEIGHT, seed=SEED).difficulty(DIFFICULTY)
    puzzle = Sudoku(WIDTH,HEIGHT).difficulty(DIFFICULTY)
    replaceNoneWithZeros(puzzle.board)
    taskBoard = np.array(puzzle.board)
    temp = TEMP_INITIAL
    
    iterations = []
    for i in range(testCount):
        _, scores, foundSolution = SAsudoku(taskBoard,  ITER, temp, COOLING_TYPE)
        iterations.append(len(scores))
        if i != 0 and ((i % 10) == 0):
            print("  ",i)
        if i == testCount - 1:
            print("  ",i + 1)

    return iterations

def main():
    COOLING_TYPE = COOLING_GEOMETRIC
    # SEED = random.randint(1, 10000)
    # random.seed(SEED)
    # np.random.seed(SEED)


    configName = 'constants'
    configValues = list(range(2))
    configCnt = len(configValues)
    results = []
    testCount = 30
    print("Config values      :", configCnt)
    print("Test count per conf:", testCount)
    for i in range(configCnt):
        print(i,": running ")
        globals()[configName] = configValues[i]
        x = configValues[i]
        results.append(runSAsudoku(testCount, COOLING_TYPE))
    
    dataFolder = 'dataFolder'
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder)

    configFolder = 'constants'
    if not os.path.exists(os.path.join(dataFolder, configFolder)):
        os.makedirs(os.path.join(dataFolder, configFolder))
    configFolderPath = os.path.join(dataFolder, configFolder)

    plt.figure(configName)
    plt.boxplot(results)

    # Set labels and title
    plt.xlabel('Value')
    plt.ylabel('Iteration')
    plt.title(configName)



    # Save the plot to a file in the subfolder
    plt.savefig(os.path.join(configFolderPath, configName + '.png'))
    text_file = os.path.join(configFolderPath, configName +'.txt')
    with open(text_file, 'w') as f:
        f.write(getConfigString(COOLING_TYPE, testCount))
    # Show the plot
    plt.show()
    print(getConfigString(COOLING_TYPE, testCount))
    
    return

main()