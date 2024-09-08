import cv2
import numpy as np
import math

def calcOptFlowOfBlocks(mag, angle, grayImg):
    '''Takes an image (gray scale) and a flow matrix as input. Divides image into blocks and calculates Optical Flow of each block'''
    
    rows = grayImg.shape[0]
    cols = grayImg.shape[1]
    noOfRowInBlock = 20
    noOfColInBlock = 20
    
    # Calculate the number of blocks and convert to integer
    xBlockSize = int(rows / noOfRowInBlock)
    yBlockSize = int(cols / noOfColInBlock)
    
    '''To calculate the optical flow of each block'''
    
    '''Declare an array initialized to 0 of the size of the number of blocks'''
    opFlowOfBlocks = np.zeros((xBlockSize, yBlockSize, 2))
    
    for index, value in np.ndenumerate(mag):
        block_x = index[0] // noOfRowInBlock
        block_y = index[1] // noOfColInBlock
        opFlowOfBlocks[block_x, block_y, 0] += mag[index[0], index[1]]
        opFlowOfBlocks[block_x, block_y, 1] += angle[index[0], index[1]]

    centreOfBlocks = np.zeros((xBlockSize, yBlockSize, 2))
    for index, value in np.ndenumerate(opFlowOfBlocks):
        block_x = index[0]
        block_y = index[1]
        opFlowOfBlocks[block_x, block_y, 0] = float(opFlowOfBlocks[block_x, block_y, 0]) / (noOfRowInBlock * noOfColInBlock)
        val = opFlowOfBlocks[block_x, block_y, 1]
        
        if index[2] == 1:
            angInDeg = math.degrees(val)
            if angInDeg > 337.5:
                k = 0
            else:
                q = angInDeg // 22.5
                a1 = q * 22.5
                q1 = angInDeg - a1
                a2 = (q + 2) * 22.5
                q2 = a2 - angInDeg
                if q1 < q2:
                    k = int(round(a1 / 45))
                else:
                    k = int(round(a2 / 45))
            opFlowOfBlocks[block_x, block_y, 1] = k
        
        if index[2] == 0:
            r = val
            x = ((block_x + 1) * noOfRowInBlock) - (noOfRowInBlock / 2)
            y = ((block_y + 1) * noOfColInBlock) - (noOfColInBlock / 2)
            centreOfBlocks[block_x, block_y, 0] = x
            centreOfBlocks[block_x, block_y, 1] = y

    return opFlowOfBlocks, noOfRowInBlock, noOfColInBlock, noOfRowInBlock * noOfColInBlock, centreOfBlocks, xBlockSize, yBlockSize
