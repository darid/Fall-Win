import re

def CalCom(sourceList):

############### read from file
    # x = []
    # y = []
    # z = []
    #
    # for i in range(0, len(sourceList)):
    #     # read all the file into a list
    #     firstLine = sourceList[i]
    #     # read line
    #     p = re.compile('\s+')
    #     fLine = p.split(firstLine)
    #     x.append(float(fLine[0]))
    #     y.append(float(fLine[1]))
    #     z.append(float(fLine[2]))
###############################################

    x = sourceList[0:24]
    y = sourceList[25:49]
    z = sourceList[50:74]

    arrayCom = []

    xHeadCom = x[3]
    yHeadCom = y[3]
    zHeadCom = z[3]

    xChestCom = 0.5 * x[20] + 0.5 * x[0]
    yChestCom = 0.5 * y[20] + 0.5 * y[0]
    zChestCom = 0.5 * z[20] + 0.5 * z[0]

    xLeftArmCom = 0.436 * x[4] + 0.564 * x[5]
    yLeftArmCom = 0.436 * y[4] + 0.564 * y[5]
    zLeftArmCom = 0.436 * z[4] + 0.564 * z[5]

    xRArmCom = 0.436 * x[8] + 0.564 * x[9]
    yRArmCom = 0.436 * y[8] + 0.564 * y[9]
    zRArmCom = 0.436 * z[8] + 0.564 * z[9]

    xLeftForearmCom = 0.430 * x[5] + 0.570 * x[6]
    yLeftForearmCom = 0.430 * y[5] + 0.570 * y[6]
    zLeftForearmCom = 0.430 * z[5] + 0.570 * z[6]

    xRForearmCom = 0.430 * x[9] + 0.570 * x[10]
    yRForearmCom = 0.430 * y[9] + 0.570 * y[10]
    zRForearmCom = 0.430 * z[9] + 0.570 * z[10]

    xLeftHandCom = 0.506 * x[6] + 0.494 * x[21]
    yLeftHandCom = 0.506 * y[6] + 0.494 * y[21]
    zLeftHandCom = 0.506 * z[6] + 0.494 * z[21]

    xRHandCom = 0.506 * x[10] + 0.494 * x[23]
    yRHandCom = 0.506 * y[10] + 0.494 * y[23]
    zRHandCom = 0.506 * z[10] + 0.494 * z[23]

    # xHipCom = 0.105*x[0]+ 0.895*(x[12]+x[16])
    # yHipCom = 0.105*y[0]+ 0.895*(y[12]+y[16])
    # zHipCom = 0.105*z[0]+ 0.895*(z[12]+z[16])
    #
    xHipCom = 0.105 * x[1] + 0.895 * (x[0])
    yHipCom = 0.105 * y[1] + 0.895 * (y[0])
    zHipCom = 0.105 * z[1] + 0.895 * (z[0])

    xLeftThighCom = 0.433 * x[12] + 0.567 * x[13]
    yLeftThighCom = 0.433 * y[12] + 0.567 * y[13]
    zLeftThighCom = 0.433 * z[12] + 0.567 * z[13]

    xRThighCom = 0.433 * x[16] + 0.567 * x[17]
    yRThighCom = 0.433 * y[16] + 0.567 * y[17]
    zRThighCom = 0.433 * z[16] + 0.567 * z[17]

    xLeftShankCom = 0.433 * x[13] + 0.567 * x[14]
    yLeftShankCom = 0.433 * y[13] + 0.567 * y[14]
    zLeftShankCom = 0.433 * z[13] + 0.567 * z[14]

    xRShankCom = 0.433 * x[17] + 0.567 * x[18]
    yRShankCom = 0.433 * y[17] + 0.567 * y[18]
    zRShankCom = 0.433 * z[17] + 0.567 * z[18]

    xLeftFootCom = 0.5 * x[14] + 0.5 * x[15]
    yLeftFootCom = 0.5 * y[14] + 0.5 * y[15]
    zLeftFootCom = 0.5 * z[14] + 0.5 * z[15]

    xRFootCom = 0.5 * x[18] + 0.5 * x[19]
    yRFootCom = 0.5 * y[18] + 0.5 * y[19]
    zRFootCom = 0.5 * z[18] + 0.5 * z[19]

    xCom = 0.081 * xHeadCom + 0.335 * xChestCom + 0.028 * (xLeftArmCom + xRArmCom) + \
            0.016 * (xLeftForearmCom + xRForearmCom) + 0.006 * (xLeftHandCom + xRHandCom) + \
            0.142 * xHipCom + 0.1 * (xLeftThighCom + xRShankCom) + 0.0465 * (xLeftShankCom + xRShankCom) + \
            0.0145 * (xLeftFootCom + xRFootCom)

    yCom = 0.081 * yHeadCom + 0.335 * yChestCom + 0.028 * (yLeftArmCom + yRArmCom) + \
            0.016 * (yLeftForearmCom + yRForearmCom) + 0.006 * (yLeftHandCom + yRHandCom) + \
            0.142 * yHipCom + 0.1 * (yLeftThighCom + yRShankCom) + 0.0465 * (yLeftShankCom + yRShankCom) + \
            0.0145 * (yLeftFootCom + yRFootCom)

    zCom =  0.081 * zHeadCom + 0.335 * zChestCom + 0.028 * (zLeftArmCom + zRArmCom) + \
            0.016 * (zLeftForearmCom + zRForearmCom) + 0.006 * (zLeftHandCom + zRHandCom) + \
            0.142 * zHipCom + 0.1 * (zLeftThighCom + zRShankCom) + 0.0465 * (zLeftShankCom + zRShankCom) + \
            0.0145 * (zLeftFootCom + zRFootCom)

    arrayCom = [xCom,yCom,zCom,xRThighCom,yRThighCom,zRThighCom,xLeftShankCom,yLeftFootCom, zLeftFootCom]
    # arrayCom.append[xCom]
    # arrayCom.append[yCom]
    # arrayCom.append[zCom]

    return arrayCom


# data = open('/home/darid/ResearchWork/Visualization/1c.txt', 'r')
# tempList = data.readlines()
#
# frameNum = -1
#
# for i in range(0, len(tempList)):
#     match = re.search(r'Frame\s[0-9]*', tempList[i])
#     if match:
#         frameNum += 1
#             # choose draw range of frame by setting frame setting
#         if 0 <= frameNum & frameNum < 1:
#                 # print range of frame
#             print match.group()
#             # readAndDrawData(tempList[i + 1:i + 26], ax)
#             com = CalCom(tempList[i + 1:i + 26])
#
# print com
