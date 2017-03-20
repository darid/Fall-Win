
import re
import os


# def main():
#     args = sys.argv[1:]
#
#     if not args:
#         print 'usage: [--summaryfile] file [file ...]'
#         sys.exit(1)
#
#         ## Notice the summary flag and remove it from args if it is present.
#         #  summary = False
#         #  if args[0] == '--summaryfile':
#         #      summary = True
#         #      del args[0]
#
#     filename = args[0]
#     data = open(filename, "r")
#     listData = data.readlines()
#     fileNameMatch = re.search(r"^\w*", filename)
#     if fileNameMatch:
#         objectFileName = fileNameMatch.group()
#     else:
#         print "Did not find!"
#     targetData = open(objectFileName + "c" + ".txt", "w")
#     cleanData(listData, targetData)
#     data.close()
#     targetData.close()


#######################################################################

def cleanData(filename,label):
    # choose file folder to store data
    os.chdir('/home/darid/ResearchWork/DataCollection/Data')
    # totalNumber = 0
    tempData = []
    with open(filename, "r") as inputData:
        listData = inputData.readlines()
        frameNum = 0
        for i in range(len(listData)):

            frameMatch = re.search('Frame\s[0-9]*', listData[i])
            if frameMatch:
                # targetData.writelines(frameMatch.group() + "\n")
                tempData.append(frameMatch.group()+"\n")
                frameNum = frameNum + 1
            # frameNumber = frameMatch.group() + "\n"
            # num = 0
            dataMatch = re.search(r'^[-]?(0\.\d+)|([1-9][0-9]*\.\d+)', listData[i])
            if dataMatch:
                # targetData.writelines(listData[i])
                tempData.append(listData[i])

    targetfile = filename.split('/')
    with open(targetfile[-6]+targetfile[-5]+targetfile[-4]+targetfile[-3]+ ".txt", "w") as outputData:
        outputData.writelines(label+"\n")
        for i in range(len(tempData)):
            outputData.writelines(tempData[i])


    return filename, frameNum


# filename = '/home/darid/ResearchWork/DataCollection/1.txt'
# #
# print cleanData(filename,label="Fall")
