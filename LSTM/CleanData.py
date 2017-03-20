
def clean(sourceList):
    for i in range(0,len(sourceList)):
        if sourceList[i] == 9999:
            # print (i,sourceList[i])
            if i+3>=len(sourceList):
                sourceList[i] = sourceList[i-3]
            else:
                sourceList[i] = (sourceList[i-3]+sourceList[i+3])/2
            # print ('sourceList:', sourceList[i])


# sList = [1,2,3,9999,4,5,6,7]
# clean(sList)
# print (sList)