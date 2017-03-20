import os

work_test ="/home/darid/ResearchWork/DataCollection/Data/Test"
i = 1
for t_f in os.listdir(work_test):
    print i, t_f
    i+=1