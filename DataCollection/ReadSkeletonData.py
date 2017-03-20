import re
import os
import time
import ClearSkeletonData


workdir = '/home/darid/ResearchWork/TST Fall detection database/'
###
# input work directory path
# ouput current path and folder lists
###
def findNextDir(path):
    os.chdir(path)
    curdir= str(os.getcwd())
    return curdir,os.listdir(curdir)



# print findNextDir(workdir)
inf_file = open('/home/darid/ResearchWork/DataCollection/Data/inf'+".txt", "w")
#
for f in os.listdir(workdir):
    pat = re.compile('Data')
    if(pat.match(f)):
        for s_f in findNextDir(workdir+f+'/')[1]:
            # os.chdir(n_p+'/'+s_f+'/')
            # nn_p = str(os.getcwd())
            t_dir = findNextDir(workdir+f+'/')[0]
            for s2_f in findNextDir(t_dir)[1]:
                t2_dir = findNextDir(t_dir)[0]+'/'+s2_f
                print s2_f
                for s3_f in findNextDir(t2_dir)[1]:
                    t3_dir = findNextDir(t2_dir)[0]+'/'+s3_f
                    for s4_f in findNextDir(t3_dir)[1]:
                        t4_dir = findNextDir(t3_dir)[0]+'/'+s4_f
                        for s5_f in findNextDir(t4_dir)[1]:
                            t5_dir = findNextDir(t4_dir)[0]+'/'+s5_f
                            for s6_f in findNextDir(t5_dir)[1]:
                                if s6_f == "Fileskeleton.txt":
                                    print "got it"
                                    inf = ClearSkeletonData.cleanData(t5_dir+'/'+s6_f,s3_f)
                                    if s2_f == "Fall":
                                        inf_file.writelines(str(inf[0])+"              "+str(inf[1])+"\n")


inf_file.close()