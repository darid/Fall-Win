import re
import os
import time
import shutil

# shutil.rmtree(/home/darid/ResearchWork/TST Fall detection database/Data3/Fall/EndUpSit/2)


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

#
for f in os.listdir(workdir):
    pat = re.compile('Data')
    if(pat.match(f)):
    # if re.search('Data.',f):
    #     p = str('/home/darid/ResearchWork/TST Fall detection database/'+f+'/')
    #     os.chdir(p)
    #     n_p = str(os.getcwd())
    #     # print next_p
    #     print f
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
                            if s5_f == 'Body':
                                print s5_f+" got it"
                            else:
                                print t4_dir+'/'+s5_f+ " delete it"
                                # delete file folders
                                shutil.rmtree(t4_dir+'/'+s5_f)
