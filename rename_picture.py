import os, sys
mypath = "/Users/yn/titan/sandbox/courses/6000B/dailycheck-master/all_baseline/breast_A2B/test_latest/images/"
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
for f in onlyfiles:
    if 'fake_B' not in f:
        continue
    print(f)
    vector_name = f.split('_')
    old_name = mypath + f
    new_name = mypath + vector_name[0] + '_' + vector_name[1] + '_' + vector_name[2] + '_'+vector_name[3] + '_'+vector_name[4]+'_'+vector_name[5]+'.png' 
    os.rename(old_name,new_name)

import os, sys
mypath = "/Users/yn/titan/sandbox/courses/6000B/dailycheck-master/all_baseline/breast_A2B/test_latest/images/"
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
for f in onlyfiles:
    if 'ANON.png'  in f:
        continue
    print(f)
    vector_name = f.split('_')
    old_name = mypath + f
    os.remove(old_name)
