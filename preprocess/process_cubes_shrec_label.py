import numpy as np
import os
from glob import glob


#====================================CUBES===============================================
dataDir = '../data/cubes'
classnames = ['apple', 'bat', 'bell', 'brick', 'camel', 'car', 'carriage', 'chopper',
              'elephant', 'fork', 'guitar', 'hammer', 'heart', 'horseshoe', 'key',
              'lmfish', 'octopus', 'shoe', 'spoon', 'tree', 'turtle', 'watch']

for label, clsName in enumerate(classnames):
    label = np.asarray([label])
    np.savetxt(f"{dataDir}/{clsName}/label.txt", label, fmt='%d')
#=====================================END================================================


#====================================SHREC===============================================
dataDir = '../data/shrec_16'
filelist = glob(f"{dataDir}/*")
classnames = [os.path.basename(file) for file in filelist]
classnames.sort()
print(classnames, len(classnames))

for label, clsName in enumerate(classnames):
    label = np.asarray([label])
    np.savetxt(f"{dataDir}/{clsName}/label.txt", label, fmt='%d')
#=====================================END================================================
