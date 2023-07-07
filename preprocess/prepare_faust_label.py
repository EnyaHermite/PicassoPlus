import numpy as np
import os

num_classes = 6890
labels = np.arange(6890)

dataDir = "../data/MPI-Faust/registrations"

import os
print(os.getcwd())

np.savetxt(os.path.join(os.path.dirname(dataDir), 'match_labels.txt'), labels, fmt='%d')