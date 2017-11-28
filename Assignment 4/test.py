import numpy as np
from utils import *

Q = np.full((10, 7, 4), -1.0)
Q[7, 3, :] = 0.0

print Q
