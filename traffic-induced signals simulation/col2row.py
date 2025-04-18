# Ensures correct row/column shape for matrix operations

import numpy as np

def col2row(s, flag):
    l = s.shape[0]
    m = s.shape[1]
    if flag == 1:
        if l == 1:
            s = s.T
            return s
    else:
        if m == 1:
            s = s.T
            return s
