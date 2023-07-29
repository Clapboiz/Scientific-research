import numpy as np
import pandas as pd

A = np.array([[3,8,7],
              [6,8,6],
              [3,2,1]])
AA= pd.DataFrame(A,
                 index=["T2","T3", "T4"],
                 columns=["keo", "banh", "sua"])
prices = np.array([[10,20,30]])
p=pd.DataFrame(prices,
               index = ["price"],
               columns=["keo", "banh", "sua"])
total = AA.dot(p.T)
print(total)
print (AA.shape)
print(p.shape)