import numpy as np
import pandas as pd

# test.csv chi la 1 file du lieu duoc sap xep va cach nhau boi dau ,
df = pd.read_csv('test.csv', header = None)
df=df[3]
df.to_csv('sonnh.csv')
