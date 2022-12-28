# import numpy as np
# import pandas as pd
# from sklearn.impute import SimpleImputer
# data = pd.read_csv('data_miss.csv', header = None)
# print(data)
# x=data.values
# imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
# imp.fit(x)
# result = imp.transform(x)
# print('ket qua la ')
# print(result)

                # covariance trong pandas
import pandas as pd
df = pd.DataFrame(
    [(1,2,1,6),
     (0,3,0,7),
     (2,0,4,3),
     (1,1,1,4)], columns = ['dogs', 'cats','bear','duck']
)

print(df)
print(df.cov())