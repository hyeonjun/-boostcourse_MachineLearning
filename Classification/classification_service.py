# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

data_source = "https://stats.idre.ucla.edu/stat/data/binary.csv"
df = pd.read_csv(data_source)
# print(df[:5])
#    admit  gre   gpa  rank
# 0      0  380  3.61     3
# 1      1  660  3.67     3
# 2      1  800  4.00     1
# 3      1  640  3.19     4
# 4      0  520  2.93     4

y_data = df["admit"].values.reshape(-1, 1)
x_data = df.iloc[:,1:]
# print(y_data[:5])
# [[0]
#  [1]
#  [1]
#  [1]
#  [0]]

# print(x_data[:5])
#    gre   gpa  rank
# 0  380  3.61     3
# 1  660  3.67     3
# 2  800  4.00     1
# 3  640  3.19     4
# 4  520  2.93     4

from sklearn import preprocessing # Min-Max Standardzation
min_max_scaler = preprocessing.MinMaxScaler() # 스케일링
x_data = min_max_scaler.fit_transform(x_data)
# print(x_data[:5])
# [[0.27586207 0.77586207 0.66666667]
#  [0.75862069 0.81034483 0.66666667]
#  [1.         1.         0.        ]
#  [0.72413793 0.53448276 1.        ]
#  [0.51724138 0.38505747 1.        ]]

from sklearn import linear_model, datasets
logreg = linear_model.LogisticRegression(fit_intercept=True)
y = y_data.ravel()
logreg.fit(x_data, y)
sum(logreg.predict(x_data) == y) / len(y)
# 0.705

theta = np.append(logreg.intercept_, logreg.coef_.ravel())
# print(theta)
#    w0           w1          w2          w3
# [-1.50619015  1.06983642  1.11573376 -1.47695382]

theta_file_name = "theta_bin.npy"
np.save(theta_file_name, theta) # theta값 저장해야함
# print(np.load(theta_file_name))


min_max = np.vstack((df.iloc[:,1:].values.min(axis=0),
                df.iloc[:,1:].values.max(axis=0)))
# print(min_max)
# [[220.     2.26   1.  ]
#  [800.     4.     4.  ]]
min_max_file_name = "min_max.npy"
np.save(min_max_file_name, min_max)
# print(np.load(min_max_file_name))
# [[220.     2.26   1.  ]
#  [800.     4.     4.  ]]

import pickle
f = open('logicmodel.pkl', 'wb')
f.seek(0)
pickle.dump(logreg,f)
f.close()
fr = open('./logicmodel.pkl', 'rb')
logreg_pickle = pickle.load(fr)
print(sum(logreg.predict(x_data) == logreg_pickle.predict(x_data)) / len(y))
# 1.0
