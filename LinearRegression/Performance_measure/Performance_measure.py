y_true = [3,-0.5,-2,7]
y_pred = [2.5,0.0,2,8]

from sklearn.metrics import median_absolute_error
print(median_absolute_error(y_true,y_pred))


from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_true,y_pred))

from sklearn.metrics import r2_score
print(r2_score(y_true, y_pred))

import numpy as np
from sklearn.model_selection import train_test_split
X,y = np.arange(10).reshape((5,2)), range(5)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)
print("X_train : ", X_train)
print("X_test : ", X_test)
print("y_train : ", y_train)
print("y_test : ", y_test)