import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score

bh_data = load_boston()
print(bh_data.keys())

boston = pd.DataFrame(bh_data.data, columns=bh_data.feature_names)

# print(bh_data.DESCR)

boston['MEDV'] = bh_data.target

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=['LSTAT', 'RM'])
y = boston['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

lin_reg_mod = LinearRegression()
lin_reg_mod.fit(X_train, y_train)

pred = lin_reg_mod.predict(X_test)

test_set_rsme = (np.sqrt(mean_squared_error(y_test, pred)))
test_set_r2 = r2_score(y_test, pred)

# Note that for rmse, the lower that value is, the better the fit
print(f"RSME: {test_set_rsme}")
# The closer towards 1, the better the fit
print(f"R-2 score: {test_set_r2}")