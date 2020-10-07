# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 21:39:08 2020

@author: shubhs15
"""

# PERFORMING REGRESSION ALGORITHMS

# Splitting train house dataset to X_train and y_train
X_train = housefinal_train.drop(['SalePrice'], axis=1)
y_train = housefinal_train['SalePrice']

# Linear Regression algorithm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

house_model_linear=LinearRegression().fit(X_train,y_train)
house_model_linear.score(X_train,y_train) # 0.84993337035374

# Since R-square is more than 0.7, which is a good fit model. Thus, the model is accepted

y_pred_linear = house_model_linear.predict(housefinal_test)
print(y_pred_linear)

mean_squared_error(y_train,y_pred_linear)
# ValueError: Found input variables with inconsistent numbers of samples: [1460, 1459]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# RandomForest Regression algorithm - Default
from sklearn.ensemble import RandomForestRegressor
house_model = RandomForestRegressor(random_state=10)
house_model.fit(X_train,y_train)
y_pred = house_model.predict(housefinal_test)
print(y_pred)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# RandomForest Regression algorithm with estimators and oob_score parameters
house_model_modify = RandomForestRegressor(n_estimators=50,oob_score=True,random_state=10)
house_model_modify.fit(X_train,y_train)
y_predict = house_model_modify.predict(housefinal_test)
print(y_predict)
print(house_model_modify.oob_score_) # 0.850320481000463
price_error_rate = 1-house_model_modify.oob_score_
print(price_error_rate) # 0.14967951899953702
# Since error rate is less than 0.3, thus we accept the model

predicted = pd.DataFrame(y_predict)
# submission_df = pd.read_csv('sample_submission.csv')
# housesubmit_data = pd.concat([submission_df['Id'],predicted],axis=1)
# housesubmit_data.columns = ['Id','SalePrice']
# housesubmit_data.to_csv('shubhs_submission.csv', index=False)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# XGBoost Regression algorithm
import xgboost
houseboost = xgboost.XGBRegressor()
houseboost.fit(X_train,y_train)
y_predboost = houseboost.predict(housefinal_test)
print(y_predboost)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# Hyper Parameter Optimization

import xgboost
regressor = xgboost.XGBRegressor()

n_estimators = [100,250,500,750,1000,1100,1250,1500]
max_depth = [2,4,5,7,10,12,15,17]
booster = ['gbtree','gblinear']
base_score = [0.2, 0.4, 0.5,0.75,1.0]
learning_rate = [0.05,0.75,0.1,0.15,0.17,0.2]
min_child_weight = [1,2,3,4,5]

# Define grid of searching parameters for Hyperparameter
hyperparameter_grid = {
   'n_estimators' : n_estimators,
   'max_depth' : max_depth,
   'booster' : booster,
   'base_score' : base_score,
   'learning_rate' : learning_rate,
   'min_child_weight' : min_child_weight
   }

# 4-fold cross validation to set up random search
from sklearn.model_selection import RandomizedSearchCV
random_cv = RandomizedSearchCV(estimator = regressor,
                               param_distributions = hyperparameter_grid,
                               cv = 5, n_iter = 75,
                               scoring = 'neg_mean_absolute_error', n_jobs = 4,
                               verbose = 5, return_train_score = True,
                               random_state = 45)
random_cv.fit(X_train,y_train)

random_cv.best_estimator_
# XGBRegressor(base_score=1.0, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.05, max_delta_step=0, max_depth=2,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=1500, n_jobs=0, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)

regressor = xgboost.XGBRegressor(base_score=1.0, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.05, max_delta_step=0, max_depth=2,
             min_child_weight=1, missing=None, monotone_constraints='()',
             n_estimators=1500, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)

house_bgbmodel = regressor.fit(X_train,y_train)

y_hypred = regressor.predict(housefinal_test)
print(y_hypred)

from sklearn.metrics import accuracy_score

accuracy_score = accuracy_score(y_train,y_hypred)
print(accuracy_score)
#  ValueError: Found input variables with inconsistent numbers of samples: [1460, 1459]

# hy_predicted = pd.DataFrame(y_hypred)
# submission_hy_df = pd.read_csv('sample_submission.csv')
# housesubmit_hy_data = pd.concat([submission_hy_df['Id'],hy_predicted],axis=1)
# housesubmit_hy_data.columns = ['Id','SalePrice']
# housesubmit_hy_data.to_csv('shubhs_hyp_submission.csv', index=False)