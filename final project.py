import pandas as pd
import seaborn as sns
# read in the excel file as a dataframe
df_manhattan = pd.read_excel('manhattandatasetonly.xlsx')

# read in the first 5 rows of the dataset
df_manhattan.head()

# Use pd.get_dummies to make the neighborhoods into numerical data 
new_df_manhattan = pd.get_dummies(df_manhattan, drop_first=True)

# correlation matrix
# check for collinearity 
cm = new_df_manhattan.corr()
sns.heatmap(cm)
# no collinearity present as no coefficient correlation between two x
# variables is greater than +- 0.85

# EXPLANATORY MODELING
# multiple linear regression - find the equation model 
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
# made a new excel sheets with only the columns deemed important by 
# the feature importance plot
df_manhattan_subset = pd.read_excel('manhattansubset.xlsx')

# function to compute a regression model
def disp_regress(df, x_feat_list, y_feat, verbose=True):
    """ multiple regression, displays model w/ coef
    
    Args:
        df (pd.DataFrame): dataframe
        x_feat_list (list): list of all features in model
        y_feat (list): target feature
        verbose (bool): toggles command line output
        
    Returns:
        reg (LinearRegression): model fit to data
    """
    # initialize regression object
    reg = LinearRegression()

    # get target variable
    x = df.loc[:, x_feat_list].values
    y = df.loc[:, y_feat].values

    # fit regression
    reg.fit(x, y)

    # predict with model 
    y_pred = reg.predict(x)
    
    if verbose:
        # print model   
        model_str = y_feat + f' = {reg.intercept_:.2f}'
        for feat, coef in zip(x_feat_list, reg.coef_):
            s_sign = ' - ' if coef < 0 else ' + '
            model_str += s_sign + f'{np.abs(coef):.2f} {feat}'
        print(model_str)

    return reg

# call function to get regression equation
disp_regress(df=df_manhattan_subset, 
             x_feat_list=df_manhattan_subset.columns[1:], 
             y_feat='rent');

##############################################################################
# PREDICTIVE MODELING

# Random Forest - feature importance graph 
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# interested x and y variables
x_feat_list = new_df_manhattan.columns[2:]
y_feat = 'rent'

# get the x and y variable data as arrays
x = new_df_manhattan.loc[:, x_feat_list].values
y_true = new_df_manhattan.loc[:, y_feat].values

# initialize random forest
rf_reg = RandomForestRegressor()

# initialize k fold
skfold = KFold(shuffle=True)

# initialize y_pred, stores predictions of y 
y_pred = np.empty_like(y_true)

for train_idx, test_idx in skfold.split(x, y_true):
    # get training data
    x_train = x[train_idx, :]
    y_train = y_true[train_idx]
    
    # get test data
    x_test = x[test_idx, :]
    
    # fit data
    rf_reg = rf_reg.fit(x_train, y_train)
    
    # estimate on test data
    y_pred[test_idx] = rf_reg.predict(x_test)

# compute and print r2 score
r2 = r2_score(y_true, y_pred)
print(r2)


# Feature importance graph

import matplotlib.pyplot as plt

# function to plot feature importance graph
def plot_feat_import(feat_list, feat_import, sort=True):
    """ plots feature importances in a horizontal bar chart
    
    Args:
        feat_list (list): str names of features
        feat_import (np.array): feature importances (MSE reduce)
        sort (bool): if True, sorts features in decreasing importance
            from top to bottom of plot 
    """
    
    if sort:
        # sort features in decreasing importance
        idx = np.argsort(feat_import).astype(int)
        feat_list = [feat_list[_idx] for _idx in idx]
        feat_import = feat_import[idx] 
        
    # plot and label feature importance
    plt.barh(feat_list, feat_import)
    plt.gcf().set_size_inches(5, len(feat_list) / 2)
    plt.xlabel('Feature importance\n(Mean decrease in r2 across all Decision Trees)')

# import seaborn library 
import seaborn as sns 

# call seaborn to make the plot nicer  
sns.set()

# fit on entire dataset 
rf_reg.fit(x, y_true)

# call the plot_feat_import to plot the plot
plot_feat_import(x_feat_list, rf_reg.feature_importances_)

##############################################################################
# PREDICTIVE MODELING

# Multiple linear regression - important x-variables along with the 
# neighborhood columns
# get the desired x and y variables 
x = new_df_manhattan[new_df_manhattan.columns[9:]]
y = new_df_manhattan['rent']

# split into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=1)

# implement the linear regression model
lm = LinearRegression() 
lm.fit(x_train, y_train) 

# coefficients 
lm.coef_ 
lm.intercept_ 

# evaluate the model: using the testing dataset
predictions = lm.predict(x_test)

# to evaluate, we compare predictions with y_test (actual data)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions) 
rmse = mse ** 0.5 
print(rmse)
# rmse = 1344.21

# compute r2 score and print it out 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, predictions)
print(r2)
# 0.81 
# This is a low rmse and high r2 generated - suggesting we should include the 
# neighborhood columns in our predictive linear regression model. 


# Multiple linear regression - Using all x-variables against the y, rent price
# Includes the neighborhoods! 
# get desired x and y variables
x = new_df_manhattan[new_df_manhattan.columns[2:]]
y = new_df_manhattan['rent']

# split into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=1)

# implement the linear regression model
from sklearn.linear_model import LinearRegression 
lm = LinearRegression() 
lm.fit(x_train, y_train) 

# coefficients 
lm.coef_ 
lm.intercept_ 

# evaluate the model: using the testing dataset
predictions = lm.predict(x_test)

# to evaluate, we compare predictions with y_test (actual data)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions) 
rmse = mse ** 0.5 
print(rmse)
# rmse = 1344.55

# compute r2 score and print it out 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, predictions)
print(r2)
# r2 = 0.81

# Multiple linear regression - only using the significant x variables 
# (based on feature importance graph findings!) 
# get desired x and y variables 
x = df_manhattan_subset[df_manhattan_subset.columns[1:]]
y = df_manhattan_subset['rent']

# split into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=1)

# implement the linear regression model
from sklearn.linear_model import LinearRegression 
lm = LinearRegression() 
lm.fit(x_train, y_train) 

# coefficients 
lm.coef_ 
lm.intercept_ 

# evaluate the model: using the testing dataset
predictions = lm.predict(x_test)

# to evaluate, we compare predictions with y_test (actual data)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, predictions) 
rmse = mse ** 0.5 
print(rmse)
# rmse = 1371.25

from sklearn.metrics import r2_score
r2 = r2_score(y_test, predictions)
print(r2)
# r2 = 0.80

# From the three multiple linear regression models we made, the r2 were all 
# pretty similar. Therefore, we would go with the last multiple linear 
# regression model we made with only the top 11 most important x-variables 
# because it is the most simplistic model with the least x-variables along with
# a strong r2 score. 
#############################################################################
# k-nearest neighbors regressor 
from sklearn.neighbors import KNeighborsRegressor
from copy import copy 
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# Using x variables that are important based on feature importance graph
# x and y variables of interest 
x_feat_list = df_manhattan_subset.columns[1:]
y_feat = 'rent'

# scale normalization
for feat in x_feat_list:
    df_manhattan_subset[feat] = df_manhattan_subset[feat] / df_manhattan_subset[feat].std()

# get the x and y from the dataset 
x = df_manhattan_subset.loc[:, x_feat_list].values
y_true = df_manhattan_subset.loc[:, y_feat].values

# initialize a knn_regressor
knn_regressor = KNeighborsRegressor()

# cross validation 
kfold = KFold(shuffle=True)

# allocate an empty array to store predictions in
y_pred = copy(y_true)

for train_idx, test_idx in kfold.split(x, y_true):
    # build arrays which correspond to x, y train /test
    x_test = x[test_idx, :]
    x_train = x[train_idx, :]
    y_true_train = y_true[train_idx]

    # fit on training data 
    knn_regressor.fit(x_train, y_true_train)

    # estimate rent 
    y_pred[test_idx] = knn_regressor.predict(x_test)
r2 = r2_score(y_true, y_pred)
print(r2)
# r2 = 0.80

# Using all important x variables plus neighborhood columns
# x and y variables of interest 
x_feat_list = new_df_manhattan.columns[9:]
y_feat = 'rent'

# scale normalization
for feat in x_feat_list:
    new_df_manhattan[feat] = new_df_manhattan[feat] / new_df_manhattan[feat].std()

# get the x and y from the dataset 
x = new_df_manhattan.loc[:, x_feat_list].values
y_true = new_df_manhattan.loc[:, y_feat].values

# initialize a knn_regressor
knn_regressor = KNeighborsRegressor()

# cross validation 
kfold = KFold(shuffle=True)

# allocate an empty array to store predictions in
y_pred = copy(y_true)

for train_idx, test_idx in kfold.split(x, y_true):
    # build arrays which correspond to x, y train /test
    x_test = x[test_idx, :]
    x_train = x[train_idx, :]
    y_true_train = y_true[train_idx]

    # fit on training data 
    knn_regressor.fit(x_train, y_true_train)

    # estimate rent 
    y_pred[test_idx] = knn_regressor.predict(x_test)
r2 = r2_score(y_true, y_pred)
print(r2)
# r2 = 0.79

# From these two k-nearest neighbors models, it is clear that the model 
# with only the top 11 most important features is a slightly better model (
# with a slightly better r2 score). Also, the top 11 most important features
# model is more simplistic with less variables, thus we would go with the
# first k-nearest neighbors model we made. 
##############################################################################
# Visualizations

# square feet vs rent - scatterplot
plt.scatter(new_df_manhattan['size_sqft'], new_df_manhattan['rent'])
plt.xlabel('size in square feet')
plt.ylabel('rent price')
plt.title('Rent Price vs Size in Square Feet')

# histogram of rent price 
plt.hist(new_df_manhattan['rent'])
plt.xlabel('rent price')
plt.ylabel('count of places with that rent price')
plt.title('Count of Places in Manhattan Within Each Rent Price Range')

# building age in years vs rent - scatterplot
plt.scatter(new_df_manhattan['building_age_yrs'], 
            new_df_manhattan['rent'])
plt.xlabel('building age in years')
plt.ylabel('rent price')
plt.title('Rent Price vs Building Age in Years')
