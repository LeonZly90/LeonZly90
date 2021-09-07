import streamlit as st  # streamlit run Location100_RF_streamlit.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
import config

st.title('Pepper ML for Chicago area(location 100) by using random forest')
# df = pd.read_csv("C:\PepperPepper\pepperProject.csv", encoding = 'unicode_escape', engine ='python')
url = f'https://raw.githubusercontent.com/LeonZly90/myData/main/pepperProject.csv?token=AG6BQ7M2G3HRK4IT4IU5ZALBD7M3S'
df = pd.read_csv(url, encoding='unicode_escape', engine='python')
df_data = df.copy()

new_sheet = pd.DataFrame(df_data,
                         columns=['OMOP_COMP_CODE', 'CTRL_JOB', 'STAGE_CODE', 'MARKET_TYPE', 'POTENTIAL_REV_AMT',
                                  'TOTAL_HOURS'])
new_sheet = new_sheet[~new_sheet['MARKET_TYPE'].isin(['Select Market', 'Self Performed Work', 'Self Performed Direct'])]
new_sheet = new_sheet[new_sheet['POTENTIAL_REV_AMT'] > 0]
location_100 = new_sheet[new_sheet.OMOP_COMP_CODE == 100]
location_100 = location_100.drop('OMOP_COMP_CODE', 1)
# st.write('location_100:\n', location_100)

JobHour_by_StageMarket = location_100.groupby(['CTRL_JOB', 'STAGE_CODE', 'MARKET_TYPE'])[
    'POTENTIAL_REV_AMT', 'TOTAL_HOURS'].sum().reset_index()
# st.write('JobHour_by_StageMarket:\n', JobHour_by_StageMarket)  # [474 rows x 5 columns]

revAmt_Hour0 = JobHour_by_StageMarket.iloc[:, -2:].abs()
# st.write(revAmt_Hour0)
# with st.echo(code_location='below'):
#     fig1 = plt.figure(1)
#     plt.scatter(revAmt_Hour0['POTENTIAL_REV_AMT'], revAmt_Hour0['TOTAL_HOURS'])
#     plt.xlabel('POTENTIAL_REV_AMT')
#     plt.ylabel('TOTAL_HOURS')
#     plt.show()
# st.write(fig1)

# clean outlier [469 rows x 5 columns]
z_scores = stats.zscore(revAmt_Hour0)
abs_z_scores = np.abs(z_scores)
revAmt_Hour1 = revAmt_Hour0[(abs_z_scores < 3).all(axis=1)]
# st.write(revAmt_Hour1)
# with st.echo(code_location='below'):
#     fig2=plt.figure(2)
#     plt.scatter(revAmt_Hour1['POTENTIAL_REV_AMT'], revAmt_Hour1['TOTAL_HOURS'])
#     plt.xlabel('POTENTIAL_REV_AMT1')
#     plt.ylabel('TOTAL_HOURS1')
#     plt.show()
# st.write(fig2)

rest = JobHour_by_StageMarket.iloc[:, :-2]
JobHour_by_StageMarket = rest.join(revAmt_Hour1, how='outer')
# @st.cache  # 👈 This function will be cached
JobHour_by_StageMarket = JobHour_by_StageMarket.dropna()
# st.write('Now JobHour_by_StageMarket:\n', JobHour_by_StageMarket)  # [469 rows x 5 columns]
# @st.cache  # 👈 This function will be cached
standardscaler = preprocessing.StandardScaler()
numer_feature = standardscaler.fit_transform(JobHour_by_StageMarket["POTENTIAL_REV_AMT"].values.reshape(-1, 1))
numer_feature = pd.DataFrame(numer_feature, columns=["POTENTIAL_REV_AMT"])
# st.write('numer_feature\n', numer_feature)

# @st.cache  # 👈 This function will be cached
ohe = preprocessing.OneHotEncoder(categories='auto')
feature_arr = ohe.fit_transform(JobHour_by_StageMarket[['STAGE_CODE', 'MARKET_TYPE']]).toarray()
feature_labels = ohe.get_feature_names()
# st.write(feature_labels)
feature_labels = np.array(feature_labels, dtype=object).ravel()
# st.write('feature_labels\n', feature_labels)
features = pd.DataFrame(feature_arr, columns=feature_labels)
# st.write('features\n', features)

predictors = np.concatenate([features, numer_feature], axis=1)
# st.write('predictors:\n', predictors)

target = JobHour_by_StageMarket['TOTAL_HOURS']
# st.write('target:\n', target)

X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.20, random_state=37)
# st.write(X_train.shape)
# st.write(X_test.shape)
# st.write(y_train.shape)
# st.write(y_test.shape)

# (328, 14)
# (141, 14)
# (328,)
# (141,)


# Random Forest # 0.7806525157351498 initial
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import time

start_time = time.time()

# reg = RandomForestRegressor(n_estimators=1000, criterion="mse")
# reg.fit(X_train, y_train)
# y_pred = reg.predict(X_test)
# r2_scoreE = r2_score(y_test, y_pred)
# st.write('\nRandom Forest\n')
# st.write("r2_score: {0}".format(r2_scoreE))
# rmse = mean_squared_error(y_test, y_pred, squared=False)
# st.write("RMSE: {0}".format(rmse))

####################################################################################
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
#
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
#
# rf = RandomForestRegressor()
# rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
#                                random_state=42, n_jobs=-1)
# # Fit the random search model
# rf_random.fit(X_train, y_train)
#
# # search.best_params_ {'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 100, 'bootstrap': True}
# # search.fit(X_train, y_train)
# st.write('\nsearch.best_params_', rf_random.best_params_)
# end_time = time.time()  # time 304.75399446487427
# st.write('time', end_time - start_time)
#
#
# best_search = rf_random.best_estimator_
# st.write('best_search\n', best_search)
# reg = best_search
####################################################################################
# search.best_params_ {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 80, 'bootstrap': True}
reg = RandomForestRegressor(n_estimators=200, min_samples_split=5, min_samples_leaf=4, max_features='auto',
                            max_depth=80, bootstrap='True')
# reg = RandomForestRegressor()
# r2_score: 0.7872974759353466
# MSE: 1107.7595622634976

# @st.cache  # 👈 This function will be cached
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

r2_scoreF = r2_score(y_test, y_pred)
# st.write('\nRF RF RF RF RF RF RF RF RF RF RF RF RF RF RF RF RF RF RF')
# st.write("accur2_score: {0}".format(r2_scoreF))  # r2_score:
mse = mean_squared_error(y_test, y_pred, squared=False)
# st.write("MSE: {0}".format(mse))

x_ax = range(len(y_test))
# with st.echo(code_location='below'):
fig3 = plt.figure(3)
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
plt.xlabel('Trained model')
plt.ylabel('HOURS')
plt.legend()
plt.show()
st.write(fig3)


# @st.cache  # 👈 This function will be cached
def predict_new_data(test_data):
    test_dataframe = pd.DataFrame(columns=JobHour_by_StageMarket.columns[1:3])
    # st.write('test_dataframe:\n', test_dataframe)
    for index, column in enumerate(test_dataframe.columns):
        test_dataframe[column] = [test_data[index]]
    # st.write('test_dataframe:\n', test_dataframe)

    cate_test_one_hot = ohe.transform(test_dataframe).toarray()
    # st.write('cate_test_one_hot\n', cate_test_one_hot)
    numer_feature = standardscaler.transform(np.array(test_data[-1]).reshape(-1, 1))
    # st.write('numer_test_stand:\n', numer_feature)
    test = np.concatenate([cate_test_one_hot, numer_feature], axis=1)
    # st.write('test:\n', test)
    return reg.predict(test)


# ['STAGE_CODE','MARKET_TYPE',"POTENTIAL_REV_AMT"]
test_data_1 = ["BO", "Higher Education", 30000000]  # 355
test_data_2 = ["SALE", "Healthcare", 20236036]  # 909
test_data_3 = ["SALE", "Healthcare", 65172520]  # 1180
test_data_4 = ["BR", "Healthcare", 297000]  # 52

# st.write("For new data forecast1:", str(round(predict_new_data(test_data_1)[0], 2)))  # 355    127.86
# st.write("For new data forecast2:", str(round(predict_new_data(test_data_2)[0], 2)))  # 909    1536.94
# st.write("For new data forecast3:", str(round(predict_new_data(test_data_3)[0], 2)))  # 1180   1385.98
# st.write("For new data forecast4:", str(round(predict_new_data(test_data_4)[0], 2)))  # 52     42.82


STAGE_CODE = np.unique(JobHour_by_StageMarket['STAGE_CODE'])
MARKET_TYPE = np.unique(JobHour_by_StageMarket['MARKET_TYPE'])
r2_scoreF = r2_scoreF*100

st.write("Accuracy rate(r2_score): {0}%".format(round(r2_scoreF, 2)))

option1 = st.sidebar.selectbox(
    'Choose your STAGE_CODE:',
    STAGE_CODE)
st.write('You selected: ', option1)

option2 = st.sidebar.selectbox(
    'Choose your MARKET_TYPE:',
    MARKET_TYPE)
st.write('You selected: ', option2)

option3 = st.sidebar.number_input(
    'Put your POTENTIAL_REV_AMT:',
)
st.write('You selected: $', option3)

test_data = [option1, option2, option3]
if float(test_data[2]) <= 0.00:
    res = 0
else:
    # st.sidebar.write('You want to predict:', test_data)
    res = round(predict_new_data(test_data)[0], 2)
st.sidebar.write("Estimate:", res, 'hours.')
st.write('Estimate:', res, 'project hours.')
