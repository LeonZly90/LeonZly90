import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, learning_curve, cross_val_score
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
import time

ohe = preprocessing.OneHotEncoder(categories='auto')
standardscaler = preprocessing.StandardScaler()


def areaData(loc):
    df = pd.read_csv("C:\PepperPepper\pepperProject.csv", encoding='unicode_escape', engine='python')
    df_data = df.copy()
    new_sheet = pd.DataFrame(df_data,
                             columns=['OMOP_COMP_CODE', 'CTRL_JOB', 'STAGE_CODE', 'MARKET_TYPE', 'POTENTIAL_REV_AMT',
                                      'TOTAL_HOURS'])
    new_sheet = new_sheet[
        ~new_sheet['MARKET_TYPE'].isin(['Select Market', 'Self Performed Work', 'Self Performed Direct'])]
    new_sheet = new_sheet[new_sheet['POTENTIAL_REV_AMT'] > 0]  # set potential revenue > 0
    location = new_sheet[new_sheet.OMOP_COMP_CODE == loc]
    # location = location.drop('OMOP_COMP_CODE', 1)
    location = location.iloc[:, 1:]
    # print('location:\n', location)

    JobHour_by_StageMarket = location.groupby(['CTRL_JOB', 'STAGE_CODE', 'MARKET_TYPE'])[
        ['POTENTIAL_REV_AMT', 'TOTAL_HOURS']].sum().reset_index()
    # print('JobHour_by_StageMarket:\n', JobHour_by_StageMarket)

    revAmt_Hour0 = JobHour_by_StageMarket.iloc[:, -2:].abs()
    # print(revAmt_Hour0)

    plt.subplot(1, 2, 1)
    plt.scatter(revAmt_Hour0['POTENTIAL_REV_AMT'], revAmt_Hour0['TOTAL_HOURS'])
    plt.xlabel('POTENTIAL_REV_AMT')
    plt.ylabel('TOTAL_HOURS')
    plt.title("Original data")

    z_scores = stats.zscore(revAmt_Hour0)
    abs_z_scores = np.abs(z_scores)
    revAmt_Hour1 = revAmt_Hour0[(abs_z_scores < 3).all(axis=1)]
    # print(revAmt_Hour1)
    plt.subplot(1, 2, 2)
    plt.scatter(revAmt_Hour1['POTENTIAL_REV_AMT'], revAmt_Hour1['TOTAL_HOURS'])
    plt.xlabel('POTENTIAL_REV_AMT1')
    plt.ylabel('TOTAL_HOURS1')
    plt.title("Outlier clean")
    plt.gcf().set_size_inches(17, 11)
    plt.suptitle("Location: %i" % loc)
    plt.show()

    rest = JobHour_by_StageMarket.iloc[:, :-2]
    JobHour_by_StageMarket = rest.join(revAmt_Hour1, how='outer')
    JobHour_by_StageMarket = JobHour_by_StageMarket.dropna()
    # print('Now JobHour_by_StageMarket:\n', JobHour_by_StageMarket)

    numer_feature = standardscaler.fit_transform(JobHour_by_StageMarket["POTENTIAL_REV_AMT"].values.reshape(-1, 1))
    numer_feature = pd.DataFrame(numer_feature, columns=["POTENTIAL_REV_AMT"])
    # print('numer_feature\n', numer_feature)

    feature_arr = ohe.fit_transform(JobHour_by_StageMarket[['STAGE_CODE', 'MARKET_TYPE']]).toarray()
    feature_labels = ohe.get_feature_names()

    feature_labels = np.array(feature_labels, dtype=object).ravel()
    # print('feature_labels\n', feature_labels)
    features = pd.DataFrame(feature_arr, columns=feature_labels)
    # print('features\n', features)

    predictors = np.concatenate([features, numer_feature], axis=1)
    # print('predictors:\n', predictors)

    target = JobHour_by_StageMarket['TOTAL_HOURS']
    # print('target:\n', target)

    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.20, random_state=37)
    # print(X_train.shape)
    # print(X_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    return X_train, X_test, y_train, y_test, JobHour_by_StageMarket


def ChooseModel(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    dr = DecisionTreeRegressor()
    svr_linear = SVR()
    svr_rbf = SVR()
    svr_poly = SVR()
    rf = RandomForestRegressor()
    gb = GradientBoostingRegressor()
    abr = AdaBoostRegressor()
    br = BaggingRegressor()

    # model_list = [lr, dr, svr_linear, svr_rbf, svr_poly, rf, gb, abr, br]
    model_list = [lr, dr, svr_linear, svr_rbf, svr_poly, rf, gb, abr]
    r2_score_list = []

    for m in model_list:
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        r2_scores = r2_score(y_test, y_pred)
        r2_score_list.append(r2_scores)
        # print("r2_scores: {0}".format(r2_scores))
    model_dic = dict(zip(model_list, r2_score_list))
    max_R2_value = max(model_dic.values())
    best_model = max(model_dic, key=model_dic.get)
    # print('best_model:', best_model, 'max_R2_value:', max_R2_value)
    return best_model, max_R2_value


def gbTrain(X_train, X_test, y_train, y_test):
    start_time = time.time()
    search_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [.001, 0.01, .1], 'max_depth': [2, 3, 4],
                   'subsample': [.5, .75, 1]}
    search = GridSearchCV(estimator=best_model, param_grid=search_grid, scoring='neg_mean_squared_error', n_jobs=1)
    search.fit(X_train, y_train)
    print('\nsearch.best_params_', search.best_params_)
    print('best_search\n', search.best_estimator_)

    end_time = time.time()
    print('time', end_time - start_time)

    # reg = search.best_estimator_
    reg = GradientBoostingRegressor(n_estimators=50,
                                    max_depth=3,
                                    learning_rate=0.1,
                                    subsample=0.75, random_state=1)

    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    r2_score_tuned = r2_score(y_test, y_pred)
    print('\n', best_model, ':')
    print("r2_score: {0}".format(r2_score_tuned))

    x_ax = range(len(y_test))
    plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
    plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.suptitle("Location: %i" % loc)
    plt.show()
    return reg


def rfTrain(X_train, X_test, y_train, y_test):
    # start_time = time.time()
    # reg = RandomForestRegressor(n_estimators=1000, criterion="mse")
    # reg.fit(X_train, y_train)
    # y_pred = reg.predict(X_test)
    # r2_scoreE = r2_score(y_test, y_pred)
    # print('\nRandom Forest\n')
    # print("r2_score: {0}".format(r2_scoreE))
    # rmse = mean_squared_error(y_test, y_pred, squared=False)
    # print("RMSE: {0}".format(rmse))
    #
    # ###################################################################################
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
    # print('\nsearch.best_params_', rf_random.best_params_)
    # end_time = time.time()  # time 304.75399446487427
    # print('time', end_time - start_time)
    #
    #
    # best_search = rf_random.best_estimator_
    # print('best_search\n', best_search)
    # reg = best_search
    ####################################################################################
    # search.best_params_ {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 80, 'bootstrap': True}
    reg = RandomForestRegressor(n_estimators=200, min_samples_split=5, min_samples_leaf=4, max_features='auto',
                                max_depth=80, bootstrap='True')
    # reg = RandomForestRegressor()
    # r2_score: 0.7872974759353466
    # MSE: 1107.7595622634976

    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    r2_scoreF = r2_score(y_test, y_pred)
    # print('\nRF RF RF RF RF RF RF RF RF RF RF RF RF RF RF RF RF RF RF')
    print("r2_score: {0}".format(r2_scoreF))  # r2_score:
    mse = mean_squared_error(y_test, y_pred, squared=False)
    print("MSE: {0}".format(mse))

    x_ax = range(len(y_test))
    plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
    plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.title('RandomForestRegressor')
    plt.show()
    return reg


def adaTrain(X_train, X_test, y_train, y_test):
    reg = AdaBoostRegressor()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    x_ax = range(len(y_test))
    plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
    plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.title('AdaBoostRegressor')
    plt.show()
    return reg


def predict_new_data(reg, test_data):
    test_dataframe = pd.DataFrame(columns=JobHour_by_StageMarket.columns[1:3])
    # print('test_dataframe:\n', test_dataframe)
    for index, column in enumerate(test_dataframe.columns):
        test_dataframe[column] = [test_data[index]]
    # print('test_dataframe:\n', test_dataframe)

    cate_test_one_hot = ohe.transform(test_dataframe).toarray()
    # print('cate_test_one_hot\n', cate_test_one_hot)
    numer_feature = standardscaler.transform(np.array(test_data[-1]).reshape(-1, 1))
    # print('numer_test_stand:\n', numer_feature)
    test = np.concatenate([cate_test_one_hot, numer_feature], axis=1)
    # print('test:\n', test)
    result = reg.predict(test)
    # print('result', result)
    return result


if __name__ == "__main__":
    locs = [100, 200, 300, 1700]
    print("Choose your Location NUMBER: IL-100, IN-200, Environment-300, OH-1700:")

    while True:
        loc = int(input())
        if loc in locs:
            print("You pick:", loc)
            X_train, X_test, y_train, y_test, JobHour_by_StageMarket = areaData(loc)
            best_model, max_R2_value = ChooseModel(X_train, X_test, y_train, y_test)
            print('best_model:', best_model, 'max_R2_value:', max_R2_value)
            print('#' * 30)
            STAGE_CODE = np.unique(JobHour_by_StageMarket['STAGE_CODE'])
            MARKET_TYPE = np.unique(JobHour_by_StageMarket['MARKET_TYPE'])
            print('Choose your STAGE_CODE:', STAGE_CODE)
            STAGE_CODE = str(input())
            print("You pick:", STAGE_CODE, '\n')

            print('Choose your MARKET_TYPE:', MARKET_TYPE)
            MARKET_TYPE = str(input())
            print("You pick:", MARKET_TYPE, '\n')

            print('Choose your POTENTIAL_REV_AMT:')
            POTENTIAL_REV_AMT = int(input())
            print("You pick: $", POTENTIAL_REV_AMT, '\n')

            test_data = [STAGE_CODE, MARKET_TYPE, POTENTIAL_REV_AMT]
            print('Your test data are:\n', test_data)

            if str(best_model) == str(RandomForestRegressor()):
                rf_reg = rfTrain(X_train, X_test, y_train, y_test)
                print('Running rf...')
                result = round(predict_new_data(rf_reg, test_data)[0], 2)
                print("For new data forecast:", result)  # 355    127.86

            elif str(best_model) == str(GradientBoostingRegressor()):
                gb_reg = gbTrain(X_train, X_test, y_train, y_test)
                print('Runing gb...')
                result = round(predict_new_data(gb_reg, test_data)[0], 2)
                print("For new data forecast1:", result)  # 355    127.86

            elif str(best_model) == str(AdaBoostRegressor()):
                ada_reg = adaTrain(X_train, X_test, y_train, y_test)
                print('Runing ada...')
                result = round(predict_new_data(ada_reg, test_data)[0], 2)
                print("For new data forecast:", result)  # 355    127.86

            break
        else:
            print('Wrong number, input again.')
            print("Choose your Location NUMBER: IL-100, IN-200, Environment-300, OH-1700:")
