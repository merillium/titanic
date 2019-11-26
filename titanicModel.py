# Submission for titanic # 
# by Derek O'Connor # 

import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns # visaulize missing values using heatmap 
from statistics import mean
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score # perform cross validation and calculate score 

path = "/Users/Derek/Desktop/kaggle/titanic"
os.chdir(path)
raw_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

def show_missing(data):
	sns.heatmap(data.isnull(),cbar=False)
	pl.show()
# show_missing(raw_data) --> 'Cabin' column has too many missing values

def mutate_features(raw_data):
    if sum(raw_data.columns == 'Survived') == 0:
        df = raw_data[['Pclass','Sex','Age','SibSp','Parch','Embarked','Fare']]
    if sum(raw_data.columns == 'Survived') == 1:
        df = raw_data[['Survived','Pclass','Sex','Age','SibSp','Parch','Embarked','Fare']]
    df['FamilySize'] = df[['SibSp', 'Parch']].sum(axis=1)
    return df

df_full = mutate_features(raw_data)
# a list of dfs with different features 

# df_features = [df_full[['Survived','Pclass','Sex','Age','SibSp','Parch','Embarked']],
                # df_full[['Survived','Pclass','Sex','Age','FamilySize','Embarked']], 
                # df_full[['Survived','Pclass','Sex','Age','SibSp','Parch']],
                # df_full[['Survived','Pclass','Sex','Age','FamilySize']],
                # df_full[['Survived','Pclass','Sex','Age','Embarked']],
                # df_full[['Survived','Pclass','Sex','Age','Parch']],
                # df_full[['Survived','Pclass','Sex','Age','SibSp']],
                # df_full[['Survived','Pclass','Sex','Age']],
                # df_full[['Survived','Sex','Age']]]

df_features = [df_full[['Survived','Pclass','Sex','Age','SibSp','Parch','Embarked']]]

def make_heatmap(df):
    f, ax = pl.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, annot=True, ax=ax)
    
# Example function call: 
# make_heatmap(df)

def survive_by_feature(df, feature):
    uniqueFeatures = df[feature].sort_values().unique()
    totalsByFeature = pd.DataFrame({'Total Survived':[0] * len(uniqueFeatures),
                                   'Total Obs':[0] * len(uniqueFeatures),
                                   'Survival Rate':[0] * len(uniqueFeatures)},
                                    index = uniqueFeatures,)
    for i in range(len(uniqueFeatures)):
        subsetData = df[df[feature] == uniqueFeatures[i]]
        totalsByFeature.iloc[i,0] = sum(subsetData['Survived'])
        totalsByFeature.iloc[i,1] = len(subsetData['Survived'])
        totalsByFeature.iloc[i,2] = sum(subsetData['Survived']) / len(subsetData['Survived'])
    plot = totalsByFeature.iloc[:,0:2].plot.bar(rot=0,title='Survival by '+feature)
    # pandas does not slice inclusive of ending value
    plot.set_xlabel(feature)
    plot.set_ylabel('count')
    return totalsByFeature

# Example function call:
# sex_NA = survive_by_feature(both_xy_NA,'Sex')
# sex = survive_by_feature(both_xy,'Sex')
    
# impute missing age data by training a linear regression / knn method on non-missing data
# and then predicting the missing data point with this method and filling the original dataframe
# the df argument is either the training set which contains the 'Survived' feature
# or the df argument is the test set which does NOT contain the 'Survived' feature

def impute_data(df,feature='Age',method='linreg'):
    y = df[[feature]]
    if sum(df.columns == 'Survived') == 0:
        X = df.drop([feature],axis=1)
    if sum(df.columns == 'Survived') == 1:
        X = df.drop([feature,'Survived'],axis=1)
    missing = y[feature].isna()
    X_test = pd.get_dummies(X[missing])
    y_train, X_train = y[-missing], pd.get_dummies(X[-missing])
    # train linreg on nonmissing data
    if method == 'remove':
        return df.dropna(subset=[feature])
    if method == 'mean':
        df[feature].fillna(df[feature].mean(), inplace=True)
        print('imputed df using mean\n', df)
        return df
    if method == 'linreg':
        linreg = LinearRegression().fit(X_train,y_train)
        y_test = pd.DataFrame({feature:linreg.predict(X_test).tolist()},index=missing[missing == True].index)
    if method == 'knn': 
        knn = KNeighborsRegressor(n_neighbors=7).fit(X_train,y_train)
        y_test = pd.DataFrame({feature:knn.predict(X_test).tolist()},index=missing[missing == True].index)
    fill_missing = pd.DataFrame(y_test[feature].str[0])
    df = df.fillna(value=fill_missing)
    # assert(sum(df['Age'] < 0) == 0)
    return df

# fit logistic, decision tree, naive bayes to target and features 
# and return the average kfold cross validation score over 10 folds 
    
def model_log(cleaned_y, cleaned_X):
    logistic = LogisticRegression(solver='liblinear').fit(cleaned_X,cleaned_y.values.ravel())
    
    # hyperparameter tuning of penalty, C using grid search
    
    # penalty = ['l1','l2']
    # C = np.logspace(0, 4, 10)
    # hyperparameters = dict(C=C, penalty=penalty)
    # clf = GridSearchCV(logistic, hyperparameters, cv=10, verbose=0)
    # best_model = clf.fit(cleaned_X,cleaned_y.values.ravel())
    # print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
    # print('Best C:', best_model.best_estimator_.get_params()['C'])
    
    # hyperparameter tuning of penalty, C using random search
    
    # penalty = ['l1','l2']
    # C = np.linspace(1,200)
    # hyperparameters = dict(C=C, penalty=penalty)
    # rsearch =  RandomizedSearchCV(logistic, hyperparameters, cv = 5, n_iter=20, random_state=40)
    # best_model = rsearch.fit(cleaned_X,cleaned_y.values.ravel())
    # print(rsearch.best_params_)
    
    # return best_model, kfold_cv(cleaned_y,cleaned_X,best_model,10)
    
    return logistic, kfold_cv(cleaned_y,cleaned_X,logistic,10)

def model_tree(cleaned_y, cleaned_X): 
    tree = DecisionTreeClassifier().fit(cleaned_X,cleaned_y)
    return tree, kfold_cv(cleaned_y,cleaned_X,tree,10)

def model_bayes(cleaned_y, cleaned_X):
    bayes = GaussianNB().fit(cleaned_X,cleaned_y.values.ravel())
    return bayes, kfold_cv(cleaned_y,cleaned_X,bayes,10)

# normalization/standardization of features might be required for knn 
def model_knn(cleaned_y, cleaned_X):
    knn = KNeighborsClassifier() 
    hyperparameters = dict(n_neighbors = np.linspace(1,20,20).astype(int))
    clf = GridSearchCV(knn, hyperparameters, cv=10, verbose=0)
    best_model = clf.fit(cleaned_X,cleaned_y.values.ravel())
    print('Best number of neighbors:',clf.best_params_)
    return best_model, kfold_cv(cleaned_y,cleaned_X,knn,10)

# random forest might combat overfitting by a decision tree
def model_random_forest(cleaned_y, cleaned_X):
    rf = RandomForestClassifier(n_estimators = 250, criterion = 'gini',
                                random_state = 2).fit(cleaned_X,cleaned_y.values.ravel())
    return rf, kfold_cv(cleaned_y,cleaned_X,rf,10)

# return mean kfold cross validation score
def kfold_cv(cleaned_y, cleaned_X, model, nfolds):        
    cv_score = cross_val_score(model, cleaned_X, cleaned_y.to_numpy().ravel(), cv=nfolds)
    print('The cv scores are: ', cv_score, 'with an average cv score', mean(cv_score))
    return mean(cv_score)

# this simple 'ensemble' might be total baloney... but I should implement it anyway
# ensemble that takes a simple vote from logistic, tree, and bayes 
def model_ensemble_vote(cleaned_y, cleaned_X):
    return 0

# select_features runs a seleted model on an array of dataframes with different features 
# and returns the features that correspond to the highest k-cross validation score
def select_features(df_features, model_function):
    best_cv_score = 0
    for i in range(len(df_features)):
        # impute missing data with specified method, and then dummify
        print('df to be imputed:\n',df_features[i])
        cleaned_data = pd.get_dummies(impute_data(df_features[i],method='mean'))

        # slice data for model to fit 
        cleaned_y = pd.DataFrame(cleaned_data['Survived'])
        cleaned_X = cleaned_data.drop(['Survived'],axis=1)
        model_current, cv_score = model_function(cleaned_y,cleaned_X)
        if cv_score > best_cv_score:
            model_best, best_cv_score = model_function(cleaned_y,cleaned_X)
            best_feature = i
            print('\ncurrent best features:',df_features[best_feature].columns.values)
            print('\ncurrent best model cv score:',best_cv_score)
            
        else:
            print('\ncurrent features do not offer improvement','\nNext features:')
        print('the best model is:',model_best)
    return list(df_features[best_feature].columns.values), model_best

def output_preds(df):
    Ids = pd.DataFrame(df['PassengerId']) #first obtain PassengerIds
    # select features for df 
    df = mutate_features(df)
    X = df[train_features[1:]] # remove 'Survived' from list 
    X = impute_data(X,feature='Age',method='mean')
    # X = impute_data(X,feature='Fare',method='mean')
    cleaned_X = pd.get_dummies(X)
    
    print('Using the following features on submission set:', cleaned_X.columns.values)
    print('\ncleaned_X\n',cleaned_X)
    print('rows containing NA\n',cleaned_X[cleaned_X.isnull().any(axis=1)])
    y_preds = train_model.predict(cleaned_X)
    
    final_preds = pd.DataFrame({'Survived':y_preds.tolist()})
    final_df = pd.concat([Ids,final_preds],axis=1)
    return final_df

# train specified model & features, evaluating by kfold cross validation
train_features, train_model = select_features(df_features, model_random_forest)
final_preds = output_preds(test_data)
# final_preds.to_csv('submission_7_22_v7.csv',index=False)
    