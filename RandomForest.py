# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 21:18:44 2015

@author: yz
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def processMissingAge():
    'Use RandomForest to fit NaN Age'
    global data_train
    
    age_df = data_train[['Age','Embarked','Fare', 'Parch', 'SibSp', 'Title','Pclass','Sex']]
    X = age_df.loc[ (data_train.Age.notnull()) ].values[:, 1:]
    y = age_df.loc[ (data_train.Age.notnull()) ].values[:, 0]
        
    rtr = RandomForestClassifier(n_estimators=2000, n_jobs=-1)
    rtr=rtr.fit(X, y)
            
    predictedAges = rtr.predict(age_df.loc[ (data_train.Age.isnull()) ].values[:, 1::])
    data_train.loc[ (data_train.Age.isnull()), 'Age' ] = predictedAges 
        
    
def childElder():
    'get childElder: <13--child and >60--elder'
    global data_train
    
    data_train['Child_elder'] = np.where(data_train.Age < 13, 'child', 'adult')
    data_train['Child_elder'] = np.where(data_train.Age > 60, 'elder', data_train['Child_elder'])
    data_train['Child_elder'] = pd.factorize(data_train['Child_elder'])[0]
   
def binAge():
    'bin Age'
    global data_train
    
    data_train['Age_bin'] = pd.qcut(data_train['Age'], 4)
    data_train['Age_bin'] = pd.factorize(data_train['Age_bin'])[0]

def quafeatureInteract(quafeatures):
    'careate an qualified interaction feature bewteen every pair, input feature name string list'
    global data_train
    global quaInteraction__count
    
    for i,x in enumerate(quafeatures):
        for j,y in enumerate(quafeatures):
            if i<j:
                interaction_feature=[str(f1) + str(f2) for f1, f2
                in zip(data_train[quafeatures[i]],data_train[quafeatures[j]])]
                data_train=pd.concat([data_train,pd.Series(interaction_feature
                , name=quafeatures[i]+'*'+quafeatures[j])],axis=1)
                data_train[quafeatures[i]+'*'+quafeatures[j]] = pd.factorize(data_train[quafeatures[i]+'*'+quafeatures[j]])[0]
                quaInteraction__count += 1

def quantfeatureInteract(quantfeatures):
    'quantative interaction: + - *'
    global data_train
    global quantInteraction__count
    
    numerics = data_train.loc[:, quantfeatures]    
    for i in range(0, numerics.columns.size-1):
        for j in range(0, numerics.columns.size-1):
            if i <= j:
                name = str(numerics.columns.values[i]) + "*" + str(numerics.columns.values[j])
                data_train = pd.concat([data_train, pd.Series(numerics.iloc[:,i] * numerics.iloc[:,j], name=name)], axis=1)
                quantInteraction__count += 1
            if i < j:
                name = str(numerics.columns.values[i]) + "+" + str(numerics.columns.values[j])
                data_train = pd.concat([data_train, pd.Series(numerics.iloc[:,i] + numerics.iloc[:,j], name=name)], axis=1)
                quantInteraction__count += 1
            if not i == j:
                name = str(numerics.columns.values[i]) + "-" + str(numerics.columns.values[j])
                data_train = pd.concat([data_train, pd.Series(numerics.iloc[:,i] - numerics.iloc[:,j], name=name)], axis=1)
                quantInteraction__count += 1
                
                
if __name__ == '__main__':
    quaInteraction__count = 0
    quantInteraction__count = 0
    
    data_train=pd.read_csv('./data/train.csv',sep='\t')
    data_train.index=range(891)    #修改行索引，数据从R导过来index从1开始，会造成concat时候的错位
    
    del data_train['PassengerId']
    del data_train['Cabin']
    del data_train['Name']
    del data_train['Ticket']
    del data_train['Surname']
    del data_train['FamilyID']
    del data_train['FamilySize']
    
    #factorize
    data_train['Sex'] = pd.factorize(data_train['Sex'])[0]
    data_train['Embarked'] = pd.factorize(data_train['Embarked'])[0]
    data_train['Title'] = pd.factorize(data_train['Title'])[0]
    
    #process date
    processMissingAge() 
    childElder()    
    binAge()
#    quafeatureInteract(['Pclass','Sex','Embarked','Title','Child_elder','Age_bin'])   #modify!
#    quantfeatureInteract(['Pclass','Age','Fare','Parch','SibSp'])     #modify!
    
    #Fit RandomForest using data_train
    print('Fitting a rough RandomForest...')
    features_list = data_train.columns.values[1:]
    X = data_train.values[:, 1:]
    y = data_train.values[:,0]
    m=RandomForestClassifier(n_estimators=1000, oob_score = True)    #modify!
    m=m.fit(X,y)
    
    #feature importance
    print('Providing ordered feature importance...')
    feature_importance = [(features_list[i], v) for i, v in enumerate(m.feature_importances_)]
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    #output results
    print('Feature list:', features_list)
    print('feature importance(first 20):', feature_importance[:20])    
    print('Out-of-bag-error is:',m.oob_score_)