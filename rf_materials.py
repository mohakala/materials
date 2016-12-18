from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, 'C:\Python34\data-analysis-python')

from mfunc import lin

# A Get the data
# B,C Clean and prepare the data
# D Prepare the features and samples
# E1 Select the training and testing sets 
# E2 Train the model    
# F Test the model


def selectsets(data,sizeTestSet):
    trainingSet=data[:-sizeTestSet]
    testSet=data[-sizeTestSet:]
    return(trainingSet,testSet)


def main():

# A Get the data
# B,C Clean and prepare the data
    rawdata='C:\\Python34\\datasets\\lowd071216.txt'
    df = pd.read_csv(rawdata)
    nSamples=len(df)
    if(True): print(df.head(10))
    print('finish reading data, length of data:',nSamples)


# D Select the features and samples
    # Features
    # featureNames=['Z','q','Nval','Nn','Lowd1','Lowd2','Lowd3']
    featureNames=['Z','q','Nval','Nn']

    print('Feature names:',featureNames)
    nFeatures=len(featureNames)
    features=np.zeros((nSamples,nFeatures))
    i=-1
    for featureName in featureNames:
        i+=1
        features[:,i]=df[featureName].factorize()[0]

    # Target vector numerical: yNum
    yNum=df['Eads'].factorize()[0]

    # Target vector binary: yBin
    # http://stackoverflow.com/questions/27117773/pandas-replace-values
    df['EadsBin']=df['Eads']
    df.loc[ np.abs(df['Eads'] +0.29) > 0.4  ,'EadsBin'] = 0
    df.loc[ np.abs(df['Eads'] +0.29) <= 0.4  ,'EadsBin'] = 1
    yBin=df['EadsBin'].factorize()[0] 

    if(True): print(df)


# E1 Select the training and testing sets 
    sizeTestSet=1
    features_train, features_test = selectsets(features, sizeTestSet)
    y_train, y_test = selectsets(yBin, sizeTestSet)



# E2 Train the model
    clf = RandomForestClassifier(n_estimators=500,max_features=3,oob_score=True,verbose=0)
    clf.fit(features_train, y_train)
    print('oob_score error:',1.0-clf.oob_score_)
    print('feature importances:',clf.feature_importances_)



if __name__ == '__main__':
    main()


# Dump

