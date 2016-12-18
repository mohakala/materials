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
    dfraw = pd.read_csv(rawdata)
    nSamples=len(dfraw)
    if(False): print("Original:\n",dfraw.head(10))
    # Note: Shuffling done here
    df=dfraw.reindex(np.random.permutation(dfraw.index))
    if(True): print("Shuffled:\n",df.head(10))
    
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
    EadsLim=0.4
    print('Limits: +-',EadsLim)
    df.loc[ np.abs(df['Eads'] +0.29) > EadsLim  ,'EadsBin'] = 0
    df.loc[ np.abs(df['Eads'] +0.29) <= EadsLim  ,'EadsBin'] = 1
    yBin=df['EadsBin'].factorize()[0] 

    if(False): print(df)


# E1 Select the training and testing sets 
    sizeTestSet=5
    # Note: Shuffling done after reading raw data
    features_train, features_test = selectsets(features, sizeTestSet)
    y_train, y_test = selectsets(yBin, sizeTestSet)



# E2 Train the classifier model
    clf = RandomForestClassifier(n_estimators=500,max_features=3,oob_score=True,verbose=0)
    clf.fit(features_train, y_train)
    print('oob_score error:',1.0-clf.oob_score_)
    print('feature names:      ',featureNames)

    print('feature importances:',clf.feature_importances_)


# F Test the classifier model
    print('Test with the true testing set, size:',y_test.size)  
    preds=clf.predict(features_test)
    print(features_test)
    print("Preds:",preds)
    print("True:",y_test)

# TODO: RF regression


if __name__ == '__main__':
    main()


# Dump

