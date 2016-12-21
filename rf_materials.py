from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import pandas as pd
import numpy as np
import platform

# import sys
# sys.path.insert(0, 'C:\Python34\data-analysis-python')
# from mfunc import lin

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

    
def readData():
    print('Reading the data')
    print('Running on:',platform.system())
    if(platform.system() == 'Linux'):
        rawdata="../../../Documents/Datasets/masterdata.dat"
#        rawdata="../../../Documents/Datasets/lowd071216.dat"
    else:
        rawdata='C:\\Python34\\datasets\\lowd071216.txt'
    dfraw = pd.read_csv(rawdata)
    if(True): print("Original:\n",dfraw.head(5))
    return(dfraw)

    
    
def main():

# A Get the data
# B,C Clean and prepare the data

    # Read the raw data
    dfraw=readData()

    # Plots
    
    # Temporary stop if needed
    # assert True==False, "Temporary stop"
    
    # Shuffle the ordering of rows and print a sample
    df=dfraw.reindex(np.random.permutation(dfraw.index))
    if(True): print("Shuffled:\n",df.head(8))
    nSamples=len(df)    
    print('finish reading data, length of data:',nSamples)


# D Select the features and samples
    # Features
    # featureNames=['Z','q','Nval','Nn','Lowd1','Lowd2','Lowd3']
    # featureNames=['Type','Z','q','Nval','Nn']
    featureNames=['Type','Z','Nn']
    print('Feature names:',featureNames)
    nFeatures=len(featureNames)
    features=np.zeros((nSamples,nFeatures))
    i=-1
    for featureName in featureNames:
        i+=1
        features[:,i]=df[featureName].factorize()[0]

    # Target vector numerical: yNum
    yNum=df['Eads'].values
    if(True): print("Eads:\n",yNum)

    # Target vector binary: yBin
    # http://stackoverflow.com/questions/27117773/pandas-replace-values
    df['EadsBin']=df['Eads']   # CHECK
    EadsLim=0.3
    print('Limits: +-',EadsLim)
    df.loc[ np.abs(df['Eads'] +0.29) > EadsLim  ,'EadsBin'] = 0
    df.loc[ np.abs(df['Eads'] +0.29) <= EadsLim  ,'EadsBin'] = 1
    yBin=df['EadsBin'].factorize()[0] 
    if(False): print("Data frame:\n",df)


    
# E1 Select the training and testing sets 
    sizeTestSet=5
    # Note: Shuffling done after reading raw data
    features_train, features_test = selectsets(features, sizeTestSet)
    y_train, y_test = selectsets(yBin, sizeTestSet)
    y_train_num, y_test_num = selectsets(yNum, sizeTestSet)
    print("Test set binary:", y_test)
    print("Test set numeric:", y_test_num)
    



# E2 Train the classifier model
    clf = RandomForestClassifier(n_estimators=500,max_features=3,oob_score=True,verbose=0)
    clf.fit(features_train, y_train)
    print('oob_score error:',1.0-clf.oob_score_)
    print('feature names:      ',featureNames)
    print('feature importances:',clf.feature_importances_)
    print(' ')

    
# F Test the classifier model
    print('Test with the true testing set, size:',y_test.size)  
    preds=clf.predict(features_test)
    print(features_test)
    print("Preds:",preds)
    print("True:",y_test)


# Train the RF regressor
# http://stackoverflow.com/questions/20095187/regression-trees-or-random-forest-regressor-with-categorical-inputs
    reg = RandomForestRegressor(n_estimators=150, min_samples_split=1)
    reg.fit(features_train,y_train_num)


# Test the RF regressor
    print("Test regression")
    preds=reg.predict(features_test)
    print("Preds:",preds)
    print("True:",y_test_num)

    
# Test regression with user input
    if (len(featureNames) == 3):
        print(" ")
        print("Give user input for:",featureNames)
        in1=float(input())
        in2=float(input())
        in3=float(input())
        userInput=np.array([in1, in2, in3]).reshape(1,-1)
        # userInput=np.array([0,26,0,8,1]).reshape(1,-1)
        print("User input:",userInput)    
        preds=reg.predict(userInput)
        print("Preds, Delta GH:      ",preds)
        preds=clf.predict(userInput)
        print("Preds, Classification:",preds)

    print("End-main")

if __name__ == '__main__':
    main()


# Dump

