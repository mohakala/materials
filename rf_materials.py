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
#        rawdata='C:\\Python34\\datasets\\lowd071216.txt'
        rawdata='C:\\Python34\\datasets\\masterdata_win.txt'

    dfraw = pd.read_csv(rawdata)
    print("Data types in dataframe:\n",dfraw.dtypes)
    if(True):
        print("Original:\n",dfraw.head(5))
    if(False):
        print(dfraw['Z'].values)

    return(dfraw)


def gridtestClf(features_train, y_train):
    print("Grid search of parameters of the classifier")

    nEst=[50,100,200,300,400,500,750,1000]
    for i in nEst:
        clf = RandomForestClassifier(n_estimators=i,oob_score=True,verbose=0)
        clf.fit(features_train, y_train)
        print('nEst, oob_score error:', i, 1.0-clf.oob_score_)
    print("Result: Some effect, looks like 500 is often best")

    nMaxf=["auto","sqrt","log2"]
    for i in nMaxf:
        clf = RandomForestClassifier(n_estimators=500,max_features=i,oob_score=True,verbose=0)
        clf.fit(features_train, y_train)
        print('nMaxfeatures, oob_score error:', i, 1.0-clf.oob_score_)
    print("Result: Some effect, auto the best?")

    nLeaf=[1,2,3,4,5,10,50]
    for i in nLeaf:
        clf = RandomForestClassifier(n_estimators=500,max_features="auto",min_samples_leaf=i,oob_score=True,verbose=0)
        clf.fit(features_train, y_train)
        print('nLeaf (leaf size), oob_score error:', i, 1.0-clf.oob_score_)
    print("Result: Looks like default=1 smaller the better")

    nDepth=[2,5,10,20]
    for i in nDepth:
        clf = RandomForestClassifier(n_estimators=500,max_features="auto",max_depth=i,oob_score=True,verbose=0)
        clf.fit(features_train, y_train)
        print('nDepth, oob_score error:', i, 1.0-clf.oob_score_)
    print("Result: Use default")

    print("Selectrion:")
    clf = RandomForestClassifier(n_estimators=500,max_features="auto",oob_score=True,verbose=0)
    clf.fit(features_train, y_train)
    print('FINAL, oob_score error:', 1.0-clf.oob_score_)
    
    print("End grid search of parameters of the classifier")
    # Temporary stop
    # assert True==False
    return()
        

# F Test the classifier model, oob_score
    gridtestClf(features_train, y_train)


    
    
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
    df=df.reset_index(drop=True)
    if(True):
        print("Shuffled:\n",df.head(4))
    nSamples=len(df)    
    print('finish reading data, length of data:',nSamples)

    ishuffle=True
    if (ishuffle):
        print("*Data frame rows shuffled")
        print("df[0]:",df['Hads'][0])
    else:       
        # for debug, dont shuffle (return to original dfraw)
        df=dfraw
        print('NOTE: For debugging, data frame not shuffled!')


# D Select the features and samples
    # Features
    # featureNames=['Type','Z','q','Nval','Nn','Lowd1','Lowd2','Lowd3']
    featureNames=['Type','Z','q','Nval','Nn']
    # featureNames=['Type','Z','Nn']
    print(' \nFeature names:',featureNames)
    nFeatures=len(featureNames)
    features=np.zeros((nSamples,nFeatures))
    i=-1
    for featureName in featureNames:
        i+=1
        features[:,i]=df[featureName].factorize()[0]
        if(False):
            print("Factorizing featName, to type:",featureName,type(features[:,i]))
            print("Type:",type(features[0,i]))
        if(False): print("i,\n feat:",i,features[:,i])

    # This transformation, below, from float to int apparently not needed for RF
    features=features.astype(int)
    print("Type of features:",type(features[0,0]))

    
    # Target vector numerical: yNum
    yNum=df['Hads'].values
    print("Type of Hads[0]:",type(df['Hads'][0]))
    print("Type of yNum[0]:",type(yNum[0]))
    if(False): print("Targets:\n",yNum)


    # Set limiting values for Hads
    HadsLim=0.4
    print('*Limits: +-',HadsLim)
    

    # Target vector binary: yBin (a new variable)
    # Create a new column 'HadsBin'
    df['HadsBin']=0
    # http://stackoverflow.com/questions/27117773/pandas-replace-values
    df.loc[ np.abs(df['Hads']) > HadsLim  ,'HadsBin'] = 0
    df.loc[ np.abs(df['Hads']) <= HadsLim  ,'HadsBin'] = 1
    # yBin=df['HadsBin'].factorize()[0] # old, unnecessary version
    yBin=df['HadsBin'].values
    yBin=yBin.astype(int) # probably unnecessary
    print("Type of yBin[0]:",type(yBin[0]))
    if(False): print("Data frame:\n",df[['Hads', 'HadsBin']])
    if(True):
        print('Target num and bin (for checking):')
        print("yNum:",yNum[:15])
        print("yBin:",yBin[:15])

    
# E1 Select the training and testing sets 
# Note: Shuffling done after reading raw data
    sizeTestSet=5
    print("*Size of the test set:",sizeTestSet)
    features_train, features_test = selectsets(features, sizeTestSet)
    y_train, y_test = selectsets(yBin, sizeTestSet)
    y_train_num, y_test_num = selectsets(yNum, sizeTestSet)
    print("Test set binary:", y_test)
    print("Test set numeric:", y_test_num)
    

# E2 Train the classifier model
    clf = RandomForestClassifier(n_estimators=500,max_features="auto",oob_score=True,verbose=0)
    clf.fit(features_train, y_train)
    print(" \nTraining the classifier")
    print('*oob_score error:',1.0-clf.oob_score_)
    print('feature names:      ',featureNames)
    print('feature importances:',clf.feature_importances_)


# F Test the classifier model, oob_score, search the grid of parameters
#   Only if size of test set is set to 1.
    if(sizeTestSet==1):
        print("\nSize of test set = 1 -> performing grid search for parameters")
        gridtestClf(features_train, y_train)
    else:
        print("\nSize of test set > 1, will not perform grid search for parameters")
    
# F Test the classifier model, test set
    print('Test with the true testing set, size:',y_test.size)  
    preds=clf.predict(features_test)
    if(False): print(features_test)
    print("Preds:",preds)
    print("True:",y_test)




# Train the RF regressor
# http://stackoverflow.com/questions/20095187/regression-trees-or-random-forest-regressor-with-categorical-inputs
    print(" \nTraining the regressor")
    reg = RandomForestRegressor(n_estimators=500, min_samples_split=1)
    reg.fit(features_train,y_train_num)


# Test the RF regressor
    print("Testing the regressor")
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

