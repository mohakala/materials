
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import platform


# A Get the data
# B,C Clean and prepare the data
# D Prepare the features and samples
# E1 Select the training and testing sets 
# E2 Train the model    
# F Test the model

def lin():
    print('-------------------------------------')

def selectsets(data, sizeTestSet):
    trainingSet=data[:-sizeTestSet]
    testSet=data[-sizeTestSet:]
    return(trainingSet,testSet)

    
def genPredictions(reg, clf, featureNames):
    """
    Set input feature values by hand and make predictions
    userInput = input features
    reg = regressor 
    clf = classifier
    """

    # Vector of values to be scanned
    scanValues=np.array([6,7,8,9,10,11])    
    # scanValues=np.array([0,25,26,27,28,29,30,41,42,46,78,79])

    predsList=[]
    for i in scanValues:    
        userInput=np.array([6, i, 1, 6]).reshape(1,-1)
        predsReg=reg.predict(userInput)
        predsClf=clf.predict(userInput)
        predsList.append(predsReg)
        print("Input, Preds, Target, Class:", userInput, predsReg, predsClf)


    scanValues=np.array([42,26,27,28,29,45])    

    predsList2=[]
    for i in scanValues:    
        userInput=np.array([6, i, 1, 5]).reshape(1,-1)
        predsReg=reg.predict(userInput)
        predsClf=clf.predict(userInput)
        predsList2.append(predsReg)
        print("Input, Preds, Target, Class:", userInput, predsReg, predsClf)

    # printy(predsList,(-999,-999)) # To plot only one set of values
    printy(predsList, predsList2)


    # Another way to test regressor with user input
    if (False):
        print(" ")
        print("Give user input for:", featureNames)
        in1=float(input())
        in2=float(input())
        in3=float(input())
        userInput=np.array([in1, in2, in3]).reshape(1,-1)
        # userInput=np.array([0,26,0,8,1]).reshape(1,-1)
        predsReg=reg.predict(userInput)
        predsClf=clf.predict(userInput)
        print("Input, Preds, Target, Class:", userInput, predsReg, predsClf)

    
def printDiag(true, pred, train, trainpred  ):
    print(true, true.shape)
    print(pred, pred.shape)
    for i in true: print(i)
    import matplotlib.pyplot as plt
    fig=plt.figure(1)
    ax=fig.add_subplot(111)
    x=np.linspace(-3,3,100)
    ax.plot(x, x, 'k--', alpha=0.75)
    ax.plot(train, trainpred, 'ro', ms=20, mfc='None', mew=2, alpha=0.35, label="training data")    
    ax.plot(true, pred, 'bo', ms=20, label="test data")

    xlimits=(-2.75, 2.0)
    ax.set_xlim(xlimits[0], xlimits[1])
    ax.set_ylim(xlimits[0], xlimits[1])
    genFontSize=26
    ax.set_xlabel(r'$\Delta G_{\rm H}$ (eV), DFT', fontsize=genFontSize)
    ax.set_ylabel(r'$\Delta G_{\rm H}$ (eV), RF', fontsize=genFontSize)
    ax.tick_params(axis='x', labelsize=genFontSize)
    ax.tick_params(axis='y', labelsize=genFontSize)

#    ax.legend(loc=0, prop={'size':22}) 
    ax.legend(loc=0, prop={'size':26}) 
    
    plt.show()
    pass


    
    
def printy(pred,true):
    """
    Print two sets of values as a function of index
    Use as second argument true = (-999, -999) when you want to plot only one set  
    """
    import matplotlib.pyplot as plt
    fig=plt.figure(1)
    ax=fig.add_subplot(111)
    ax.plot([-12.0, len(pred)+1], [-0.5,-0.5], 'k--')       
    ax.plot([-12.0, len(pred)+1], [+0.5,+0.5], 'k--')   

    label1='Predicted'  # label1='Set 1 (or Predicted)'
    ax.plot(pred, 'x', mfc='None', ms=28, mew=4, label=label1)        

    if (true[0] != -999):
        label2='True'       # label2='Set 2 (or True)'
        ax.plot(true, 'o', mfc='None', ms=28, mew=4, label=label2)

    ax.legend(loc=0, prop={'size':22}) 
    genFontSize=26
    ax.set_xlabel('Index of test case', fontsize=genFontSize)
    ax.set_ylabel(r'$\Delta G_{\rm H}$ (eV)', fontsize=genFontSize)
    # ax.set_ylabel('Target', fontsize=genFontSize)
    ax.tick_params(axis='x', labelsize=genFontSize)
    ax.tick_params(axis='y', labelsize=genFontSize)

    xlimits=(-0.5, len(pred)-0.5)
    ax.set_xlim(xlimits[0], xlimits[1])

    plt.show()
    
    
def readData():
    print('Reading the data')
    print('Running on:',platform.system())
    if(platform.system() == 'Linux'):
        rawdata="../../../Documents/Datasets/masterdata.dat"
    else:
        rawdata='C:\\Python34\\datasets\\masterdata_win.txt'

    dfraw = pd.read_csv(rawdata)
    print("Data types in dataframe:\n",dfraw.dtypes)
    if(True):
        print("Head of original:\n",dfraw.head(5))
    if(False):
        print(dfraw['Z'].values)

    return(dfraw)


def gridtestClf(features_train, y_train, featureNames):
    print("Grid search of parameters of the classifier")
    lin()

    nEst=[50,100,200,300,400,500]
    for i in nEst:
        clf = RandomForestClassifier(n_estimators=i,oob_score=True,verbose=0)
        clf.fit(features_train, y_train)
        print('nEst, oob_score error:', i, 1.0-clf.oob_score_)
        par1, ___ = cross_val(clf, features_train, y_train, featureNames)
        # print('nEst, CV score:', i, par1)
        
    print("Result: Some effect, looks like 100 is often best")
    lin()

    nMaxf=["auto","sqrt","log2"]
    for i in nMaxf:
        clf = RandomForestClassifier(n_estimators=100,max_features=i,oob_score=True,verbose=0)
        clf.fit(features_train, y_train)
        print('nMaxfeatures, oob_score error:', i, 1.0-clf.oob_score_)
    print("Result: Some effect, looks like log2 the best")
    lin()

    nLeaf=[1,2,3,4,5,10,50]
    for i in nLeaf:
        clf = RandomForestClassifier(n_estimators=100,max_features="auto",min_samples_leaf=i,oob_score=True,verbose=0)
        clf.fit(features_train, y_train)
        print('nLeaf (leaf size), oob_score error:', i, 1.0-clf.oob_score_)
    print("Result: Looks like default=1 smaller the better")
    lin()

    nDepth=[2,5,10,20]
    for i in nDepth:
        clf = RandomForestClassifier(n_estimators=100,max_features="auto",max_depth=i,oob_score=True,verbose=0)
        clf.fit(features_train, y_train)
        print('nDepth, oob_score error:', i, 1.0-clf.oob_score_)
    print("Result: Use default")
    lin()


    print("Selection:")
    clf = RandomForestClassifier(n_estimators=100,max_features="auto",oob_score=True,verbose=0)
    clf.fit(features_train, y_train)
    print('FINAL, oob_score error:', 1.0-clf.oob_score_)
    cross_val(clf, features_train, y_train, featureNames)
    
    print("End grid search of parameters of the classifier")
    return()
        

### F Test the classifier model, oob_score
##    gridtestClf(features_train, y_train)

    return()


def r2values(x, y, ypred, p):
    # http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    # Returns the values of r2 and adjusted r2
    yave=np.mean(y)
    ssreg = np.sum((ypred - yave)**2)   
    ssreg = ssreg  # This values is not needed
    sstot = np.sum((y - yave)**2) 
    ssres = np.sum((ypred - y)**2)
    r2 = 1- (ssres/sstot)  # not (ssreg/sstot) 
    n=len(y) 
    r2adj = 1- ( (ssres/(n-p-1)) / (sstot/(n-1)) ) 
    return(r2, r2adj)


def test(model, X, target):
    """
    General help function to do the test over the test set
    If target is non-integral, make also the plot
    model  = classifier which has the method predict: model.predict(X)
    X      = input samples, for example X = features_test
    target = known target values
    """
    import numbers
    if (target.size <= 1):
        print('No tests since target.size < = 1')
        return()
    print('Test with test set, size:', target.size) 
    if(False): 
        print("\nTest systems:", X)
    preds = model.predict(X)
    preds_true_list = np.concatenate((preds.reshape(-1,1), target.reshape(-1,1)), axis=1)
    for value in preds_true_list:
        if(False):
            print("Preds, True:", value)   
    score = model.score(X, target)
    print('*Score (test set):', score)
    r2, r2adj = r2values(0, target, preds, 0)
    if(False):
        print('*Score (test set), own r2:', r2)
    if (isinstance(target[0], numbers.Integral)):
        pass
    else:
        if(True):
            printy(preds, target)  # Make plot when target is a float
    return(score)



def cross_val(model, X, target, featureNames):
    from sklearn.cross_validation import KFold #For K-fold cross validation
    """
    General help function to do cross-validation
    model  = classifier which has the method predict: model.predict(X)
    X      = input samples, for example X = features_test
    target = known target values
    """

    # n-fold cross-validation
    n_folds=5
    kf = KFold(target.shape[0], n_folds=n_folds, shuffle=True)        
    error = []
    for train, test in kf:
        train_predictors = (X[train,:])
        train_target = target[train]
        if(False):
            print('predictors:', train_predictors, 'shape:', train_predictors.shape, '\n')
            print('target:', train_target)
        model.fit(train_predictors, train_target)
        error.append(model.score(X[test,:], target[test]))
    score = np.round(  np.mean(error), 4  )    
    score_error_of_mean = np.round(  np.std(error, ddof=1) / np.sqrt(len(error)), 4  )
    std = np.round(np.std(error, ddof=1), 4)
    print('*Score (cross-validation):', score, '+-', score_error_of_mean, '(std: ', std, ')', 'folds:', n_folds, ', total data:', len(target), ', validat. data:', len(test))
    if(False):
        print('error in the folds:', np.round(error, 2))
    return(score, score_error_of_mean)

    
    
def main():

# A Get the data
# B,C Clean and prepare the data

    # Read the raw data
    dfraw=readData()
        
    # Shuffle the ordering of rows and print a sample
    ishuffle=True
    if (ishuffle):
        np.random.seed(1)  
        print('Keep the intial shuffle fixed (always the same, with seed(1))')
        print('- important since we want a frozen, always same test set')
        df=dfraw.reindex(np.random.permutation(dfraw.index))
        df=df.reset_index(drop=True)
        if(True):
            print("===> Head of shuffled:\n",df.head(2))
        print("*Data frame rows shuffled")
        print("df[0]:",df['Hads'][0])  
        np.random.seed(None)  
    else:       
        # For debug, dont shuffle (use original dfraw)
        df=dfraw
        print('*NOTE: For debugging, data frame NOT shuffled!')

    print('Finished reading data, length of data:',len(df))
    lin()

# A.1 Quick illustrate the target values
    if(False):
        printy(df['Hads'].values,(-999,-999))     
    
    
# B.1 Drop one outlier (Hads value )
    print('Drop the Hads outlier which is larger than 6.0')
    df = df[df.Hads < 6.0]
    if(False):
        printy(df['Hads'].values,(-999,-999))     
    print('Finished reading data, length of data:',len(df))
    lin()

# C.1. Later: Study separately the case types 0 and 1-6
    """
    TO DO
    """
    pass    
    

# C.2 Add a new feature 'Coord' in the dataframe
    # - for cases 0,2,3,6 it is 6
    # - for case 5 it is 5
    # - for case 1,4 it is 4
    df.loc[ df['Type'] == 0 ,'Coord'] = 6
    df.loc[ df['Type'] == 2 ,'Coord'] = 6
    df.loc[ df['Type'] == 3 ,'Coord'] = 6
    df.loc[ df['Type'] == 6 ,'Coord'] = 6
    df.loc[ df['Type'] == 5 ,'Coord'] = 5
    df.loc[ df['Type'] == 1 ,'Coord'] = 4
    df.loc[ df['Type'] == 4 ,'Coord'] = 4
    print("Add new feature Coord to dataframe")

    

# D Select the features and samples
    featureNames=['Type','Nval','Nn','Coord'] # Best for RF (according to CV)
    # featureNames=['Type','Z','Nn','Coord']
    # featureNames=['Type','Z','Nval','Nn','Coord']
    # featureNames=['Type','Z','Nn']
    print(' \nFeature names:',featureNames)
    nFeatures=len(featureNames)
    nSamples=len(df) 
    features=np.zeros((nSamples,nFeatures))
    i=-1
    for featureName in featureNames:
        i+=1
        # features[:,i]=df[featureName].factorize()[0]
        # Try this alternative:
        features[:,i]=df[featureName].values
        if(False):
            print("Factorizing featName, to type:",featureName,type(features[:,i]))
            print("Type:",type(features[0,i]))
        if(False): print("i,\n feat:",i,features[:,i])

    # This transformation below from float to int kept but apparently not needed for RF
    features=features.astype(int)
    print("Type of features:",type(features[0,0]))

    
    # Target vector numerical: yNum
    yNum=df['Hads'].values
    print("Type of Hads[0]:",type(df['Hads'][0]))
    print("Type of yNum[0]:",type(yNum[0]))
    if(False): print("Targets:\n",yNum)

    lin()    

    # Set limiting values for Hads
    HadsLim=0.5
    print('\n*Limits: +-',HadsLim)


    # Target vector binary: yBin (a new variable)

    # Create a new column 'HadsBin' in the dataframe
    # http://stackoverflow.com/questions/27117773/pandas-replace-values
    df.loc[ np.abs(df['Hads']) > HadsLim  ,'HadsBin'] = 0
    df.loc[ np.abs(df['Hads']) <= HadsLim  ,'HadsBin'] = 1
    yBin=df['HadsBin'].values
    yBin=yBin.astype(int) 
    print("Type of yBin[0]:",type(yBin[0]))

    if(False): 
        print("Data frame:\n",df[['Hads', 'HadsBin']])
    if(True):
        print('Target num and bin (for checking):')
        print("yNum:",yNum[:15])
        print("yBin:",yBin[:15])


                     
# E1 Select the training and testing sets. Set doGridsearch = True/False
    sizeTestSet = 10

    print("\n*Size of the test set:",sizeTestSet)
    features_train, features_test = selectsets(features, sizeTestSet)
    y_train, y_test = selectsets(yBin, sizeTestSet)
    y_train_num, y_test_num = selectsets(yNum, sizeTestSet)
    print("Test set binary:", y_test)
    print("Test set numeric:", y_test_num)
    print('Note: We use always the same test data')
    lin()
    lin()


# E1b Now must shuffle randomly the training set
    if(False):
        # THis part not any more needed, since df already shuffled earlier
        print("Shuffling the training set (test set fixed from the beginning)")
        print("- shuffle with random_seed to always get same order")
        print("- cross-validation includes the randomization")    
        if (False):
            print(features_train.shape) 
            print(y_train.shape)
            print(y_train_num.shape)

        # For shuffling, pack into one matrix and shuffle it along rows
        auxarray = np.concatenate((features_train, y_train.reshape(-1,1), y_train_num.reshape(-1,1)),axis=1)
        print(auxarray.shape)
        np.random.seed(None)  
        np.random.shuffle(auxarray)
        features_train = auxarray[:,0:features_train.shape[1]]   
        y_train = auxarray[:,features_train.shape[1]]   
        y_train_num = auxarray[:,features_train.shape[1]+1]   
        
        if (False):
            print(features_train.shape) 
            print(y_train.shape)
            print(y_train_num.shape)


        
#       
# Classifiers    
#


    

# E2 Train the Random Forest classifier model
    clf = RandomForestClassifier(n_estimators=100,max_features="auto",oob_score=True,verbose=0)
    clf.fit(features_train, y_train)
    lin()
    print(" \nTraining the Random Forest classifier")
    print('*oob_score error (training set):',1.0-clf.oob_score_,' oob_score:',clf.oob_score_)
    print('*Score (training set):',clf.score(features_train, y_train))
    print('feature names:      ',featureNames)
    print('feature importances:',clf.feature_importances_)
    cross_val(clf, features_train, y_train, featureNames)
    test(clf, features_test, y_test)

    lin()

# E1b Study the effect of the training set size on CV score
# - do CV (also evaluate test data, but should not be considered here)
# - seems that maximum amount of training data available (all - test data) is generally good
# - studied for examples in the polymer article
#
# Max size is 119 - sizeTestSet(=12) = 107, and then y_train is the original y_train   
    print('Study the effect of amount of training data')           
    print('Result: Generally maximum amount of training data should be used')
    if(False):                           
        trainSize = np.array([70, 80, 90, len(y_train)])
        print('shape y_train:', y_train.shape)
        print(trainSize)
        errors = []
        for i in trainSize:
            features_train_par = features_train[:i, :] 
            y_train_par = y_train[:i]
            clf.fit(features_train, y_train)
            par1, ___ = cross_val(clf, features_train_par, y_train_par, featureNames)
            par2 = test(clf, features_test, y_test)
            errors.append([par1, par2])
            err = np.array(errors).reshape(-1,2)
        print('accuracies CV, test:\n',err)
    
        import matplotlib.pyplot as plt
        plt.figure()
        print(err.shape, trainSize.shape)
        print(trainSize.reshape(-1,1).ravel())
        print(err[0,:])
        plt.plot(trainSize, err[:,0], 'o-', label="cv")
        plt.plot(trainSize, err[:,1], 'o-', label='test')
        plt.legend()
        plt.show()

    
#    def collectErrorList():
#        siz_test = 12 
#        siz_train = np.array([60, 70, 80, 90, 100, 107])
#        ecv = np.array([0.7, 0.729, 0.673, 0.6, 0.4])
#        et = np.array([0.917, 0.583, 0.833, 0.333, 0.75 ])
 


# G Grid search for Random Forest parameters
    doGridsearch = False
    if(doGridsearch):
        # Search the grid of parameters
        print("Performing grid search for parameters")
        print("Size of the test set (not used in training):",sizeTestSet)
        gridtestClf(features_train, y_train, featureNames)
    else:
        print("No grid search for parameters")


# G1 How sensitive is the model to test data variation. Thhis 
# could be done with random sampling from validation set. 
# However, since cross-validation is used, it already indicates 
# something on the sensitivity



# G2 Final training of the classifier, with final selection
    lin()
    lin()
    print("FINAL SELECTION Training the Random Forest classifier, FINAL SELECTION")
    # Cross-validation part (random_state = None)
    print("CROSS-VALIDATION DATA:")
    clf = RandomForestClassifier(n_estimators=100, max_features="auto", oob_score=True, verbose=0, random_state=None)
    clf.fit(features_train, y_train)
    cross_val(clf, features_train, y_train, featureNames)

    cross_val_scores = np.array([0.8132, 0.8047, 0.7637, 0.7947, 0.7921, 0.7932, 0.8147, 0.7842])
    std = np.round(np.std(cross_val_scores, ddof=1), 4)
    sem = std / np.sqrt(len(cross_val_scores))  # standard error of the mean
    print('*Ave, std, std-of-mean of CV scores:', np.mean(cross_val_scores), '/', std, '/', sem)
    
    # Test set part
    clf = RandomForestClassifier(n_estimators=100, max_features="auto", oob_score=True, verbose=0, random_state=1)
    clf.fit(features_train, y_train)
    print("TEST DATA: fix random_state to always get the same model")
    print('*oob_score error (training set):',1.0-clf.oob_score_,' oob_score:',clf.oob_score_)
    print('*Score (training set):',clf.score(features_train, y_train))
    print('feature names:      ',featureNames)
    print('feature importances:',clf.feature_importances_)
    test(clf, features_test, y_test)
    lin()

    
# E2.2 Train the logistic-reg classifier 
    lin()
    print(" \nTraining the logistic regression classifier")
    logclf = LogisticRegression()
    logclf.fit(features_train, y_train)
    print('*log reg score (training set):',logclf.score(features_train, y_train))
    cross_val(logclf, features_train, y_train, featureNames)
    test(logclf, features_test, y_test)


# E2.3 Train the decision tree classifier
    lin()
    print(" \nTraining the decision tree classifier")
    dtclf = DecisionTreeClassifier()
    dtclf.fit(features_train, y_train)
    dtclf_score = dtclf.score(features_train, y_train)
    print('*decision tree score (training set):',dtclf_score)
    cross_val(dtclf, features_train, y_train, featureNames)
    test(dtclf, features_test, y_test)


#       
# Regressors    
#

    lin()
    lin()
    lin()


# Train the RF regressor
# http://stackoverflow.com/questions/20095187/regression-trees-or-random-forest-regressor-with-categorical-inputs
    print(" \nTraining the Random Forest regressor")
    # reg = RandomForestRegressor(n_estimators=100, min_samples_split=1, oob_score=True,)
    reg = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=1)
    reg.fit(features_train,y_train_num)
    print('*oob_score error (training set):',1.0-reg.oob_score_,' oob_score:',reg.oob_score_)
    print('*Score (R2) (training set)', reg.score(features_train, y_train_num))
    print('feature importances:',reg.feature_importances_)
    test(reg, features_test, y_test_num)


# Train the linear regressor
    lin()
    print(" \nTraining the linear regressor")
    linreg = LinearRegression()
    linreg.fit(features_train, y_train_num)
    print('*Score (R2) (training set)', linreg.score(features_train, y_train_num))
    test(linreg, features_test, y_test_num)


# Print results for 7 samples
    # printy(y_test_num[:7], (-999,-999))
    printDiag(y_test_num, reg.predict(features_test), y_train_num, reg.predict(features_train) )
    

# Test the RF regressor with test set and make a plot (printy)
#    if(sizeTestSet > 1):
#        print("\nTesting the regressor with test set")
#        preds=reg.predict(features_test)
#        preds_true_list = np.concatenate((preds.reshape(-1,1), y_test_num.reshape(-1,1)), axis=1)
#        for value in preds_true_list:
#            print("Preds, True:", value)
#        printy(preds, y_test_num)

        
        
# Generate more predictions (reg and clf) from user input    
    if(False): 
        genPredictions(reg, clf, featureNames)
        
        
    print("End-main")  

    # Temporary stop if needed: 
    # assert True==False, "Temporary stop"




if __name__ == '__main__':
    main()

# Next ideas
# - check which cases are badly predicted and train more


# Dump

# F Test the Random Forest classifier model, oob_score, search the grid of parameters
#    if(sizeTestSet > 1):        
#        print('Test with the true testing set, size:', y_test.size)  
#        preds=clf.predict(features_test)
#        print("Preds:", preds)
#        print("True:", y_test)


