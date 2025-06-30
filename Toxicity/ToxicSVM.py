#for spotify dataset
#includes timing and new gurobi functions with max k
from openai import OpenAI
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import gurobipy as gp
from gurobipy import GRB
from sklearn.preprocessing import StandardScaler
import time
import random
from gurobipy import quicksum
from sklearn.metrics import f1_score, roc_auc_score,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
random.seed(0)


options = {
"WLSACCESSID":"a9ee3346-4b70-4d35-a517-fe1941ffe2ef",
"WLSSECRET":"8c55cef6-a34c-436b-ab49-7de37868044b",
"LICENSEID":2674818,
}

env = gp.Env(params=options)

def GetLLMFeatures(contextFilepath, featuresToGet, features):
    #now feed headers and context to chatgpt and ask it to return which n features to include in readable format
    n = featuresToGet # f string doesn't work for some reason
    with open(contextFilepath,"r") as f:
        context = f.read()
    #get full response
    response = client.chat.completions.create(
                model = "gpt-3.5-turbo", #gpt-3.5-turbo, gpt-4o
                messages=[{"role":"developer","content": context + f"""Your Task:
                            Please print only a list of the available features in the order of their significance to predicting the desired variable, listing the most significant first, based on the above data. 
                           This list should be in a csv format, seperating features with a comma then a space, maintaining the exact feature names including capitalization.
                            For example, when given a list of features: FeaTure2, feature1, ftr3 : you would return the following: feature1, FeaTure2, ftr3, etc. in that format, ordered by significance.
                           These features should be selected based on their relevance and likelyhood to predict the variable given by and using the context. 
                           At least {n} of the available features should be returned. The only available features to be picked are given by the user, following this message."""},
                        {"role":"user","content":", ".join(features)},
                ],
            )
    #get chosen features
    LLMfeatures = response.choices[0].message.content

    #print(LLMfeatures)
    finalFeatures = LLMfeatures.split(", ")

    return finalFeatures

def NarrowDownDFLLM(df,contextFilePath, featuresToGet):
    headers = df.columns.tolist()

    #get features chosen by llm
    newHeaders = GetLLMFeatures(contextFilePath, featuresToGet,headers)

    valid_cols = list()
    for col in newHeaders:
        if col in df.columns and col not in valid_cols:
            valid_cols.append(col)
    valid_cols = valid_cols[:featuresToGet] #cut off any extra columns if llm included too many (they are ranked in order of importance so least important get cut off first )
    return df[valid_cols].copy()

def gurobiSVM(X, y,k,gamma=1.0,M=1000,L0Regularization=False):
    # Create a Gurobi environment and a model object
    with gp.Model("", env=env) as model:
        samples, features = X.shape
        assert samples == y.shape[0]

        #coefficients
        a = [model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"a{i}") for i in range(features)]
        
        # intercept
        beta = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="beta")


        #slack variables
        slack = [model.addVar(lb=0,name=f"xi{i}") for i in range(samples)]

        #l0 norm selectors
        if(L0Regularization):
            z = model.addVars(features, vtype=GRB.BINARY, name="z")
        
            # Link w and z constraints
            for j in range(features):
                model.addConstr(a[j] <= M * z[j])
                model.addConstr(a[j] >= -M * z[j])

            #costrain max k
            model.addConstr(quicksum(z[j] for j in range(features)) <= k, name="feature_budget")
        
        #constraints
        for i in range(samples):
            model.addConstr(
                y[i] * (quicksum(a[j]*X[i][j] for j in range(features)) - beta) >= 1 - slack[i],
                name=f"margin_{i}"
            )

        model.setObjective(
            quicksum(a[j]*a[j] for j in range(features)) + gamma * quicksum(slack),
            GRB.MINIMIZE
        )

        model.setParam('OutputFlag', 1)
        model.params.timelimit = 60
        model.params.mipgap = 0.001

        model.optimize()
        #print(model.Status)
            
        equation = {}
        if model.SolCount > 0:
            equation['a'] = [a[j].X for j in range(features)]
            equation['beta'] = beta.X
        else:
            #didnt converge, return other apporximatination
            print("Optimization did not converge")
            equation['a'] = [1 for _ in range(features)]
            equation['beta'] = 1
        return equation

def split_folds(features, response, train_mask):
    """
    Assign folds to either train or test partitions based on train_mask.
    """
    xtrain = features[train_mask,:]
    xtest = features[~train_mask,:]
    ytrain = response[train_mask]
    ytest = response[~train_mask]
    return xtrain, xtest, ytrain, ytest

def cross_validate(features, response, k, folds, standardize, seed,gamma=1):
    """
    Train an L0-Regression for each fold and report the cross-validated MSE.
    """
    if seed is not None:
        np.random.seed(seed)
    samples, dim = features.shape
    assert samples == response.shape[0]
    fold_size = int(np.ceil(samples / folds))
    # Randomly assign each sample to a fold
    shuffled = np.random.choice(samples, samples, replace=False)
    roc = 0
    # Exclude folds from training, one at a time, 
    #to get out-of-sample estimates of the roc
    for fold in range(folds):
        idx = shuffled[fold * fold_size : min((fold + 1) * fold_size, samples)]
        train_mask = np.ones(samples, dtype=bool)
        train_mask[idx] = False
        xtrain, xtest, ytrain, ytest = split_folds(features, response, train_mask)
        if standardize:
            scaler = StandardScaler()
            scaler.fit(xtrain)
            xtrain = scaler.transform(xtrain)
            xtest = scaler.transform(xtest)
        equation = gurobiSVM(xtrain, ytrain,k,gamma=gamma,M=1000)
        ypred = findYPred(xtest,equation)
        roc += roc_auc_score(ytest, ypred) / folds
    # Report the average out-of-sample roc
    return roc

def GridSearchK(features, response, maxfeatures,folds=2, standardize=False, seed=None):
    """
    Select the best L0-Regression model by performing grid search on the budget.
    """
    dim = features.shape[1]
    best_roc = np.inf
    best = 0
    #Find highest possible features
    max_k = dim if maxfeatures >= dim else maxfeatures
    # Grid search to find best number of features to consider
    for i in range(1, max_k + 1,10):
        val = cross_validate(features, response, i, folds=folds,
                             standardize=standardize, seed=seed)
        if val < best_roc:
            best_roc = val
            best = i
    if standardize:
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)
    equation = gurobiSVM(features, response,best,gamma=1,M=1000)
    return equation

def GridSearchG(features, response, maxfeatures,folds=5, standardize=False, seed=None):
    """
    Select the best gamma by performing grid search on the budget.
    """
    best_roc = np.inf
    best = 0
    # Grid search to find best number of features to consider
    for i in [.0001,.001,.01,.1,1,10]:
        val = cross_validate(features, response, maxfeatures, folds=folds,
                             standardize=standardize, seed=seed,gamma=i)
        if val < best_roc:
            best_roc = val
            best = i
    if standardize:
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)
    print(best)
    equation = gurobiSVM(features, response,maxfeatures,gamma=best,M=1000)
    return equation



def findYPred(X,equation):
    decision_scores = X @ equation["a"] - equation["beta"]  # (dot product of each row with a) - beta

    #convert to bipolar
    y_pred = (decision_scores > 0).astype(int)
    return np.where(y_pred == 0, -1, 1)


def TrainAppendResults(df,y,k,seed,results,model):
    #split, standardize, train bss, and predict on specified df and seed, and append data to specified lists

    #split data, bss first
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y,random_state = seed)

    #standardize test and train sep
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    #or with cross validation
    equation = gurobiSVM(X_train_std, y_train.to_numpy(),50,gamma=1,M=1000,L0Regularization=True)#uses featureAmount for k, or col dim if smaller
    #equation = GridSearchK(X_train_std,y_train.to_numpy(),k,standardize=False,seed=seed)
    #equation = GridSearchG(X_train_std,y_train.to_numpy(),k,standardize=False,seed=seed)
    # Predict and evaluate (@ is matrix multiplication) #headers? array types?

    y_pred = findYPred(X_test_std,equation)

    results[model]["acc"].append(accuracy_score(y_test, y_pred))
    results[model]["roc"].append(roc_auc_score(y_test, y_pred))
    results[model]["f1"].append(f1_score(y_test, y_pred))

    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NonToxic','Toxic'])
    # disp.plot()
    # plt.show()

    #add training results
    y_pred = findYPred(X_train_std,equation)

    results[f"{model}train"]["acc"].append(accuracy_score(y_train, y_pred))
    results[f"{model}train"]["roc"].append(roc_auc_score(y_train, y_pred))
    results[f"{model}train"]["f1"].append(f1_score(y_train, y_pred))

    #return weights to use for matched feature comparison
    return equation["a"]


def match_features(givenFeatures,otherFeatures):
    """otherFeatures is the features that givenFeatures is being compared to (SVM)"""
    totalMatched = sum(1 for feature in givenFeatures if feature in otherFeatures)
    return totalMatched/len(givenFeatures)

def save_results(results,featuresSpecified,ModelName):
    output = {
            'acc': results[ModelName]['acc'],
            'roc': results[ModelName]['roc'],
            'f1': results[ModelName]["f1"],
            "time (sec)": results[ModelName]['timing'],
            "features specified": featuresSpecified
        }
    if ModelName in ["LLM","LLMtrain"]:
        output["features chosen by LLM"] = results[ModelName]["featuresChosenByLLM"] #extra column that tells how many features the llm returns (should be equal to features specified, but may not be if LLM didn't listen)
    if "matched features" in results[ModelName]:
        output["features matched to SVM"] = results[ModelName]["matched features"]
    pd.DataFrame(output).to_csv(f'output{ModelName}p{featuresSpecified[0]}.csv', index=True)

def run_trial(model,df,y,seed,featureAmount,results,contextFile=None,otherFeatureNames=None):
    #1 get df for specific model
    start = time.perf_counter()
    match model:
        case "SVM":
            #original df
            currdf = df
        case "LLM":
            #get newdf with chosen columns using llm 
            currdf = NarrowDownDFLLM(df,contextFile,featureAmount) #here is where you specify how many features the LLM should choose
        
            #find number of features chosen by llm, make sure its not 0
            llmFeatureAmount = currdf.shape[1]
            print("Number of columsn:" ,llmFeatureAmount)
            if llmFeatureAmount < 1:
                print(f"LLM didn't give any features") #error
            results["LLM"]["featuresChosenByLLM"].append(llmFeatureAmount)
            results["LLMtrain"]["featuresChosenByLLM"].append(llmFeatureAmount)
        case "Rand":
            currdf = df[random.sample(df.columns.tolist(),featureAmount)].copy()

    #2 trainappend results

    Coef = TrainAppendResults(currdf,y,featureAmount,seed,results,model)
    #record time of whole trial
    end = time.perf_counter()
    results[model]["timing"].append(end -start)
    results[f"{model}train"]["timing"].append(end -start)

    if model == "SVM":
        ChosenFeatureNames = list()
        for i in range(len(currdf.columns)):
            if Coef[i] != 0:
                ChosenFeatureNames.append(currdf.columns[i])
        return ChosenFeatureNames
    else:
        #find matched features with BSS
        if otherFeatureNames:
            #do just for train
            if "matched features" not in results[f"{model}train"]:
                results[f"{model}train"]["matched features"] = list()
            results[f"{model}train"]["matched features"].append(match_features(currdf.columns,otherFeatureNames))
                


# Use the OpenAI client library to add your API key.
load_dotenv(dotenv_path="APIKEY.env")
client = OpenAI(
    api_key = os.getenv("APIKEY")
)

#--------------------------------------------------DATA CLEANING-------------------------------------------------------

#find dataset with 1000 features (genes?)
df = pd.read_csv("Toxicity/ToxicityData.csv") 
#drop rows where the target is na
df = df[~df["Class"].isna()]

#get numerilc cols 
numerical_cols = df.select_dtypes(include='number').columns.tolist()
#fillna for numerical
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
#separate target and df
y = df["Class"]
df.drop("Class",axis=1,inplace=True)

#turn categorical y into binary 1 for toxic 0 for nontoxic
y = pd.Series([1 if tox == "Toxic" else -1 for tox in y])

#--------------------------------------------------MODEL TRAINING-------------------------------------------------------


TRIALS = 1 #this number of trials for each unique combination of feature amount and model type
FEATURES = [100] #list of features to try [10,15,20]

for featureAmount in FEATURES:
    #initialize lists to keep track of data

    featuresSpecified = [featureAmount] *TRIALS #make a TRIAL long list of the number 'feature'

    results = {
        'LLM' : {"acc":[],"roc":[],"f1":[],"timing": [],"featuresChosenByLLM":[]},
        'Rand' : {"acc":[],"roc":[],"f1":[],"timing": []},
        'LLMtrain' : {"acc":[],"roc":[],"f1":[],"timing": [],"featuresChosenByLLM":[]},
        'Randtrain' : {"acc":[],"roc":[],"f1":[],"timing": []}
    }


    currTrial = 0
    while currTrial < TRIALS:
        #SVMChosenFeatureNames = run_trial("SVM",df,y,currTrial,featureAmount,results) 

        #///////[LLM]\\\\\\\
        run_trial("LLM",df,y,currTrial,featureAmount,results,contextFile="Toxicity/contextToxicity.txt")#,otherFeatureNames=SVMChosenFeatureNames)


        #///////[Rand]\\\\\\\
        run_trial("Rand",df,y,currTrial,featureAmount,results)#otherFeatureNames=SVMChosenFeatureNames)
        
        currTrial += 1

    for model in ["LLM","Rand"]:
        save_results(results,featuresSpecified,model)
        save_results(results,featuresSpecified,f"{model}train")