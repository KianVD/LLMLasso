#for spotify dataset
#includes timing and new gurobi functions with max k
from openai import OpenAI
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import gurobipy as gp
from gurobipy import GRB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
import time
import random
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
                           The only available features to be picked are given by the user, following this message."""},
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

def miqp(features, response, non_zero, verbose=False):
    """
    Deploy and optimize the MIQP formulation of L0-Regression.
    """
    assert isinstance(non_zero, (int, np.integer))
    # Create a Gurobi environment and a model object
    with gp.Model("", env=env) as regressor:
        samples, dim = features.shape
        assert samples == response.shape[0]
        assert non_zero <= dim

        # Append a column of ones to the feature matrix to account for the y-intercept
        X = np.concatenate([features, np.ones((samples, 1))], axis=1)  

        # Decision variables
        norm_0 = regressor.addVar(lb=non_zero, ub=non_zero, name="norm")
        beta = regressor.addMVar((dim + 1,), lb=-GRB.INFINITY, name="beta") # Weights
        intercept = beta[dim] # Last decision variable captures the y-intercept

        regressor.setObjective(beta.T @ X.T @ X @ beta
                               - 2*response.T @ X @ beta
                               + np.dot(response, response), GRB.MINIMIZE)

        # Budget constraint based on the L0-norm
        regressor.addGenConstrNorm(norm_0, beta[:-1], which=0, name="budget")

        if not verbose:
            regressor.params.OutputFlag = 0
        regressor.params.timelimit = 60 #60
        regressor.params.mipgap = 0.001
        regressor.optimize()

        coeff = np.array([beta[i].X for i in range(dim)])
        return intercept.X, coeff

# Define functions necessary to perform hyper-parameter tuning via cross-validation

def split_folds(features, response, train_mask):
    """
    Assign folds to either train or test partitions based on train_mask.
    """
    xtrain = features[train_mask,:]
    xtest = features[~train_mask,:]
    ytrain = response[train_mask]
    ytest = response[~train_mask]
    return xtrain, xtest, ytrain, ytest

def cross_validate(features, response, non_zero, folds, standardize, seed):
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
    mse_cv = 0
    # Exclude folds from training, one at a time, 
    #to get out-of-sample estimates of the MSE
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
        intercept, beta = miqp(xtrain, ytrain, non_zero)
        ypred = np.dot(xtest, beta) + intercept
        mse_cv += mse(ytest, ypred) / folds
    # Report the average out-of-sample MSE
    return mse_cv

def L0_regression(features, response, maxfeatures,folds=5, standardize=False, seed=None):
    """
    Select the best L0-Regression model by performing grid search on the budget.
    """
    dim = features.shape[1]
    best_mse = np.inf
    best = 0
    #Find highest possible features
    max_k = dim if maxfeatures >= dim else maxfeatures
    # Grid search to find best number of features to consider
    for i in range(1, max_k + 1):
        val = cross_validate(features, response, i, folds=folds,
                             standardize=standardize, seed=seed)
        if val < best_mse:
            best_mse = val
            best = i
    best = max_k
    if standardize:
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)
    intercept, beta = miqp(features, response, best)
    return intercept, beta

def TrainAppendResults(df,y,seed,featureAmount,results,model):
    #split, standardize, train bss, and predict on specified df and seed, and append data to specified lists

    #split data, bss first
    start = time.perf_counter()
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state = seed)

    #standardize test and train sep
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    #do best subset selection
    intercept, coefficients = miqp(X_train_std, y_train.to_numpy(), min(featureAmount,X_train_std.shape[1]))#uses featureAmount for k, or col dim if smaller
    #or with cross validation
    #intercept, coefficients = L0_regression(X_train_std,y_train.to_numpy(),featureAmount,standardize=True,seed=seed) #set seed and feature as max k 

    # Predict and evaluate (@ is matrix multiplication) #headers? array types?
    y_pred = (X_test_std @ coefficients) +intercept

    end = time.perf_counter()

    results[model]["r2"].append(r2_score(y_test, y_pred))
    results[model]["mse"].append(mean_squared_error(y_test, y_pred))
    results[model]["timing"].append(end -start)
    results[model]["finalFeaturesChosen"].append(sum(1 for coef in coefficients if coef != 0))#sum the amount of features actually used the final model
    return coefficients

def match_features(givenFeatures,otherFeatures,featureAmount):
    totalMatched = sum(1 for feature in givenFeatures if feature in otherFeatures)
    return totalMatched/featureAmount

def save_results(results,featuresSpecified,featureAmount,ModelName):
    output = {
            'r2': results[ModelName]['r2'],
            'mse': results[ModelName]['mse'],
            'rmse (Spotify Streams)': np.sqrt(results[ModelName]["mse"]),
            "time (sec)": results[ModelName]['timing'],
            "features specified": featuresSpecified,
            "features used in BSS":results[ModelName]["finalFeaturesChosen"]
        }
    if ModelName in ["LLM","Rand"]:
        output["features matched to BSS"] = results[ModelName]["matched features"] #optimal features chosen (?)
    if ModelName == "LLM":
        output["features chosen by LLM"] = results[ModelName]["featuresChosenByLLM"] #extra column that tells how many features the llm returns (should be equal to features specified, but may not be if LLM didn't listen)

    pd.DataFrame(output).to_csv(f'output{ModelName}p{featureAmount}.csv', index=True)

def run_trial(model,df,y,seed,featureAmount,results,contextFile=None,otherFeatureNames=None):
    #1 get df for specific model
    match model:
        case "BSS":
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
        case "Rand":
            currdf = df[random.sample(df.columns.tolist(),featureAmount)].copy()

    #2 trainappend results

    Coef = TrainAppendResults(currdf,y,seed,featureAmount,results,model)

    #3 return
    if model == "BSS":
        BSSChosenFeatureNames = list()
        for i in range(len(currdf.columns)):
            if Coef[i] != 0:
                BSSChosenFeatureNames.append(currdf.columns[i])
        return BSSChosenFeatureNames
    else:
        #find matched features with BSS
        if otherFeatureNames:
            results[model]["matched features"].append(match_features(currdf.columns,otherFeatureNames,featureAmount))


# Use the OpenAI client library to add your API key.
load_dotenv(dotenv_path="APIKEY.env")
client = OpenAI(
    api_key = os.getenv("APIKEY")
)

#--------------------------------------------------DATA CLEANING-------------------------------------------------------

#find dataset with 1000 features (genes?)
df = pd.read_csv("RAC/RAC_train.csv") 
#drop rows where the target is na
df = df[~df["temperature"].isna()]


#drop mof cat column
df.drop('mof', axis=1, inplace=True)

#get numerilc cols 
numerical_cols = df.select_dtypes(include='number').columns.tolist()
#fillna for numerical
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
#separate target and df
y = df["temperature"]
df.drop("temperature",axis=1,inplace=True)

#--------------------------------------------------MODEL TRAINING-------------------------------------------------------


TRIALS = 10 #this number of trials for each unique combination of feature amount and model type
FEATURES = [10,30] #list of features to try [10,15,20]

for featureAmount in FEATURES:
    #initialize lists to keep track of data

    featuresSpecified = [featureAmount] *TRIALS #make a TRIAL long list of the number 'feature'

    results = {
        'BSS' : {"r2":[],"mse":[],"timing": [],"finalFeaturesChosen":[]},
        'LLM' : {"r2":[],"mse":[],"timing": [],"finalFeaturesChosen":[],"matched features":[],"featuresChosenByLLM":[]},
        'Rand' : {"r2":[],"mse":[],"timing": [],"finalFeaturesChosen":[],"matched features":[]}
    }


    currTrial = 0
    while currTrial < TRIALS:
        #///////[BSS]\\\\\\\
        BSSChosenFeatureNames = run_trial("BSS",df,y,currTrial,featureAmount,results)

        #///////[LLM]\\\\\\\
        run_trial("LLM",df,y,currTrial,featureAmount,results,contextFile="RAC/contextRac.txt",otherFeatureNames=BSSChosenFeatureNames)


        #///////[Rand]\\\\\\\
        run_trial("Rand",df,y,currTrial,featureAmount,results,otherFeatureNames=BSSChosenFeatureNames)
        
        currTrial += 1

    for model in ["BSS","LLM","Rand"]:
        save_results(results,featuresSpecified,featureAmount,model)