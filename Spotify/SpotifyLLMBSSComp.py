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
        regressor.params.timelimit = 60
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
    if standardize:
        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)
    intercept, beta = miqp(features, response, best)
    return intercept, beta
    
def TrainAndAppendResults(df,y,seed,featureAmount,r2s,mses,timing,finalFeaturesChosen):
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
    intercept, coefficients = L0_regression(X_train_std,y_train.to_numpy(),featureAmount,standardize=True,seed=seed) #set seed and feature as max k 

    # Predict and evaluate (@ is matrix multiplication) #headers? array types?
    y_pred = (X_test_std @ coefficients) +intercept

    end = time.perf_counter()

    r2s.append(r2_score(y_test, y_pred))
    mses.append(mean_squared_error(y_test, y_pred))
    timing.append(end -start)
    finalFeaturesChosen.append(sum(1 for coef in coefficients if coef != 0))#sum the amount of features actually used the final model
    return coefficients

# Use the OpenAI client library to add your API key.
load_dotenv(dotenv_path="APIKEY.env")
client = OpenAI(
    api_key = os.getenv("APIKEY")
)

#--------------------------------------------------DATA CLEANING-------------------------------------------------------

#find dataset with 1000 features (genes?)
df = pd.read_csv("Spotify/Most Streamed Spotify Songs 2024.csv",encoding='ISO-8859-1') 
#drop rows where the target is na
df = df[~df["Spotify Streams"].isna()]


#drop stupid TIDAL colun (all nOne)
df = df.drop(columns="TIDAL Popularity")

#clean data and standardize
df["Release Date"] = pd.to_datetime(df["Release Date"])#turn to datetime Extract features like year, month, day, or even weekday

df['Year'] = df['Release Date'].dt.year
df['Month'] = df['Release Date'].dt.month
df['Day'] = df['Release Date'].dt.day
df['Weekday'] = df['Release Date'].dt.weekday

df = df.drop(columns="Release Date")

#df["ISRC"] = #sepearte into parts 
df["Country Code"] = df["ISRC"].str[:2]
df["Registrant Code"] = df["ISRC"].str[2:5]
df["Designation Code"] = df["ISRC"].str[7:12]

df = df.drop(columns="ISRC")
df = df.drop(columns="Designation Code")
#take out quotes and commas from following
for column in ["Spotify Streams","Spotify Playlist Count","Spotify Playlist Reach",
               "YouTube Views","YouTube Likes","TikTok Posts","TikTok Likes","TikTok Views",
               "YouTube Playlist Reach","AirPlay Spins","Deezer Playlist Reach","Pandora Streams",
               "Pandora Track Stations","Soundcloud Streams","Shazam Counts"]:
    df[column] = df[column].str.replace(',', '', regex=False).astype('Float64') #use Int64 for null values

#get numerilc cols 
numerical_cols = df.select_dtypes(include='number').columns.tolist()
#fillna for numerical
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

#convert categorical columns to numerical and fillna
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

for i in range(len(categorical_cols)):
    df[categorical_cols[i]] = df[categorical_cols[i]].astype(str)
    df[categorical_cols[i]] = df[categorical_cols[i]].fillna(df[categorical_cols[i]].mode().iloc[0])
    df[categorical_cols[i]] = le.fit_transform(df[categorical_cols[i]]) #label encoding
    #newdf = pd.concat([newdf.drop(columns=categorical_cols[i]),pd.get_dummies(newdf, columns=[categorical_cols[i]])],axis=1) #one hot encoding

#separate target and df
y = df["Spotify Streams"]
df = df.drop(columns=["Spotify Streams"])

#--------------------------------------------------MODEL TRAINING-------------------------------------------------------


TRIALS = 10 #this number of trials for each unique combination of feature amount and model type
FEATURES = [10,15] #list of features to try [10,15,20]

for featureAmount in FEATURES:
    #initialize lists to keep track of data
    BSSr2s = list()
    LLMr2s = list()
    Randr2s = list()

    BSSmses = list()
    LLMmses = list()
    Randmses = list()

    BSStiming = list()
    LLMtiming = list()
    Randtiming = list()

    featuresSpecified = [featureAmount] *TRIALS #make a TRIAL long list of the number 'feature'

    featuresChosenByLLM = list()

    BSSfinalFeaturesChosen = list()
    LLMfinalFeaturesChosen = list()
    RandfinalFeaturesChosen = list()

    LLMMatchFeaturesToBSS = list()
    RandMatchFeaturesToBSS = list()

    currTrial = 0
    while currTrial < TRIALS:
        #///////[BSS]\\\\\\\
        #original df
        BSSdf = df
        
        BSSCoef = TrainAndAppendResults(BSSdf,y,currTrial,featureAmount,BSSr2s,BSSmses,BSStiming,BSSfinalFeaturesChosen)

        #find the names of the features chosen by BSS
        BSSChosenFeatureNames = list()
        for i in range(len(BSSdf.columns)):
            if BSSCoef[i] != 0:
                BSSChosenFeatureNames.append(BSSdf.columns[i])

        #///////[LLM]\\\\\\\
        #get newdf with chosen columns using llm 
        LLMdf = NarrowDownDFLLM(df,"Spotify/contextSpotify.txt",featureAmount) #here is where you specify how many features the LLM should choose
        
        #find number of features chosen by llm, make sure its not 0
        llmFeatureAmount = len(LLMdf.columns)
        print("Number of columsn:" ,llmFeatureAmount)
        if llmFeatureAmount < 1:
            print(f"LLM didn't give any features")
            continue
        featuresChosenByLLM.append(llmFeatureAmount)

        #find matched features with BSS
        totalMatched = 0
        for feature in LLMdf.columns:
            if feature in BSSChosenFeatureNames:
                totalMatched += 1
        LLMMatchFeaturesToBSS.append(totalMatched/featureAmount)

        TrainAndAppendResults(LLMdf,y,currTrial,featureAmount,LLMr2s,LLMmses,LLMtiming,LLMfinalFeaturesChosen)

        #///////[Random]\\\\\\\
        Randdf = df[random.sample(df.columns.tolist(),featureAmount)].copy()

        #find matched features with BSS
        totalMatched = 0
        for feature in Randdf.columns:
            if feature in BSSChosenFeatureNames:
                totalMatched += 1
        RandMatchFeaturesToBSS.append(totalMatched/featureAmount)

        TrainAndAppendResults(Randdf,y,currTrial,featureAmount,Randr2s,Randmses,Randtiming,RandfinalFeaturesChosen)
        
        currTrial += 1

    outputdfBSS = pd.DataFrame({
            'r2': BSSr2s,
            'mse': BSSmses,
            'rmse (Spotify Streams)': np.sqrt(BSSmses),
            "time (sec)": BSStiming,
            "features specified": featuresSpecified,
            "features chosen through BSS":BSSfinalFeaturesChosen
        })
    outputdfBSS.to_csv(f'outputBSSp{featureAmount}.csv', index=True)
    
    outputdfLLM = pd.DataFrame({
            'r2': LLMr2s,
            'mse': LLMmses,
            'rmse (Spotify Streams)': np.sqrt(LLMmses),
            "time (sec)": LLMtiming,
            "features specified": featuresSpecified,
            "features chosen by LLM": featuresChosenByLLM, #extra column that tells how many features the llm returns (should be equal to features specified, but may not be if LLM didn't listen)
            "features chosen through BSS":LLMfinalFeaturesChosen,
            "features matched to BSS": LLMMatchFeaturesToBSS #optimal features chosen (?)
        })
    outputdfLLM.to_csv(f'outputLLMp{featureAmount}.csv', index=True)

    outputdfRand = pd.DataFrame({
            'r2': Randr2s,
            'mse': Randmses,
            'rmse (Spotify Streams)': np.sqrt(Randmses),
            "time (sec)": Randtiming,
            "features specified": featuresSpecified,
            "features chosen through BSS":RandfinalFeaturesChosen,
            "features matched to BSS": RandMatchFeaturesToBSS #optimal features chosen (?)
        })
    outputdfRand.to_csv(f'outputRandp{featureAmount}.csv', index=True)