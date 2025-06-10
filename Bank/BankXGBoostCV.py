#for spotify dataset
#use open ai to narrow down 1000 features to 100, then narrow down those features with best subset selection
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
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import f1_score, roc_auc_score,accuracy_score
import xgboost as xgb


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
                model = "gpt-3.5-turbo",
                messages=[{"role":"developer","content": context + f"""Your Task:
                            Please print only a list of exactly {n} features based on the above data in a csv format, seperating features with a comma then a space, 
                           maintaining the exact feature names while not changing capitalization.
                        For example, when given a list of features: feature1, FeaTure2, ftr3 : you would return a csv with the selected features, possibly including: feature1, FeaTure2, ftr3, etc. 
                           Also, selecting these features from the following based on their relevance and likelyhood to predict the variable given by and using the context. 
                           Let me stress again the importance of returning exactly {n} features without changing their names from the given input in any way, and returning this comma seperated table as specified."""},
                        {"role":"user","content":" ".join(features)},
                ],
            )
    #get chosen features
    LLMfeatures = response.choices[0].message.content

    print(LLMfeatures)
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
    valid_cols = valid_cols[:featuresToGet] #cut off any extra columns if llm included too many
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

def L0_regression(features, response, folds=5, standardize=False, seed=None):
    """
    Select the best L0-Regression model by performing grid search on the budget.
    """
    dim = features.shape[1]
    best_mse = np.inf
    best = 0
    # Grid search to find best number of features to consider
    for i in range(1, dim + 1):
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

# Use the OpenAI client library to add your API key.
load_dotenv(dotenv_path="APIKEY.env")
client = OpenAI(
    api_key = os.getenv("APIKEY")
)

#find dataset with 1000 features (genes?)
df = pd.read_csv("bank-additional-full.csv",sep=";") 
#clean data 
#take out quotes and commas from following

#get numerilc cols 
numerical_cols = df.select_dtypes(include='number').columns.tolist()
#fillna for numerical
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

#convert categorical columns to numerical and fillna
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
df[categorical_cols] = df[categorical_cols].astype(str)
#label encode
for col in categorical_cols:
    if df[col].nunique() > 2:
        #one hot
        df = pd.get_dummies(df, columns=[col])
    else:
        #label encode
        df[col] = le.fit_transform(df[col])


#separate target and df
y = df["y"]
df = df.drop(columns=["y"])

TRIALS = 10
currTrial = 0
acc = list()
f1 = list()
rocauc = list()
while currTrial < TRIALS:
    currTrial += 1
    
    #get newdf with chosen columns using llm
    newdf = NarrowDownDFLLM(df,"Bank/contextBank.txt",15) #here is where you specify how many features the LLM should choose
    #newdf = df
    
    print("Number of columsn:" ,len(newdf.columns))
    if len(newdf.columns) < 1:
        continue

    #split data
    import secrets #import secrets to get better random seed
    X_train, X_test, y_train, y_test = train_test_split(newdf, y, test_size=0.2, random_state = secrets.randbelow(1_000_000))

    #standardize test and train sep
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Convert to DMatrix (XGBoost's internal format)
    dtrain = xgb.DMatrix(X_train_std, label=y_train)

    # Estimate class imbalance
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    # Set parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': scale_pos_weight,
        'learning_rate': 0.1,
        'max_depth': 4
    }

    # 5-fold CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=100,
        nfold=5,
        stratified=True,
        metrics='auc',
        early_stopping_rounds=10
    )

    best_n_rounds = len(cv_results)
    final_model = xgb.train(
        params,
        dtrain,
        num_boost_round=best_n_rounds
    )

    # To predict:
    dtest = xgb.DMatrix(X_test_std)
    y_pred_proba = final_model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)


    acc.append(accuracy_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred))
    print(f1_score(y_test, y_pred))
    rocauc.append(roc_auc_score(y_test, y_pred))

data = {
    'acc': acc,
    'f1': f1,
    "rocauc":rocauc
}

outputdf = pd.DataFrame(data)
outputdf.to_csv('output.csv', index=True)