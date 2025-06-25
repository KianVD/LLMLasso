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

def gurobiSVM(features, response):
    # Create a Gurobi environment and a model object
    with gp.Model("", env=env) as model:
        samples, dim = features.shape
        assert samples == response.shape[0]

        #coefficients
        a = [model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"a{i}") for i in range(features.shape[1])]
        
        # intercept
        beta = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="beta")


        #slack variables
        slack = [model.addVar(name=f"xi{i}") for i in range(samples)]
        
        
        #constraints
        for i in range(samples):
            model.addConstr(
                response[i] * (quicksum(a[j]*features[i][j] for j in range(dim)) - beta) >= 1 - slack[i],
                name=f"margin_{i}"
            )

        gamma = 1 #cv this
        model.setObjective(
            quicksum(a[j]*a[j] for j in range(dim)) + gamma * quicksum(slack),
            GRB.MINIMIZE
        )

        model.setParam('OutputFlag', 1)
        #t1 = time.time()
        model.optimize()
        #tfinal = time.time() - t1
            
        equation = {}
        if model.Status == GRB.OPTIMAL:
            equation['a'] = [a[j].X for j in range(features.shape[1])]
            equation['beta'] = beta.X

        return equation


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

    #or with cross validation
    equation = gurobiSVM(X_train_std, y_train.to_numpy())#uses featureAmount for k, or col dim if smaller
    # Predict and evaluate (@ is matrix multiplication) #headers? array types?
    # Decision values
    decision_scores = X_test_std @ equation["a"] - equation["beta"]  # (dot product of each row with a) - beta

    #convert to bipolar
    y_pred = (decision_scores > 0).astype(int)
    y_pred = np.where(y_pred == 0, -1, 1)

    end = time.perf_counter()
    results[model]["acc"].append(accuracy_score(y_test, y_pred))
    results[model]["roc"].append(roc_auc_score(y_test, y_pred))
    results[model]["f1"].append(f1_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NonToxic','Toxic'])
    disp.plot()
    plt.show()

    results[model]["timing"].append(end -start)
    results[model]["finalFeaturesChosen"].append(sum(1 for coef in equation["a"] if coef != 0))#sum the amount of features actually used the final model
    return equation["a"]

def match_features(givenFeatures,otherFeatures,featureAmount):
    totalMatched = sum(1 for feature in givenFeatures if feature in otherFeatures)
    return totalMatched/featureAmount

def save_results(results,featuresSpecified,featureAmount,ModelName):
    output = {
            'acc': results[ModelName]['acc'],
            'roc': results[ModelName]['roc'],
            'f1': results[ModelName]["f1"],
            "time (sec)": results[ModelName]['timing'],
            "features specified": featuresSpecified,
            "features used in BSS":results[ModelName]["finalFeaturesChosen"]
        }
    if (ModelName in ["LLM","Rand"]) and ("matched features" in results[ModelName]):
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
            if "matched features" not in results[model]:
                results[model]["matched features"] = list()
            results[model]["matched features"].append(match_features(currdf.columns,otherFeatureNames,featureAmount))
                


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


TRIALS = 10 #this number of trials for each unique combination of feature amount and model type
FEATURES = [9] #list of features to try [10,15,20]

for featureAmount in FEATURES:
    #initialize lists to keep track of data

    featuresSpecified = [featureAmount] *TRIALS #make a TRIAL long list of the number 'feature'

    results = {
        'BSS' : {"acc":[],"roc":[],"f1":[],"timing": [],"finalFeaturesChosen":[]},
        'LLM' : {"acc":[],"roc":[],"f1":[],"timing": [],"finalFeaturesChosen":[],"featuresChosenByLLM":[]},
        'Rand' : {"acc":[],"roc":[],"f1":[],"timing": [],"finalFeaturesChosen":[]}
    }


    currTrial = 0
    while currTrial < TRIALS:
        #///////[BSS]\\\\\\\
        BSSChosenFeatureNames = run_trial("BSS",df,y,currTrial,featureAmount,results) 

        #///////[LLM]\\\\\\\
        run_trial("LLM",df,y,currTrial,featureAmount,results,contextFile="Toxicity/contextToxicity.txt",otherFeatureNames=BSSChosenFeatureNames)


        #///////[Rand]\\\\\\\
        run_trial("Rand",df,y,currTrial,featureAmount,results,otherFeatureNames=BSSChosenFeatureNames)
        
        currTrial += 1

    for model in ["BSS","LLM","Rand"]:
        save_results(results,featuresSpecified,featureAmount,model)