#normal lasso to compare
#use open ai to narrow down 1000 features to 100, then narrow down those features with best subset selection
from openai import OpenAI
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score,accuracy_score

#find dataset with 1000 features (genes?)
df = pd.read_csv("SwarmBehaviour/Swarm_Behaviour.csv")
y = df["Swarm_Behaviour"]

#feed names of these features to ai and ask it to narrow down to top 100 most important (format so that code can read it?)
df = df.drop(columns=["Swarm_Behaviour"])
headers = df.columns.tolist()

#run model for results

#isolate new headers from llm
#newHeaders = ["mutation_count","age_at_diagnosis","cancer_type_detailed","chemotherapy","hormone_therapy","lymph_nodes_examined_positive","pr_status","tumor_size","tumor_stage"]
#newHeaders = headers
TRIALS = 10
currTrial = 0
acc = list()
f1 = list()
rocauc = list()
while currTrial < TRIALS:
    currTrial += 1
    #randomly select 100 columns
    newHeaders = random.sample(headers,100)

    newdf=pd.DataFrame()
    valid_cols = [col for col in newHeaders if col in df.columns]
    newdf = df[valid_cols].copy()

    #fillna for numerical 
    numerical_cols = newdf.select_dtypes(include=['int','float','double','long']).columns.tolist()

    for i in range(len(numerical_cols)):
        newdf[numerical_cols[i]] = newdf[numerical_cols[i]].fillna(newdf[numerical_cols[i]].mean())

    #convert categorical columns to numerical and fillna
    categorical_cols = newdf.select_dtypes(include=['object', 'category']).columns.tolist()

    for i in range(len(categorical_cols)):
        newdf[categorical_cols[i]] = newdf[categorical_cols[i]].astype(str)
        newdf[categorical_cols[i]] = newdf[categorical_cols[i]].fillna(newdf[categorical_cols[i]].mode().iloc[0])
        #newdf[categorical_cols[i]] = le.fit_transform(newdf[categorical_cols[i]])
        newdf = pd.concat([newdf.drop(columns=categorical_cols[i]),pd.get_dummies(newdf, columns=[categorical_cols[i]])],axis=1)

    #machine learning or linear regression
    X_train, X_test, y_train, y_test = train_test_split(newdf, y, test_size=0.2)

    # Fit Lasso model
    lasso = LogisticRegression(penalty='l1', solver='liblinear')
    lasso.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = lasso.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    acc.append(accuracy_score(y_test, y_pred))
    print("F1 score:", f1_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred))
    print("ROC-AUC score:", roc_auc_score(y_test, y_pred))
    rocauc.append(roc_auc_score(y_test, y_pred))

data = {
    'acc': acc,
    'f1': f1,
    'roc-auc': rocauc
}

outputdf = pd.DataFrame(data)
outputdf.to_csv('output.csv', index=True)