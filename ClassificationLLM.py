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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score,accuracy_score

# Use the OpenAI client library to add your API key.
load_dotenv(dotenv_path="APIKEY.env")
client = OpenAI(
    api_key = os.getenv("APIKEY")
)


#find dataset with 1000 features (genes?)
df = pd.read_csv("Swarm_Behaviour.csv")
y = df["Swarm_Behaviour"]


#feed names of these features to ai and ask it to narrow down to top 100 most important (format so that code can read it?)
df = df.drop(columns=["Swarm_Behaviour"])
headers = df.columns.tolist()

TRIALS = 10
currTrial = 0
acc = list()
f1 = list()
rocauc = list()
while currTrial < TRIALS:
    currTrial += 1
    #now feed headers and context to chatgpt and ask it to return which n features to include in readable format
    n = 100 # f string doesn't work for some reason
    with open("contextSwarm.txt","r") as f:
        context = f.read()
    #get full response
    response = client.chat.completions.create(
                model = "gpt-3.5-turbo",
                messages=[{"role":"developer","content": context + """Your Task:
                            Please print only a list of exactly 100 features based on the above data in a python readable format maintaining the exact feature names while not changing capitalization, 
                        For example, when given a list of features: feature1 FeaTure2 ftr3 : you would return: feature1 FeaTure2 ftr3 . Also, selecting these features from the following based on their 
                        relevance and likelyhood to predict the variable given by and using the context. Let me stress again the importance of returning exactly 100 features without changing their names."""},
                        {"role":"user","content":" ".join(headers)},
                ],
            )
    #get chosen features
    LLMfeatures = response.choices[0].message.content

    print(LLMfeatures)

    #run model for results

    #isolate new headers from llm
    newHeaders = LLMfeatures.split()

    #if the llm returns more than 100 features then cut it off
    newHeaders = newHeaders[:100]

    newdf=pd.DataFrame()
    valid_cols = [col for col in newHeaders if col in df.columns]
    newdf = df[valid_cols].copy()

    print("Number of columsn:" ,len(newdf.columns))

    #fillna for numerical 
    numerical_cols = newdf.select_dtypes(include=['int','float','double','long']).columns.tolist()

    for i in range(len(numerical_cols)):
        newdf[numerical_cols[i]] = newdf[numerical_cols[i]].fillna(newdf[numerical_cols[i]].mean())

    #convert categorical columns to numerical and fillna
    categorical_cols = newdf.select_dtypes(include=['object', 'category']).columns.tolist()

    for i in range(len(categorical_cols)):
        newdf[categorical_cols[i]] = newdf[categorical_cols[i]].astype(str)
        newdf[categorical_cols[i]] = newdf[categorical_cols[i]].fillna(newdf[categorical_cols[i]].mode().iloc[0])
        #newdf[categorical_cols[i]] = le.fit_transform(newdf[categorical_cols[i]]) #label encoding
        newdf = pd.concat([newdf.drop(columns=categorical_cols[i]),pd.get_dummies(newdf, columns=[categorical_cols[i]])],axis=1) #one hot encoding

    #machine learning or linear regression
    X_train, X_test, y_train, y_test = train_test_split(newdf, y, test_size=0.2)

    # Fit logistic regression model
    model = LogisticRegression(penalty='l1', solver='liblinear')
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)

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