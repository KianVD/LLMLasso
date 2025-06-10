#use open ai to narrow down 1000 features to 100, then narrow down those features with best subset selection
from openai import OpenAI
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.linear_model import Lasso

def GetLLMFeatures(contextFilepath, featuresToGet, features):
    #now feed headers and context to chatgpt and ask it to return which n features to include in readable format
    n = featuresToGet # f string doesn't work for some reason
    with open(contextFilepath,"r") as f:
        context = f.read()
    #get full response
    response = client.chat.completions.create(
                model = "gpt-3.5-turbo",
                messages=[{"role":"developer","content": context + """Your Task:
                            Please print only a list of exactly 10 features based on the above data in a python readable format maintaining the exact feature names while not changing capitalization, 
                        For example, when given a list of features: feature1, FeaTure2, ftr3 : you would return: feature1, FeaTure2, ftr3 . Also, selecting these features from the following based on their 
                        relevance and likelyhood to predict the variable given by and using the context. Let me stress again the importance of returning exactly 10 features without changing their 
                           names from the given input in any way, and returning this comma seperated as specified"""},
                        {"role":"user","content":" ".join(features)},
                ],
            )
    #get chosen features
    LLMfeatures = response.choices[0].message.content

    print(LLMfeatures)

    #run model for results

    #isolate new headers from llm
    return LLMfeatures.split(sep=", ")

def NarrowDownDFLLM(df,contextFilePath, featuresToGet):
    headers = df.columns.tolist()

    #get features chosen by llm
    newHeaders = GetLLMFeatures(contextFilePath, featuresToGet,headers)

    valid_cols = list()
    for col in newHeaders:
        if col in df.columns and col not in valid_cols:
            valid_cols.append(col)
    valid_cols = valid_cols[:10] #cut off any extra columns if llm included too many
    return df[valid_cols].copy()


# Use the OpenAI client library to add your API key.
load_dotenv(dotenv_path="APIKEY.env")
client = OpenAI(
    api_key = os.getenv("APIKEY")
)

#find dataset with 1000 features (genes?)
df = pd.read_csv("BreastCancer/METABRIC_RNA_Mutation.csv")
y = df["overall_survival_months"]


#feed names of these features to ai and ask it to narrow down to top 100 most important (format so that code can read it?)
df = df.drop(columns=["overall_survival_months","overall_survival","death_from_cancer"])


TRIALS = 10
currTrial = 0
r2s = list()
mses = list()
while currTrial < TRIALS:
    currTrial += 1
    
    #get newdf with chosen columns using llm
    newdf = NarrowDownDFLLM(df,"BreastCancer/contextBreast.txt",10)

    print("Number of columsn:" ,len(newdf.columns))

    #fillna for numerical 
    numerical_cols = newdf.select_dtypes(include='number').columns.tolist()

    for i in range(len(numerical_cols)):
        newdf[numerical_cols[i]] = newdf[numerical_cols[i]].fillna(newdf[numerical_cols[i]].mean())

    #convert categorical columns to numerical and fillna
    categorical_cols = newdf.select_dtypes(include=['object', 'category']).columns.tolist()

    for i in range(len(categorical_cols)):
        newdf[categorical_cols[i]] = newdf[categorical_cols[i]].astype(str)
        newdf[categorical_cols[i]] = newdf[categorical_cols[i]].fillna(newdf[categorical_cols[i]].mode().iloc[0])
        newdf[categorical_cols[i]] = le.fit_transform(newdf[categorical_cols[i]]) #label encoding
        #newdf = pd.concat([newdf.drop(columns=categorical_cols[i]),pd.get_dummies(newdf, columns=[categorical_cols[i]])],axis=1) #one hot encoding

    #machine learning or linear regression
    X_train, X_test, y_train, y_test = train_test_split(newdf, y, test_size=0.2)

    # Fit logistic regression model
    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)

    print("R^2 Score:", r2_score(y_test, y_pred))
    r2s.append(r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    mses.append(mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

data = {
    'r2': r2s,
    'mse': mses
}

outputdf = pd.DataFrame(data)
outputdf.to_csv('output.csv', index=True)