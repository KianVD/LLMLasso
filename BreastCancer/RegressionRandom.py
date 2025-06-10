#normal lasso to compare
#use open ai to narrow down 1000 features to 100, then narrow down those features with best subset selection
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
import random
from sklearn.linear_model import Lasso

#find dataset with 1000 features (genes?)
df = pd.read_csv("METABRIC_RNA_Mutation.csv")
y = df["overall_survival_months"]

#feed names of these features to ai and ask it to narrow down to top 100 most important (format so that code can read it?)
df = df.drop(columns=["overall_survival_months","overall_survival","death_from_cancer"])
headers = df.columns.tolist()

#run model for results

#isolate new headers from llm
#newHeaders = ["mutation_count","age_at_diagnosis","cancer_type_detailed","chemotherapy","hormone_therapy","lymph_nodes_examined_positive","pr_status","tumor_size","tumor_stage"]
#newHeaders = headers
TRIALS = 10
currTrial = 0
r2s = list()
mses = list()
while currTrial < TRIALS:
    currTrial += 1
    #randomly select 100 columns
    newHeaders = random.sample(headers,10)

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