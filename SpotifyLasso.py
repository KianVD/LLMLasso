#for spotify dataset
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
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

# Use the OpenAI client library to add your API key.
load_dotenv(dotenv_path="APIKEY.env")
client = OpenAI(
    api_key = os.getenv("APIKEY")
)

#find dataset with 1000 features (genes?)
df = pd.read_csv("Most Streamed Spotify Songs 2024.csv",encoding='ISO-8859-1') 
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

TRIALS = 10
currTrial = 0
r2s = list()
mses = list()
while currTrial < TRIALS:
    currTrial += 1
    
    #no llm interference
    newdf = df

    print("Number of columsn:" ,len(newdf.columns))

    #split data
    X_train, X_test, y_train, y_test = train_test_split(newdf, y, test_size=0.2)

    #standardize test and train sep
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Lasso with cross-validated penalization (lambda)
    lasso = linear_model.Lasso(alpha=0.1)
    lasso.fit(X_train_std, y_train)

    # Predict and evaluate (@ is matrix multiplication) #headers? array types?
    y_pred = lasso.predict(X_test_std)

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