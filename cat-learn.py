#!/bin/python3

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore') 

df = pd.read_csv("data.csv")

df = df.drop(np.where(df["price"] >= 60000000)[0])
df = df.drop(np.where(df["year"] <= 1975)[0])
df = df.drop(df[(df.price >= 10000000) & (df.year <= 2000) & (df.box == "механика")].index)

df.loc[ df["box"] == "механика", "box"] = 0
df.loc[ df["box"] == "вариатор", "box"] = 1
df.loc[ df["box"] == "робот", "box"] = 2
df.loc[ df["box"] == "автомат", "box"] = 3

df["box"] = df["box"].astype(int)

df['model'] = OrdinalEncoder().fit_transform(df[['model']])
df['generation'] = OrdinalEncoder().fit_transform(df[['generation']])
df['body'] = OrdinalEncoder().fit_transform(df[['body']])
df['color'] = OrdinalEncoder().fit_transform(df[['color']])
df['fuel'] = OrdinalEncoder().fit_transform(df[['fuel']])
df['wheel'] = OrdinalEncoder().fit_transform(df[['wheel']])
df['city'] = OrdinalEncoder().fit_transform(df[['city']])
df['brand'] = OrdinalEncoder().fit_transform(df[['brand']])

X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)

tic = time.perf_counter()
cat = CatBoostRegressor(iterations=15000, depth=15, task_type="GPU")

cat.fit(X_train, y_train)
y_pred = cat.predict(X_test)
    
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = mse ** 0.5

toc = time.perf_counter()
print(f"TIME {toc-tic:0.4f}")
print(f'R2 {r2} | MAE {mae}')
print(f'RMSE {rmse} | MSE {mse}')
