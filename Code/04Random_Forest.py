# %%
import pandas as pd
import random
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# %%
df = pd.read_csv("log10_scaled.csv")
df
# %%
X = df.iloc[:, 1:-1]
Y = df['log_rate']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)
# %%!
rfr_est = RandomForestRegressor(random_state=42, n_estimators=245,oob_score=True)
rfr_est.fit(X_train, Y_train)
# %%
y_train_hat = rfr_est.predict(X_train)
y_test_hat = rfr_est.predict(X_test)
y_full = rfr_est.predict(X)
# %%
import numpy as np
yhat_train = np.c_[Y_train,y_train_hat]
yhat_test = np.c_[Y_test,y_test_hat]
yhat = np.c_[Y, y_full]
pd.DataFrame(yhat_train,columns=['Y_true','Y_predicted']).to_csv('RandomF_train.csv',index=False)
pd.DataFrame(yhat_test,columns=['Y_true','Y_predicted']).to_csv('RandomF_test.csv',index=False)
pd.DataFrame(yhat,columns=['Y_true','Y_predicted']).to_csv('RandomF_full.csv',index=False)
# %%
for i in range(1, 51): 
    random_state = random.randint(1, 1000)
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X, Y, test_size=0.2, random_state=random_state)
    rfr_est = RandomForestRegressor(random_state=42, n_estimators=245,oob_score=True)
    model_tosave =rfr_est.fit(X_train_i, y_train_i)
    filename = f'RF_{i}.joblib'
    dump(model_tosave, filename)