# %%
import numpy as np
import pandas as pd
import random
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from lineartree import LinearBoostRegressor
# %%
df = pd.read_csv( "log10_scaled.csv")
df
# %%
X = df.iloc[:, 1:-1]
Y = df['log_rate']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)
# %%
lbr_est = LinearBoostRegressor(
    Ridge(alpha=1.179),
    random_state=42,
    min_samples_leaf=2,
    min_samples_split=2,
    max_depth=8,
    n_estimators=345
)
lbr_est.fit(X_train.values, Y_train.values)
# %%
y_train_hat = lbr_est.predict(X_train.values)
y_test_hat = lbr_est.predict(X_test.values)
y_full = lbr_est.predict(X.values)
# %%
yhat_train = np.c_[Y_train, y_train_hat]
yhat_test = np.c_[Y_test, y_test_hat]
yhat = np.c_[Y, y_full]
pd.DataFrame(yhat_train, columns=['Y_true', 'Y_predicted']).to_csv(
    'LinearBoost_train.csv', index=False)
pd.DataFrame(yhat_test, columns=['Y_true', 'Y_predicted']).to_csv(
    'LinearBoost_test.csv', index=False)
pd.DataFrame(yhat, columns=['Y_true', 'Y_predicted']).to_csv(
    'LinearBoost_full.csv', index=False)
# %%
for i in range(1, 51):
    random_state = random.randint(1, 1000)
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X, Y, test_size=0.2, random_state=random_state)
    lbr_est = LinearBoostRegressor(
        Ridge(alpha=1.179),
        random_state=42,
        min_samples_leaf=2,
        min_samples_split=2,
        max_depth=8,
        n_estimators=345
    )
    model_tosave = lbr_est.fit(X_train_i, y_train_i)
    filename = f'REBR_{i}.joblib'
    dump(model_tosave, filename)
