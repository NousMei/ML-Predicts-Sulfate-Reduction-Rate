# %%
import numpy as np
import pandas as pd
import random
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
# %%
df = pd.read_csv("log10_scaled.csv")
df.describe()
# %%
X = df.iloc[:, 1:-1]
Y = df['log_rate']
# %%
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)
# %%
pipelineRidge = Pipeline([
    ('poly', PolynomialFeatures(degree=4)),
    ('rs', RobustScaler()),
    ('rig', Ridge(alpha=0.268))
])
pipelineRidge.fit(X_train, Y_train)
# %%
y_train_hat = pipelineRidge.predict(X_train)
y_test_hat = pipelineRidge.predict(X_test)
y_full_hat = pipelineRidge.predict(X)
# %%
yhat_train = np.c_[Y_train, y_train_hat]
yhat_test = np.c_[Y_test, y_test_hat]
yhat = np.c_[Y, y_full_hat]
pd.DataFrame(yhat_train, columns=['Y_true', 'Y_predicted']).to_csv(
    'Ridge_train.csv', index=False)
pd.DataFrame(yhat_test, columns=['Y_true', 'Y_predicted']).to_csv(
    'Ridge_test.csv', index=False)
pd.DataFrame(yhat, columns=['Y_true', 'Y_predicted']
             ).to_csv('Ridge_full.csv', index=False)
# %%
for i in range(1, 51):
    random_state = random.randint(1, 1000)
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X, Y, test_size=0.2, random_state=random_state)
    pipelineRidge = Pipeline([
        ('poly', PolynomialFeatures(degree=4)),
        ('rs', RobustScaler()),
        ('rig', Ridge(alpha=0.268))])
    model = pipelineRidge.fit(X_train_i, y_train_i)
    filename = f'Ridge_{i}.joblib'
    dump(model, filename)
