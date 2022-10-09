# %%
import numpy as np
import pandas as pd
import random
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
# %%
df = pd.read_csv("log10_scaled.csv")
df
# %%
X = df.iloc[:, 1:-1]
Y = df['log_rate']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)
# %%
pipelineBaye = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('rs', RobustScaler()),
    ('rig', BayesianRidge(alpha_1=0.001, alpha_2=1e-07,
                          lambda_1=1e-07, lambda_2=1e-05, tol=0.01))
])
pipelineBaye.fit(X_train, Y_train)
# %%
y_train_hat = pipelineBaye.predict(X_train)
y_test_hat = pipelineBaye.predict(X_test)
y_full_hat = pipelineBaye.predict(X)
# %%
yhat_train = np.c_[Y_train, y_train_hat]
yhat_test = np.c_[Y_test, y_test_hat]
yhat = np.c_[Y, y_full_hat]
pd.DataFrame(yhat_train, columns=['Y_true', 'Y_predicted']).to_csv(
    'Baye_train.csv', index=False)
pd.DataFrame(yhat_test, columns=['Y_true', 'Y_predicted']).to_csv(
    'Baye_test.csv', index=False)
pd.DataFrame(yhat, columns=['Y_true', 'Y_predicted']
             ).to_csv('Baye_full.csv', index=False)
# %%
for i in range(1, 51):
    random_state = random.randint(1, 1000)
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X, Y, test_size=0.2, random_state=random_state)
    pipelineBaye = Pipeline([
        ('poly', PolynomialFeatures(degree=3)),
        ('rs', RobustScaler()),
        ('rig', BayesianRidge(alpha_1=0.001, alpha_2=1e-07,
                              lambda_1=1e-07, lambda_2=1e-05, tol=0.01))
    ])
    model = pipelineBaye.fit(X_train_i, y_train_i)
    filename = f'Baye_{i}.joblib'
    dump(model, filename)
