# %%
# -*- coding : utf-8-*-
# coding:unicode_escape
from keras import backend as k
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
import keras_tuner as kt
import tensorflow
from numpy.random import seed
seed(42)
tensorflow.random.set_seed(42)
# %%
df = pd.read_csv("log10_scaled.csv")
df
# %%
X = df.iloc[:, 1:-1]
Y = df['log_rate']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)
# %%
rs = RobustScaler()
X_train_scale = rs.fit_transform(X_train)
X_test_scale = rs.transform(X_test)
X_scale = rs.fit_transform(X)
# %%


def build_model(hp):

    model = Sequential()
    model.add(Dense(units=hp.Int("units", min_value=4, max_value=20, step=2),
                    input_dim=4,
                    activation=hp.Choice("activation", ["relu", "elu", "selu"])))
    for i in range(hp.Int("num_layers", 1, 11)):
        model.add(
            Dense(
                units=hp.Int(f"units_{i}", min_value=4, max_value=20, step=2),
                activation=hp.Choice("activation", ["relu", "elu", "selu"]),
                kernel_constraint=max_norm(3.)
            )
        )
    if hp.Boolean("dropout"):
        model.add(Dropout(rate=0.25))
    model.add(Dense(1,))
    learning_rate = hp.Float("lr", min_value=1e-4,
                             max_value=1e-2, sampling="log")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mse", "mae"],
    )
    return model


build_model(kt.HyperParameters())
# %%
tuner = kt.Hyperband(
    hypermodel=build_model,
    objective='val_mse',
    max_epochs=200,
    directory="..\directory_keras"
)
tuner.search_space_summary()
# %%
callbacks = [tensorflow.keras.callbacks.EarlyStopping(
    monitor='mse', mode='min', patience=5, verbose=1)]
tuner.search(X_train_scale, Y_train, epochs=100, validation_split=0.2,
             callbacks=callbacks, verbose=2)
# %%
best_hps = tuner.get_best_hyperparameters(1)[0]
besthp_model = tuner.hypermodel.build(best_hps)
besthp_model.fit(X_train_scale, Y_train,
                 epochs=200, validation_split=0.2, batch_size=20)
# %%
besthp_model.evaluate(X_test_scale, Y_test, return_dict=True)
# %%
y_train_hat = besthp_model.predict(X_train_scale)
y_test_hat = besthp_model.predict(X_test_scale)
y_full = besthp_model.predict(X_scale)
# %%
yhat_train = np.c_[Y_train, y_train_hat]
yhat_test = np.c_[Y_test, y_test_hat]
yhat = np.c_[Y, y_full]
pd.DataFrame(yhat_train, columns=['Y_true', 'Y_predicted']).to_csv(
    'ANN_train.csv', index=False)
pd.DataFrame(yhat_test, columns=['Y_true', 'Y_predicted']).to_csv(
    'ANN_test.csv', index=False)
pd.DataFrame(yhat, columns=['Y_true', 'Y_predicted']
             ).to_csv('ANN_full.csv', index=False)
# %%
for i in range(1, 51):
    random_state = random.randint(1, 1000)
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X, Y, test_size=0.2, random_state=random_state)
    rs = RobustScaler()
    X_train_scale = rs.fit_transform(X_train_i)
    X_test_scale = rs.transform(X_test_i)
    model_tosave = tuner.hypermodel.build(best_hps)
    model_tosave.fit(X_train_scale, y_train_i, epochs=200,
                     validation_split=0.2, batch_size=20)
    filename = f'ANN_{i}'
    model_tosave.save(filename)
    k.clear_session()
