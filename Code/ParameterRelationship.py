# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
df_rel = pd.read_csv("MonodParams_***_round*.csv")
df_log = df_rel[['pH', 'T']].copy()
df_log[['logV', 'logK1', 'logK2', 'logKin']] = np.log10(
    df_rel[['V', 'K1', 'K2', 'K_in']])
# %%
X = df_log[['pH', 'T']]  # df_log['logK1']
Y = df_log['logK2']  # df_log['logV'], df_log['logK1']

lire_model = LinearRegression()
lire_model.fit(X, Y)
Y_predict = lire_model.predict(X)

sign = ['+' if lire_model.intercept_ > 0 else '']
str = r'$R^2$' + \
    f' = {lire_model.score(X, Y):.3f} MAE = {mean_absolute_error(Y, Y_predict):.3f}\n' +\
    f'logK$_2$ = {lire_model.coef_[0]:.3f}pH' +\
    f'{lire_model.coef_[1]:.3f}T' + sign[0] +\
    f'+{lire_model.intercept_:.3f}'

fig, ax = plt.subplots(figsize=(3, 2))
ax.scatter(x=Y, y=Y_predict, c='grey')
ax.plot([plt.xlim()[0], plt.xlim()[1]], [plt.xlim()[0],
                                         plt.xlim()[1]], '--', transform=ax.transData)
ax.set(xlabel='# your x label', ylabel='# your y label',
       title='# your title name')
ax.text(0.1, 0.8, str, wrap=True, transform=ax.transAxes)
plt.savefig('your figure name.png', bbox_inches='tight', dpi=500)
plt.show()
