# %%
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%

# %%
df_train = pd.read_csv("*****_train.csv")
df_test = pd.read_csv("*****_test.csv")
df_full = pd.read_csv("*****_full.csv")
# %%
Y_train = df_train['Y_true']
Y_train_hat = df_train['Y_predicted']
Y_test = df_test['Y_true']
Y_test_hat = df_test['Y_predicted']
Y_full = df_full['Y_true']
Y_full_hat = df_full['Y_predicted']
# %%
r2_train = r2_score(Y_train, Y_train_hat)
r2_test = r2_score(Y_test, Y_test_hat)
r2_full = r2_score(Y_full, Y_full_hat)
mae_train = mean_absolute_error(Y_train, Y_train_hat)
mae_test = mean_absolute_error(Y_test, Y_test_hat)
mae_full = mean_absolute_error(Y_full, Y_full_hat)
# %%
fig, ax = plt.subplots(figsize=(3, 2), dpi=800)
plt.scatter(x=Y_train, y=Y_train_hat, c='# your color',
            s=3, label='training set')
plt.scatter(x=Y_test, y=Y_test_hat, c='# your color', s=3, label='testing set')
# plt.scatter(x=Y_full, y=Y_full_hat, c='# your color',label='Full set')
plt.plot([0, plt.xlim()[1]], [0, plt.xlim()[1]], '--', color='gray',
         transform=ax.transData)
plt.xlabel('Observed rate (mg L$^{-1}$ d$^{-1}$)')
plt.ylabel('Predicted rate (mg L$^{-1}$ d$^{-1}$)')
str = 'Train: ' + r'$R^2$' + \
    f' = {r2_train:.3f} MAE = {mae_train:.3f}\n' + 'Test: ' + \
    r'$R^2$' + f' = {r2_test:.3f} MAE = {mae_test:.3f}'
# str = r'$R^2$' + f' = {r2_full:.3f} MAE = {mae_full:.3f}'
plt.text(2, 0, str, wrap=True)
plt.legend(loc='upper left')
plt.savefig('title name.png', bbox_inches='tight', dpi=800)
plt.show()
# %%
