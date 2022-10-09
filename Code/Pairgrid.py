# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df_rel = pd.read_csv("MonodParams_****_round*.csv")
df_log = df_rel[['pH', 'T']].copy()
df_log[['logV', 'logK1', 'logK2', 'logKin']] = np.log10(
    df_rel[['V', 'K1', 'K2', 'K_in']])

g = sns.PairGrid(data=df_log)
g.figure.suptitle(t='# your title', x=0.5, y=1.01)
g.map_diag(plt.hist, color="0.5", edgecolor='0.5', alpha=0.7)
g.map_offdiag(plt.scatter, color='m')
g.savefig('your figure name.png', bbox_inches='tight', dpi=800)
