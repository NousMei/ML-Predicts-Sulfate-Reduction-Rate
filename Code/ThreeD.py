# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
# %%
# round 1 #


def func_inhib1(X, V, K1, K2, K_in):
    return V*X.iloc[:, 0]/(K1+X.iloc[:, 0])*K_in/(X.iloc[:, 0]+K_in)*X.iloc[:, 1]/(K2+X.iloc[:, 1])


# round 2/3 #
# def func_inhib2or3(X, K1, K_in, K2):
#     # mlogK1+n
#     m='# coefficient value of logK1'
#     n='# constant'
#     return np.power(10, m*np.log10(K1)+n)*X.iloc[:, 0]/(K1+X.iloc[:, 0])*K_in/(X.iloc[:, 0]+K_in)*X.iloc[:, 1]/(K2+X.iloc[:, 1])
# %%
df_rate = pd.read_csv('# GeneratedData_predrate_***.csv')
df_rel = pd.read_csv("MonodParams_ANN_round1.csv")[['pH', 'T']].copy()
# %%
para_1st = ['V', 'K1', 'K2', 'K_in']
para_2nd = ['K1', 'Kin']
para_3rd = ['Kin']
# %%
fig = plt.figure(figsize=(30, 20))
for j in range(1, 301):
    X, Y = df_rate[['conc', 'COD']], df_rate.iloc[:, j+1]
    pH, T = df_rel.iat[j-1, 0], df_rel.iat[j-1, 1]
    # Choose which round you want to plot
    # ## round == 1 ##
    popt, pcov = curve_fit(func_inhib1, X, Y, bounds=(
        [0]*len(para_1st), [np.inf]*len(para_1st)), maxfev=50000)
    df_rel.loc[j-1, para_1st] = popt
    # ################
    # ## round == 2 ##
    '''a1: coefficient of pH after round1
    b1: coefficient of T after round1
    c1: const after round1 '''
    # K2_fixed = np.power(10, a1*pH+b1*T+c1) # make sure you replace 'a1','b1','c1' with values, the same inround 3
    # popt, pcov = curve_fit(lambda X, K1, K_in: func_inhib2or3(X, K1, K_in, K2_fixed), X, Y, bounds=(
    #     [0]*len(param_2nd), [np.inf]*len(param_2nd)), maxfev=50000)
    # df_rel.loc[j-1, param_2nd] = popt
    # ################
    # ## round == 3 ##
    '''a3: coefficient of pH after round3
    b3: coefficient of T after round3
    c3: const after round3'''
    # K1_fixed = np.power(10, a3*pH+b3*T+c3)
    # K2_fixed = np.power(10, a1*pH+b1*T+c1)
    # popt, pcov = curve_fit(lambda X, K_in: func_inhib2or3(X, K_in, K1_fixed, K2_fixed), X, Y, bounds=(
    #     [0]*len(param_3rd), [np.inf]*len(param_3rd)), maxfev=50000)
    # df_rel.loc[j-1, param_3rd] = popt
    # ################
    if j <= 20:
        cmap = '# select the cmap you like'
        ax = fig.add_subplot(4, 5, j, projection='3d')

        ax.ticklabel_format(axis='both', style='sci',
                            scilimits=(-1, 2), useMathText=True)
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.scatter3D(df_rate['conc'], df_rate['COD'], df_rate.iloc[:, j+1])
        ax.plot_trisurf(df_rate['conc'].values,
                        df_rate['COD'].values,
                        # round 1 #
                        func_inhib1(df_rate[['conc', 'COD']], *popt),
                        # round 2 #
                        # func_inhib2or3(df_rate[['conc', 'COD']], *popt, K2_fixed),
                        # round 3 #
                        # func_inhib2or3(df_rate[['conc', 'COD']], *popt, K1_fixed, K2_fixed),
                        cmap=camp, alpha=0.8)
        # round 1 #
        R2 = r2_score(df_rate.iloc[:, j+1],
                      func_inhib1(df_rate[['conc', 'COD']], *popt))
        # round 2 #
        # R2 = r2_score(df_rate.iloc[:, j+1], func_inhib2or3(df_rate[['conc', 'COD']], *popt, K2_fixed))
        # round 3 #
        # R2 = r2_score(df_rate.iloc[:, j+1], func_inhib2or3(df_rate[['conc', 'COD']], *popt, K1_fixed, K2_fixed))
        ax.set_xlabel(xlabel='$S_{Sulfate}$ \n(mg/L)')
        ax.set_ylabel(ylabel='COD\n(mg/L)')
        ax.set_zlabel(zlabel='Rate*\n(mg/Ld)')
        ax.text2D(x=0.1, y=0.8, s=f'$R^2$ = {R2:.3f}', transform=ax.transAxes)
plt.savefig('your figure name.png', bbox_inches='tight', dpi=1000)
plt.show()
df_rel.to_csv('your file name.csv', index=False)
# %%
