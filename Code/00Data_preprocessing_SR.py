# %%
import numpy as np
import pandas as pd
# %%
df = pd.read_excel("Raw_data.xlsx")
# %%
# Drop negtive values
df.drop(['density(cells/mL)'], axis=1, inplace=True)
df.drop(df[(df['rate(mg/(L·d))'] <= 0) |
           (df['tCOD(mg/L)'] <= 0)].index, axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
df
# %%
df_filled = df.fillna(df.median())
df_filled.describe()
# %%
df_filled['log_rate'] = np.log10(df_filled['rate(mg/(L·d))'])
df_filled['log_conc'] = np.log10(df_filled['conc(mg/L)'])
df_filled['log_tCOD'] = np.log10(df_filled['tCOD(mg/L)'])
df_filled.drop(['rate(mg/(L·d))', 'conc(mg/L)',
                'tCOD(mg/L)'], axis=1, inplace=True)
