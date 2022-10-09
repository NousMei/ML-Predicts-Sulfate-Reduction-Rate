# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df = pd.read_excel("log10_scaled.csv")
# %%
fig, ax = plt.subplots(figsize=(3, 2), dpi=500)
sns.stripplot(data=df, x='scale', y='rate(mg/(L·d))',
              marker='d', alpha=0.5, size=3)
sns.pointplot(x='scale', y='rate(mg/(L·d))', data=df,
              join=True, markers='d')
plt.yscale('log')
sns.despine()
plt.grid(axis='y', c='k', ls=':', lw=0.5)
plt.ylabel('rate (mg L$^{-1}$ d$^{-1}$)')
plt.savefig('figure name.png', bbox_inches='tight', dpi=800)
# %%
# heatmap
fig, ax = plt.subplots(figsize=(3, 2))
heatmap = sns.heatmap(df.corr(method='spearman'), cmap='# your cmap color', center=0, annot=True, fmt='.3g',
                      linewidth=0.5, annot_kws={'size': 6})
fig = heatmap.get_figure()
fig.savefig('your figure name.png', dpi=800, bbox_inches='tight')
