from sklearn.feature_selection import f_classif,SelectPercentile
from src.data.load_data import load_dataset
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os

fig_loc= os.path.abspath('../../outputs/figures')

pd.set_option('display.float_format', '{:,.10f}'.format)

mpl.rcParams['figure.dpi'] = 400

df = load_dataset()

# print(df.columns)

# print(df.dtypes)

# REMOVE the age, id and pay_2:6 variables
keep_vars = ['limit_bal', 'education', 'marriage', 'age',
             'pay_1','bill_amt1', 'bill_amt2','bill_amt3',
             'bill_amt4', 'bill_amt5', 'bill_amt6', 'pay_amt1',
             'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5',
             'pay_amt6','default payment next month']

# keep variables above i.e. remove id, sex and pay_2:6
df = df[keep_vars]
df.dropna(inplace=True)
corr_mat =  df.corr()


plt.figure()
sns.heatmap(corr_mat,cbar=True, center=0,annot=True,fmt="0.2f",annot_kws={"size":"xx-small"})
plt.title('Pearson Correlation Plot')
plt.savefig(os.path.join(fig_loc,'features_Heatmap.png'))

X = df.iloc[:,:-1].values

y = df.iloc[:,-1].values
# print(X.shape,y.shape)


f_statistics, p_values = f_classif(X,y)

f_stats_df =  pd.DataFrame({'f_stat':f_statistics,'p_values':p_values},index=df.columns[:-1])
# f_stats_df['inverse_p_values'] = 1- f_stats_df['p_values']
plt.figure()
(f_stats_df.sort_values('f_stat',ascending=False))['f_stat'].plot(kind='barh')
plt.savefig(os.path.join(fig_loc,'f_statistics_bar_plot.png'))


selector =  SelectPercentile(f_classif,percentile=20)
selector.fit(X,y)

print(f' (df.iloc[:,:-1].columns[selector.get_support()])')

print(X[:,selector.get_support()].shape)
#print(selector.__dict__)
