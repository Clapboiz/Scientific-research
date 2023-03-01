import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(sns.get_dataset_names())
sns.set_theme()
tips_df = sns.load_dataset('tips')
print(tips_df.head())
print(tips_df["total_bill"].value_counts().sort_values(ascending=False))
sns.histplot(data=tips_df["total_bill"]) #histogram
sns.kdeplot(data=tips_df['total_bill']); #kde plot
sns.displot(data=tips_df, x="total_bill", col="time", kde=True); #displot
sns.barplot(data=tips_df, x="sex", y="tip", estimator=np.mean); #barplot
tips_df["sex"].value_counts()
sns.countplot(data=tips_df, x="sex"); #count plot
sns.boxplot(data=tips_df, x="day", y="total_bill", hue="sex", palette='Blues'); #box plot
plt.legend(loc=0);

tips_fg = sns.FacetGrid(data=tips_df, row="smoker", col="time") #Create a class instance of Facet Grid class
tips_fg.map(sns.scatterplot,  'total_bill', 'tip'); #facet grid

kws= dict(s=100, edgecolor='b', alpha=.7)
new_fg = sns.FacetGrid(data=tips_df, col="sex",
                       hue="smoker",
                       col_order=["Female", "Male"],
                       palette='Set2',
                       height=4, aspect=1.4,
                       legend_out=True)

new_fg.map(sns.scatterplot, 'total_bill', 'tip', **kws)
new_fg.add_legend();

#joint plot
penguins_df = sns.load_dataset('penguins')
penguins_df.head()
sns.jointplot(data=penguins_df, x="flipper_length_mm", y="bill_length_mm", hue="species");

#pair plots
sns.pairplot(data=penguins_df, hue="species");

#heat maps
flights_df = sns.load_dataset("flights")
flights_df.head()
flights = pd.pivot_table(flights_df, index='month', columns='year', values='passengers')
print(flights)
sns.heatmap(data=flights, cmap='Blues', linecolor='white', linewidths=1);
plt.show()