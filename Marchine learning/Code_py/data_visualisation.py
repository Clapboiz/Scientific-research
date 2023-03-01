import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

adv_df = pd.read_csv("ad.csv")
print(adv_df.head())
#drop cot stt khong co label
adv_df_new = adv_df.drop(adv_df.columns[0], axis=1)
print(adv_df_new.head())

adv_corr = adv_df_new.corr() #Get correlation data
print(adv_corr)
#data cang gan +- 1 thi cang manh va cang gan 0 thi cang yeu
#sau khi print thi ta thay rang 2 nua doi xung voi nhau qua duong cheo chinh
ones_corr = np.ones_like(adv_corr, dtype=bool)
print(ones_corr)

# np's triu: return only upper triangle matrix
mask = np.triu(ones_corr)
print(mask)
sns.heatmap(data=adv_corr, mask=mask); #lay phia tren (phan true)

#remove hang va cot ngoai cung phia tren
adjusted_mask = mask[1:, :-1] #dieu chinh mask
print(adjusted_mask)



adjusted_adv_corr = adv_corr.iloc[1:, :-1]
fig, ax = plt.subplots(figsize=(10,8))

#That method uses HUSL colors, so you need hue, saturation, and lightness.
#I used hsluv.org to select the colors of this chart.
cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

sns.heatmap(data=adjusted_adv_corr, mask=adjusted_mask,
            annot=True, annot_kws={"fontsize":13}, fmt=".2f", cmap=cmap,
            vmin=-1, vmax=1,
            linecolor='white', linewidths=0.5);

yticks = [i.upper() for i in adjusted_adv_corr.index]
xticks = [i.upper() for i in adjusted_adv_corr.columns]

ax.set_yticklabels(yticks, rotation=0, fontsize=13);
ax.set_xticklabels(xticks, rotation=90, fontsize=13);
title = 'matrix tuong quan\n'
ax.set_title(title, loc='center', fontsize=18);

plt.show()


