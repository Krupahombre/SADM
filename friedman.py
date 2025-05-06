# import pandas as pd
# import numpy as np
# from scipy import stats
# from scipy.stats import studentized_range
# import scikit_posthocs as sp
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # 1. Wczytaj dane
# df = pd.read_csv('data/ChickWeight.csv')
#
# # 2. Odfiltruj dni ≤ 10
# df10 = df[df['Time'] <= 10].copy()
#
# # 3. Pivot na szeroki format i dropna
# wide10 = df10.pivot(index='Chick', columns='Time', values='weight').dropna()
#
# # 4. Test Friedmana
# stat, p = stats.friedmanchisquare(*[wide10[day] for day in wide10.columns])
# print(f"Friedman χ² (dni ≤ 10) = {stat:.3f}, p = {p:.3f}")
#
# # 5. Post-hoc Nemenyi i heatmap p-wartości (bez maski)
# if p < 0.05:
#     pvals_df = sp.posthoc_nemenyi_friedman(wide10)
#     print("Nemenyi p-values (dni ≤ 10):\n", pvals_df)
#
#     plt.figure(figsize=(6,5))
#     sns.heatmap(pvals_df, annot=True, fmt=".3f", square=True,
#                 cbar_kws={'label':'p-value'})
#     plt.title('Nemenyi post-hoc p-values (dni ≤ 10)')
#     plt.xlabel('Dzień')
#     plt.ylabel('Dzień')
#     plt.tight_layout()
#     plt.show()
# else:
#     print("Brak istotnych różnic (p ≥ 0.05), pomijam heatmapę post-hoc.")
#
# # 6. Critical-Difference diagram
#
# # 6.1. Rangi i średnie rangi
# ranks     = wide10.rank(axis=1)
# avg_ranks = ranks.mean().sort_values()
# names     = avg_ranks.index.astype(str)
# values    = avg_ranks.values
#
# # 6.2. Oblicz CD
# k      = len(values)
# n      = wide10.shape[0]
# alpha  = 0.05
# q_crit = studentized_range.ppf(1 - alpha, k, np.inf)
# CD     = q_crit * np.sqrt(k * (k + 1) / (6 * n))
#
# # 6.3. Narysuj CD-diagram
# plt.figure(figsize=(8,2))
# plt.hlines(1, values.min() - .1, values.max() + .1, color='gray')
# plt.plot(values, np.ones(k), 'o', color='black')
# for x,label in zip(values, names):
#     plt.text(x, 1.02, label, ha='center', va='bottom', rotation=45)
# x0 = values.min()
# plt.plot([x0, x0 + CD], [0.9, 0.9], lw=2, color='black')
# plt.text(x0 + CD/2, 0.88, f'CD = {CD:.2f}', ha='center', va='top')
# plt.yticks([])
# plt.xlabel('Średnia ranga')
# plt.title('CD Diagram (dni ≤ 10)')
# plt.tight_layout()
# plt.show()



import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import studentized_range
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Wczytaj dane
df = pd.read_csv('data/ChickWeight.csv')

# 2. Pivot na szeroki format (wszystkie dni)
wide = df.pivot(index='Chick', columns='Time', values='weight').dropna()

# 3. Test Friedmana na pełnym zakresie dni
times = wide.columns.tolist()
stat, p = stats.friedmanchisquare(*[wide[t] for t in times])
print(f"Friedman χ² (pełny zakres dni) = {stat:.3f}, p = {p:.3f}")

# 4. Post-hoc Nemenyi i pełna heatmap p-wartości
if p < 0.05:
    pvals_df = sp.posthoc_nemenyi_friedman(wide)
    print("Nemenyi p-values (pełny zakres dni):\n", pvals_df)

    plt.figure(figsize=(8,6))
    sns.heatmap(pvals_df, annot=True, fmt=".3f", square=True,
                cbar_kws={'label':'p-value'},
                linewidths=0.5, linecolor='white')
    plt.title('Nemenyi post-hoc p-values (wszystkie dni)')
    plt.xlabel('Dzień')
    plt.ylabel('Dzień')
    plt.tight_layout()
    plt.show()
else:
    print("Brak istotnych różnic (p ≥ 0.05), pomijam heatmapę post-hoc.")

ranks     = wide.rank(axis=1)
avg_ranks = ranks.mean().sort_values()
names     = avg_ranks.index.astype(str)
values    = avg_ranks.values

k      = len(values)
n      = wide.shape[0]
alpha  = 0.05
q_crit = studentized_range.ppf(1 - alpha, k, np.inf)
CD     = q_crit * np.sqrt(k*(k+1)/(6*n))

# 5. Rysowanie
fig, ax = plt.subplots(figsize=(12,2.5))
ax.hlines(1, values.min()-0.3, values.max()+0.3, color='gray', lw=1)
ax.scatter(values, np.ones_like(values), s=50, color='black', zorder=5)

for x, lab in zip(values, names):
    ax.text(x, 1.02, lab, ha='center', va='bottom', rotation=45, fontsize=10)

start = values.min()
end   = start + CD
ax.plot([start, end], [0.9, 0.9], lw=3, color='black', solid_capstyle='butt')
ax.text((start+end)/2, 0.87, f'CD = {CD:.2f}', ha='center', va='top', fontsize=10)

ax.set_ylim(0.85, 1.1)
ax.set_yticks([])
ax.set_xlabel('Średnia ranga', fontsize=12)
ax.set_title('CD Diagram (pełny zakres dni)', fontsize=14)

sns.despine(ax=ax, top=True, right=True, left=True)

plt.tight_layout()
plt.show()
