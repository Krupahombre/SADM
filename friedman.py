import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import studentized_range
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Wczytaj dane
df = pd.read_csv('data/ChickWeight.csv')

# 2. Boxplot mas wg dnia
plt.figure(figsize=(10,5))
sns.boxplot(x='Time', y='weight', data=df)
plt.title('Rozkład masy piskląt w kolejnych dniach')
plt.xlabel('Dzień')
plt.ylabel('Masa')
plt.tight_layout()
plt.show()

# 3. Pivot na szeroki format (wszystkie dni)
wide = df.pivot(index='Chick', columns='Time', values='weight').dropna()

# 4. Test Friedmana
times = wide.columns.tolist()
stat, p = stats.friedmanchisquare(*[wide[t] for t in times])
print(f"Friedman χ² = {stat:.3f}, p = {p:.3f}")

# 5. Post-hoc Nemenyi i heatmapa
if p < 0.05:
    pvals = sp.posthoc_nemenyi_friedman(wide)
    plt.figure(figsize=(8,6))
    sns.heatmap(pvals, annot=True, fmt=".3f", square=True,
                cbar_kws={'label':'p-value'},
                linewidths=0.5, linecolor='white')
    plt.title('Nemenyi post-hoc (wszystkie dni)')
    plt.xlabel('Dzień')
    plt.ylabel('Dzień')
    plt.tight_layout()
    plt.show()

# 6. Średnie rangi i CD
ranks     = wide.rank(axis=1)
avg_ranks = ranks.mean().sort_values()
names     = avg_ranks.index.astype(str).tolist()
values    = avg_ranks.values

k      = len(values)
n      = wide.shape[0]
alpha  = 0.05
q_crit = studentized_range.ppf(1 - alpha, k, np.inf)
CD     = q_crit * np.sqrt( k*(k+1)/(6*n) )

# 7. Wyznacz grupy dni nieróżniące się (różnica rang ≤ CD)
groups = []
i = 0
while i < k:
    start_val = values[i]
    grp = [i]
    j = i + 1
    while j < k and values[j] - start_val <= CD:
        grp.append(j)
        j += 1
    groups.append(grp)
    i = j

# 8. Rysowanie CD-diagramu z grupującymi liniami
fig, ax = plt.subplots(figsize=(12,4))
# pozioma linia bazowa
ax.hlines(1, values.min()-0.3, values.max()+0.3, color='gray', lw=1)
# punkty dni
ax.scatter(values, np.ones_like(values), s=60, color='black', zorder=5)
# etykiety dni
for x, lab in zip(values, names):
    ax.text(x, 1.02, lab, ha='center', va='bottom', rotation=45)

# pasek CD
start = values.min()
end   = start + CD
ax.plot([start, end], [0.9, 0.9], lw=3, color='black', solid_capstyle='butt')
ax.text((start+end)/2, 0.87, f'CD = {CD:.2f}', ha='center', va='top')

# linie grupujące
y = 1.08
for grp in groups:
    x0 = values[grp[0]]
    x1 = values[grp[-1]]
    ax.plot([x0, x1], [y, y], lw=2, color='black')
    y += 0.05

# formatowanie osi
ax.set_ylim(0.85, y+0.02)
ax.set_yticks([])
ax.set_xlabel('Średnia ranga')
ax.set_title('CD-diagram z grupami (pełny zakres dni)')
sns.despine(ax=ax, top=True, right=True, left=True)
plt.tight_layout()
plt.show()
