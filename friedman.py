import pandas as pd
from scipy import stats
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Wczytaj CSV
df = pd.read_csv('data/ChickWeight.csv')

# 2. Sprawdź nagłówki
print("Columns:", df.columns.tolist())
# → ['id', 'weight', 'Time', 'Chick', 'Diet']

# 3. Pivotuj, używając 'weight' (małej litery)
wide = df.pivot(index='Chick', columns='Time', values='weight').dropna()

# 4. Test Friedmana
stat, p = stats.friedmanchisquare(
    *[wide[day] for day in wide.columns]
)
print(f"Friedman χ² = {stat:.3f}, p = {p:.3f}")

# 5. Post-hoc Nemenyi
actions = wide.columns.tolist()
if p < 0.05:
    pvals = sp.posthoc_nemenyi_friedman(wide.values)
    pvals_df = pd.DataFrame(pvals, index=actions, columns=actions)
    print("Nemenyi p-values:\n", pvals_df)

    # 6. Heatmap tylko dla dni ≤ 10
days_upto_10 = [day for day in actions if day <= 10]
if p < 0.05 and days_upto_10:
    pvals_sub = pvals_df.loc[days_upto_10, days_upto_10]
    plt.figure(figsize=(6,5))
    sns.heatmap(pvals_sub, annot=True, fmt=".3f", square=True,
                cbar_kws={'label':'p-value'})
    plt.title('Nemenyi post-hoc p-values (Days ≤ 10)')
    plt.xlabel('Day')
    plt.ylabel('Day')
    plt.tight_layout()
    plt.show()
else:
    print("Brak istotnych różnic (p >= 0.05) lub brak dni ≤ 10, pomijam post-hoc heatmap.")
