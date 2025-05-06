import pandas as pd
from scipy import stats

hf = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
alc = pd.read_csv('data/student-por.csv')

print("--- Non-parametric Tests ---")
# 4.1 HF: Mann-Whitney U for serum_creatinine smokers vs non-smokers
cre_s = hf[hf['smoking']==1]['serum_creatinine']
cre_ns = hf[hf['smoking']==0]['serum_creatinine']
u, p_u = stats.mannwhitneyu(cre_s, cre_ns)
print(f"Mann-Whitney serum_creatinine smokers vs non-smokers: U = {u}, p = {p_u:.3f}")

# 4.2 Alcohol: Kruskal-Wallis for G3 across studytime (1-4)
groups = [alc[alc['studytime']==i]['G3'] for i in sorted(alc['studytime'].unique())]
h, p_h = stats.kruskal(*groups)
print(f"Kruskal-Wallis G3 by studytime: H = {h:.3f}, p = {p_h:.3f}\n")