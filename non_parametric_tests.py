import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Wczytanie danych
hf = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
alc = pd.read_csv('data/student-por.csv')

# 4.1 HF: Mann-Whitney U for ejection_fraction by DEATH_EVENT (0 = survived, 1 = died)
ef_dead   = hf[hf['DEATH_EVENT'] == 1]['ejection_fraction']
ef_alive  = hf[hf['DEATH_EVENT'] == 0]['ejection_fraction']
u, p_u = stats.mannwhitneyu(ef_dead, ef_alive, alternative='two-sided')
print(f"Mann-Whitney ejection_fraction (died vs survived): U = {u:.1f}, p = {p_u:.3f}")

# --- boxplot ejection_fraction by DEATH_EVENT ---
plt.figure()
hf.boxplot(column='ejection_fraction', by='DEATH_EVENT')
plt.title('Ejection fraction by survival status')
plt.suptitle('')
plt.xlabel('DEATH_EVENT (0 = survived, 1 = died)')
plt.ylabel('Ejection fraction (%)')
plt.tight_layout()
plt.show()


# 4.2 Alcohol: Kruskal-Wallis for G3 across studytime (1-4)
groups = [alc[alc['studytime']==i]['G3'] for i in sorted(alc['studytime'].unique())]
h, p_h = stats.kruskal(*groups)
print(f"Kruskal-Wallis G3 by studytime: H = {h:.3f}, p = {p_h:.3f}\n")

# --- boxplot G3 by studytime ---
plt.figure()
alc.boxplot(column='G3', by='studytime')
plt.title('Final grade (G3) by study time')
plt.suptitle('')
plt.xlabel('Study time category (1–4)')
plt.ylabel('G3 (final grade)')
plt.tight_layout()
plt.show()
