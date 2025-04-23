# Import libraries
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import scikit_posthocs as sp

hf = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
alc = pd.read_csv('data/student-por.csv')

# 2. Descriptive statistics
print("--- Descriptive Statistics ---")
print("Heart Failure:")
print(hf.describe(), end="\n\n")


# 4. Non-parametric tests (min 2)
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

# # 5. Friedman test + post-hoc Nemenyi + Visualization
# print("--- Friedman Test + Post-hoc Nemenyi + Visualization ---")
#
# # 5.0 Konwersja na numeryczne i usunięcie braków
# cols = ['G1','G2','G3']
# alc[cols] = alc[cols].apply(pd.to_numeric, errors='coerce')
# df_fried = alc[cols].dropna()
# if df_fried.empty:
#     print("Brak kompletnych danych w G1–G3 → pomijam test Friedmana.")
# else:
#     # właściwy Friedman
#     f_stat, p_f = stats.friedmanchisquare(
#         df_fried['G1'], df_fried['G2'], df_fried['G3']
#     )
#     print(f"Friedman test on G1,G2,G3: chi2 = {f_stat:.3f}, p = {p_f:.3f}\n")
#
#     if p_f < 0.05:
#         print("Friedman istotny (p < 0.05) — przeprowadzam post-hoc Nemenyi.")
#         nem = sp.posthoc_nemenyi_friedman(df_fried)
#         nem = pd.DataFrame(nem, index=cols, columns=cols)
#         print("Post-hoc Nemenyi p-values:\n", nem, "\n")
#
#         plt.figure()
#         sns.heatmap(nem, annot=True, fmt=".3f", cbar_kws={'label':'p-value'},
#                     square=True, linewidths=.5)
#         plt.title("Nemenyi post-hoc p-values")
#         plt.ylabel("Semester")
#         plt.xlabel("Semester")
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("Friedman nieistotny (p >= 0.05) — pomijam post-hoc Nemenyi.\n")


# 6. Kaplan-Meier survival curves
print("--- Kaplan-Meier Survival Curve ---")
kmf = KaplanMeierFitter()
T = hf['time']  # duration
E = hf['DEATH_EVENT']  # event occurred
kmf.fit(durations=T, event_observed=E)
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.tight_layout()
plt.show()

# 7. Logistic regression + ROC
print("--- Logistic Regression + ROC Curve ---")
# Prepare data
X = hf[['age','ejection_fraction','serum_creatinine']]
y = hf['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]
# ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1], [0,1], linestyle='--')
plt.title('ROC Curve - Heart Failure')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()
