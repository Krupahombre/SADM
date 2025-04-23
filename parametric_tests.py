import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris
from statsmodels.stats.oneway import anova_oneway
import pingouin as pg
import matplotlib.pyplot as plt

# 1. Wczytanie Iris do DataFrame
iris = load_iris(as_frame=True)
df = iris.frame.rename(columns={
    'sepal length (cm)': 'sepal_length',
    'sepal width (cm)':  'sepal_width',
    'petal length (cm)': 'petal_length',
    'petal width (cm)':  'petal_width',
    'target':            'species'
})
df['species'] = df['species'].map(lambda i: iris.target_names[i])

# 2. Statystyki opisowe
def descriptive_stats(df):
    print("--- Descriptive Statistics by Species ---")
    desc = df.groupby('species')[['sepal_length','sepal_width','petal_length','petal_width']].mean()
    print(desc, "\n")

descriptive_stats(df)

# 3. Student's t-test: sepal_width dla setosa vs versicolor
print("--- Student's t-test: sepal_width (setosa vs versicolor) ---")
x = df[df['species']=='setosa']['sepal_width']
y = df[df['species']=='versicolor']['sepal_width']

# 3.1 Sprawdzenie normalności (Shapiro–Wilk)
for grp, data in [('setosa', x), ('versicolor', y)]:
    w, p = stats.shapiro(data)
    print(f"Shapiro {grp}: W = {w:.3f}, p = {p:.3f}")

# 3.2 Homogeniczność wariancji (Levene)
w_lev, p_lev = stats.levene(x, y)
print(f"Levene: W = {w_lev:.3f}, p = {p_lev:.3f}")

equal_var = p_lev > 0.05
# 3.3 Test t-Studenta
t_stat, p_val = stats.ttest_ind(x, y, equal_var=equal_var)
print(f"t-test (equal_var={equal_var}): t = {t_stat:.3f}, p = {p_val:.3f}\n")

# 4. Welch's ANOVA: petal_length dla wszystkich gatunków
print("--- Welch's ANOVA: petal_length across species ---")
# grupy
groups = [df[df['species']==s]['petal_length'] for s in iris.target_names]

# 4.1 Sprawdzenie normalności w każdej grupie (Shapiro–Wilk)
for s, data in zip(iris.target_names, groups):
    w, p = stats.shapiro(data)
    print(f"Shapiro {s}: W = {w:.3f}, p = {p:.3f}")

# 4.2 Homogeniczność wariancji (Levene)
w_lev_all, p_lev_all = stats.levene(*groups)
print(f"Levene (all groups): W = {w_lev_all:.3f}, p = {p_lev_all:.3f}")

# 4.3 Welch's ANOVA
tmp = pd.DataFrame({ 'petal_length': df['petal_length'], 'species': df['species'] })
welch = anova_oneway(tmp['petal_length'], tmp['species'], use_var='unequal')
print(f"Welch ANOVA F = {welch.statistic:.3f}, p = {welch.pvalue:.3f}\n")

# 5. Post-hoc test Games–Howell przy użyciu pingouin
print("--- Games–Howell post-hoc test for petal_length ---")
gh = pg.pairwise_gameshowell(data=df, dv='petal_length', between='species')
print(gh, "\n")

# 6. Wizualizacja: boxplot
print("--- Boxplot of petal_length by species ---")
plt.figure()
df.boxplot(column='petal_length', by='species')
plt.title('Petal length by species')
plt.suptitle('')
plt.xlabel('Species')
plt.ylabel('Petal length (cm)')
plt.tight_layout()
plt.show()
