from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib import pyplot as plt
import pandas as pd

hf = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")
kmf = KaplanMeierFitter()

fig1, ax1 = plt.subplots()
kmf.fit(hf["time"], event_observed=hf["DEATH_EVENT"], label="wszyscy")
kmf.plot_survival_function(ax=ax1)
ax1.set_xlabel("Czas (dni)")
ax1.set_ylabel("Prawdopodobieństwo przeżycia")
ax1.set_title("Krzywa przeżycia - wszyscy pacjenci")
fig1.tight_layout()

fig2, ax2 = plt.subplots()
for grp, lbl in [(1, "palący"), (0, "nie palący")]:
    mask = hf["smoking"] == grp
    kmf.fit(
        hf.loc[mask, "time"],
        event_observed=hf.loc[mask, "DEATH_EVENT"],
        label=lbl
    )
    kmf.plot_survival_function(ax=ax2)

ax2.set_xlabel("Czas (dni)")
ax2.set_ylabel("Prawdopodobieństwo przeżycia")
ax2.set_title("Krzywe przeżycia Kaplana–Meiera\npalący vs niepalący")
fig2.tight_layout()

plt.show()

result = logrank_test(
    hf.loc[hf.smoking == 1, "time"],
    hf.loc[hf.smoking == 0, "time"],
    event_observed_A=hf.loc[hf.smoking == 1, "DEATH_EVENT"],
    event_observed_B=hf.loc[hf.smoking == 0, "DEATH_EVENT"]
)
print(result)
