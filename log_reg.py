import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
X = df.drop(['time', 'DEATH_EVENT'], axis=1)
y = df['DEATH_EVENT']

Cs = [0.25, 0.50, 0.75]
scoring = ['accuracy', 'balanced_accuracy', 'precision', 'recall']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

records = []
for C in Cs:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=C, solver='liblinear', max_iter=1000))
    ])
    cv_res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, return_train_score=False)

    rec = {'C': C}
    for m in scoring:
        scores = cv_res[f'test_{m}']
        rec[m] = f"{scores.mean():.4f} ± {scores.std():.4f}"
    records.append(rec)

results_df = pd.DataFrame(records)
results_df.rename(columns={
    'accuracy': 'Accuracy',
    'balanced_accuracy': 'Balanced Accuracy',
    'precision': 'Precision',
    'recall': 'Recall'
}, inplace=True)

print(results_df.to_string(index=False))
