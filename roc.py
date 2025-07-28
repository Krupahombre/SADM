# import pandas as pd
# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt
#
# # 1) Load data
# df = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
# X = df.drop(['time', 'DEATH_EVENT'], axis=1)
# y = df['DEATH_EVENT']
#
# # 2) Parameters
# Cs = [0.25, 0.50, 0.75]
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# mean_fpr = np.linspace(0, 1, 100)
#
# plt.figure(figsize=(8, 6))
#
# # 3) Compute ROC for each C
# for C in Cs:
#     tprs = []
#     aucs = []
#     for train_idx, test_idx in cv.split(X, y):
#         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#         y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#
#         # Pipeline: scaling + logistic regression
#         scaler = StandardScaler().fit(X_train)
#         clf = LogisticRegression(C=C, solver='liblinear', max_iter=1000)
#         X_train_scaled = scaler.transform(X_train)
#         clf.fit(X_train_scaled, y_train)
#
#         X_test_scaled = scaler.transform(X_test)
#         probas = clf.predict_proba(X_test_scaled)[:, 1]
#
#         fpr, tpr, _ = roc_curve(y_test, probas)
#         roc_auc = auc(fpr, tpr)
#         aucs.append(roc_auc)
#
#         # Interpolate TPR
#         tprs.append(np.interp(mean_fpr, fpr, tpr))
#         tprs[-1][0] = 0.0
#
#     # Compute mean and std
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = np.mean(aucs)
#     std_auc = np.std(aucs)
#
#     plt.plot(mean_fpr, mean_tpr,
#              label=f'C={C} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
#
# # Plot diagonal
# plt.plot([0, 1], [0, 1], linestyle='--')
#
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curves for Logistic Regression (CV)')
# plt.legend(loc='lower right')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

df = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
X = df.drop(['time', 'DEATH_EVENT'], axis=1)
y = df['DEATH_EVENT']

Cs = [0.25, 0.50, 0.75]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mean_fpr = np.linspace(0, 1, 100)

plt.figure(figsize=(8, 6))

for C in Cs:
    tprs = []
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler().fit(X_train)
        model = LogisticRegression(C=C, solver='liblinear', max_iter=1000)
        X_train_scaled = scaler.transform(X_train)
        model.fit(X_train_scaled, y_train)

        X_test_scaled = scaler.transform(X_test)
        probas = model.predict_proba(X_test_scaled)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, probas)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        tpr_i = np.interp(mean_fpr, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr,
             label=f'C={C} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')

plt.plot([0, 1], [0, 1], '--', label='Random')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curves for Logistic Regression (5-fold CV)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

