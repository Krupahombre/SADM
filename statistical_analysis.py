import pandas as pd

hf = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
alc = pd.read_csv('data/student-por.csv')

print("--- Descriptive Statistics ---")
print("Heart Failure:")
print(hf.describe(), end="\n\n")