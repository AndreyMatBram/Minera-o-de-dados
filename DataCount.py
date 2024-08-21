import pandas as pd

data = pd.read_csv('Dataset\ObesityDataSet.csv',
                   na_values='?')
ContGender = data['Gender'].value_counts()
ContFamily = data['family_history_with_overweight'].value_counts()
ContFAVC = data['FAVC'].value_counts()
ContCAEC = data['CAEC'].value_counts()
ContSMOKE = data['SMOKE'].value_counts()
ContSCC = data['SCC'].value_counts()
ContCALC = data['CALC'].value_counts()
ContMTRANS = data['MTRANS'].value_counts()
ContNObeyesdad = data['NObeyesdad'].value_counts()

print(ContGender, '\n\n---------\n')
print(ContFamily, '\n\n---------\n')
print(ContFAVC, '\n\n---------\n')
print(ContCAEC, '\n\n---------\n')
print(ContSMOKE, '\n\n---------\n')
print(ContSCC, '\n\n---------\n')
print(ContCALC, '\n\n---------\n')
print(ContMTRANS, '\n\n---------\n')
print(ContNObeyesdad, '\n\n---------\n')