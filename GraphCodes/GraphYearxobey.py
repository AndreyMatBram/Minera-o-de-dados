import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Dataset\ObesityDataSet.csv',
                   na_values='?')

# Mapeamento da classificação
mapaObey = {'Insufficient_Weight':0, 'Normal_Weight':1, 'Overweight_Level_I':2, 'Overweight_Level_II':3, 'Obesity_Type_I':4, 'Obesity_Type_II':5, 'Obesity_Type_III':6 }

# Cria nova coluna com números que refletem o tipo de obesidade baseado no mapeamento
data['NumObey'] = data['NObeyesdad'].map(mapaObey)

# Define cores para os tipos de obesidade
color = data['FAVC'].map(lambda x: 'red' if x == 'yes' else 'blue' )

data.plot(kind='scatter', x='Age', y='NumObey', color=color)

plt.title('Grafico AgexNObey')
plt.xlabel('Idade')
plt.ylabel('Classe de Obesidade')
plt.show()