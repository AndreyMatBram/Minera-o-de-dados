import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Dataset\ObesityDataSet.csv',
                   na_values='?')

#mapeamento da classificação, pois o factorize fica fora de ordem por utilizar ordem alfabetica
mapa = {'Insufficient_Weight':0, 'Normal_Weight':1, 'Overweight_Level_I':2, 'Overweight_Level_II':3, 'Obesity_Type_I':4, 'Obesity_Type_II':5, 'Obesity_Type_III':6 }

#cria nova coluna com numeros que refletem o tipo de obesidade baseado ao mapeamento
data['NumObey'] = data['NObeyesdad'].map(mapa)


hasHistory = len(data[(data['family_history_with_overweight'] == 'yes')])
obesityWithHistory = len(data[(data['family_history_with_overweight'] == 'yes') & (data['NumObey'] >= 4)])
obesityWithOutHistory = len(data[(data['family_history_with_overweight'] == 'no') & (data['NumObey'] >= 4)])
nonObesityWithHistory = len(data[(data['family_history_with_overweight'] == 'yes') & (data['NumObey'] < 4)])

# Dados
categories = ["C/ Histórico", "Obeso c/ Histórico", "Obeso s/ Histórico", "Não Obeso c/Histórico"]
values = [hasHistory, obesityWithHistory, obesityWithOutHistory, nonObesityWithHistory]

# Criar o gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(categories, values, color=["blue", "green", "red", "purple"])
plt.ylabel("Contagem")
plt.title("Contagem de Obesidade com e sem Histórico Familiar")
plt.show()

