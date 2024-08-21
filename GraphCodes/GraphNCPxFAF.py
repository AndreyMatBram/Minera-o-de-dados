import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Dataset\ObesityDataSet.csv',
                   na_values='?')

#mapeamento da classificação, pois o factorize fica fora de ordem por utilizar ordem alfabetica
mapa = {'Insufficient_Weight':0, 'Normal_Weight':1, 'Overweight_Level_I':2, 'Overweight_Level_II':3, 'Obesity_Type_I':4, 'Obesity_Type_II':5, 'Obesity_Type_III':6 }

#cria nova coluna com numeros que refletem o tipo de obesidade baseado ao mapeamento
data['NumObey'] = data['NObeyesdad'].map(mapa)


#Define cores para os pontos azul = não obeso, vermelho= obeso
color = data['NumObey'].map(lambda x: 'blue' if x < 4 else 'red')

data.plot(kind='scatter', x='NCP', y='FAF', color=color)
plt.title('Grafico NumRefei x FreqAttFisicas')
plt.xlabel('Numero de Refeições')
plt.ylabel('Frequencia de Atividades Fisicas')
plt.show()
