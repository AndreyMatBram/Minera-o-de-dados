import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Dataset\ObesityDataSet.csv',
                   na_values='?')

#mapeamento da classificação, pois o factorize fica fora de ordem por utilizar ordem alfabetica
mapaObey = {'Insufficient_Weight':0, 'Normal_Weight':1, 'Overweight_Level_I':2, 'Overweight_Level_II':3, 'Obesity_Type_I':4, 'Obesity_Type_II':5, 'Obesity_Type_III':6 }
mapaBetFood = {'no':0, 'Sometimes':1, 'Frequently':2, 'Always':3}

#cria nova coluna com numeros que refletem o tipo de obesidade baseado ao mapeamento
data['NumObey'] = data['NObeyesdad'].map(mapaObey)

#agora para a frequencia de consumo entre refeiçoes 
data['NumCAEC'] = data['CAEC'].map(mapaBetFood)

#Define cores para os tipos de obesidade
color = data['FAVC'].map(lambda x: 'red' if x == 'yes' else 'blue' )

data.plot(kind='scatter', x='NumCAEC', y='NumObey', color=color)

plt.title('Grafico CAECxNObey')
plt.xlabel('Consumo entre Refeição')
plt.ylabel('Classe de Obesidade')
plt.show()