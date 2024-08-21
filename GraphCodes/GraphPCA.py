import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler

data = pd.read_csv('Dataset\ObesityDataSet.csv',
                   na_values='?')

#mapeamento da classificação, pois o factorize fica fora de ordem por utilizar ordem alfabetica
mapa = {'Insufficient_Weight':0, 'Normal_Weight':1, 'Overweight_Level_I':2, 'Overweight_Level_II':3, 'Obesity_Type_I':4, 'Obesity_Type_II':5, 'Obesity_Type_III':6 }

#cria nova coluna com numeros que refletem o tipo de obesidade baseado ao mapeamento
data['NumObey'] = data['NObeyesdad'].map(mapa)


#Define cores para os pontos azul = não obeso, vermelho= obeso
color = data['NumObey'].map(lambda x: 'blue' if x < 4 else 'red')

#cria o PCA
pca = PCA(n_components=3)
target = 'NumObey'
features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'FAF', 'TUE']

x = data.loc[:, features].values
y = data.loc[:,[target]].values

x = StandardScaler().fit_transform(x)
principalComponents = pca.fit_transform(x)

# Plot the 3D scatter plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(principalComponents[:, 0], principalComponents[:, 1], principalComponents[:, 2], c=color)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

plt.show()
