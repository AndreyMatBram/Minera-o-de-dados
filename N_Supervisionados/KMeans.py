import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler

def plot_samples(projected, labels, title):
    fig = plt.figure()
    u_labels = np.unique(labels)
    for i in u_labels:
        plt.scatter(projected[labels == i , 0] , projected[labels == i , 1] , label = i,
                    edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('tab10', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.legend()
    plt.title(title)

data = pd.read_csv('Dataset\ObesityDataSet.csv',
                   na_values='?')

# Mapeamento da classificação
mapaSex = {'Male':0, "Female":1}
mapaFamily = {'yes':0, 'no':1}
mapaFAVC = {'yes':0, 'no':1}
mapaCAEC = {'no':0, 'Sometimes':1, 'Frequently':2, 'Always':3}
mapaSMOKE = {'no':0, 'yes':1}
mapaSCC = {'no':0, 'yes':1}
mapaCALC = {'no':0, 'Sometimes':1, 'Frequently':2, 'Always':3}
mapaMTRANS = {'Automobile':0, 'Motorbike':1, 'Bike':2, 'Public_Transportation':3, 'Walking':4}
mapaObey = {'Insufficient_Weight':0, 'Normal_Weight':1, 'Overweight_Level_I':2, 'Overweight_Level_II':3, 'Obesity_Type_I':4, 'Obesity_Type_II':5, 'Obesity_Type_III':6 }

# Cria nova coluna com números que refletem o tipo de obesidade baseado no mapeamento
data['NumSex'] = data['Gender'].map(mapaSex)
data['NumFamily'] = data['family_history_with_overweight'].map(mapaFamily)
data['NumFAVC'] = data['FAVC'].map(mapaFAVC)
data['NumCAEC'] = data['CAEC'].map(mapaCAEC)
data['NumSMOKE'] = data['SMOKE'].map(mapaSMOKE)
data['NumSCC'] = data['SCC'].map(mapaSCC)
data['NumCALC'] = data['CALC'].map(mapaCALC)
data['NumMTRANS'] = data['MTRANS'].map(mapaMTRANS)
data['NumObey'] = data['NObeyesdad'].map(mapaObey)

# Criar o PCA
pca = PCA(n_components=2)

target = 'NumObey'
features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'FAF', 'TUE']

x = data.loc[:, features].values
y = data.loc[:,[target]].values

x = StandardScaler().fit_transform(x)


# Aplicar o PCA nas 9 colunas
pca_data = pca.fit_transform(x)

bestScore,b=0, 0

for i in range(2,20):
    kmeans=KMeans(n_clusters=i, random_state=27).fit(pca_data)
    score=silhouette_score(labels=kmeans.labels_,
                       X=pca_data)
    if bestScore<score:
        bestScore=score
        b=i

kmeans=KMeans(n_clusters=b).fit(pca_data)

plot_samples(pca_data, kmeans.labels_, f"score {round(bestScore,2)} , Clusters: {b}")

plt.show()




