import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

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

# Calculate distance between two points
def minkowski_distance(a, b, p=1):    
    # Store the number of dimensions
    dim = len(a)    
    # Set initial distance to 0
    distance = 0
    
    # Calculate minkowski distance using parameter p
    for d in range(dim):
        distance += abs(a[d] - b[d])**p
        
    distance = distance**(1/p)    
    return distance


def knn_predict(X_train, X_test, y_train, y_test, k, p):    
    # Make predictions on the test data
    # Need output of 1 prediction per test data point
    y_hat_test = []

    for test_point in X_test:
        distances = []

        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)
        
        # Store distances in a dataframe
        df_dists = pd.DataFrame(data=distances, columns=['dist'], 
                                index=y_train.index)
        
        # Sort distances, and only consider the k closest points
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

        # Create counter object to track the labels of k closest neighbors
        counter = Counter(y_train[df_nn.index])

        # Get most common label of all the nearest neighbors
        prediction = counter.most_common()[0][0]
        
        # Append prediction to output list
        y_hat_test.append(prediction)
        
    return y_hat_test


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], '.2f'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    else:
        print('Confusion matrix, without normalization')
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    print(cm)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    


X = data.drop(['NumObey', 'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad'], axis=1)
y = data['NumObey']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# STEP 2 - TESTS USING knn classifier from sk-learn
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_hat_test = knn.predict(X_test)

# Get test accuracy score
accuracy = accuracy_score(y_test, y_hat_test)*100
f1 = f1_score(y_test, y_hat_test,average='macro')*100
print("Acurracy K-NN from sk-learn: {:.2f}%".format(accuracy))
print("F1 Score K-NN from sk-learn: {:.2f}%".format(f1))

# Get test confusion matrix    
cm = confusion_matrix(y_test, y_hat_test)        
plot_confusion_matrix(cm, ["0","1","2","3","4","5","6"], False, f'MC accuracy: {accuracy}')      
plot_confusion_matrix(cm, ["0","1","2","3","4","5","6"], True, f'MC Normalize accuracy: {accuracy}' )  

plt.show()




