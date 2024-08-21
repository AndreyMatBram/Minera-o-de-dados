import itertools
import json
from io import StringIO
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from Mapper import BaseMapper

def _make_map_adapter(column: str):
    return BaseMapper.get_mapper(column).map

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
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], '.2f'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    else:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    

def main():

    io = StringIO(open("./entry.json", "r").read())
    entry = json.load(io)
    
    categorical = entry["categorical"]

    data = pd.read_csv('Dataset\ObesityDataSet.csv',
                    na_values='?',
                    converters={
                        col: _make_map_adapter(col) for col in categorical
                    },
                )


    X = data.drop('NObeyesdad', axis=1)
    y = data["NObeyesdad"]
    
    # Split the data - 75% train, 25% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)    

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = DecisionTreeClassifier(random_state=27)
    clf.fit(X_train, y_train)
    tree.plot_tree(clf)
    print( f' Profundidade: {clf.get_depth()}')

    predictions = clf.predict(X_test)
    
    accuracy = clf.score(X_test, y_test)
    # Get test confusion matrix    
    cm = confusion_matrix(y_test, predictions)        
    plot_confusion_matrix(cm, ["0","1","2","3","4","5","6"], False, f'MC accuracy: {accuracy}')      
    plot_confusion_matrix(cm, ["0","1","2","3","4","5","6"], True, f'MC Normalize accuracy: {accuracy}' ) 

    plt.show()
    
    


if __name__ == "__main__":
    main()