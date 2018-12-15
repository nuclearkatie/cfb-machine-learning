import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import itertools


def test_predictions(y, predictions, title, plot_cm=True, cmap=plt.cm.Greens):

    cm = confusion_matrix(y,predictions)
    
    print(classification_report(y,predictions))
    print("This set of predictions was %s%% accurate\n" %
          str(round(100*accuracy_score(y,predictions),2)))

    if plot_cm == True:
        plt.figure()
        plot_confusion_matrix(cm, classes=('Home Wins', 'Away Wins'),
                      title=title, cmap=cmap)

    return cm

def plot_confusion_matrix(cm, classes=('0','1'),
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
