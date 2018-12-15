import numpy as np
import pandas as pd
from lime import lime_text
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing as preprocessing
import lime
import lime.lime_tabular
from lime.lime_text import LimeTextExplainer
from sklearn.neural_network import MLPClassifier

target_names = np.array(['Home wins', 'Away wins'])

def misclassified_games(X, y, home, away, predictions):
    """
    misclassified = misclassified_games(X, y, home, away, predictions)

    Locates and outputs all games misclassified by a scikit-learn binary
    classifier

    inputs:
        - X           (pandas DataFrame): feature vectors
        - y           (pandas DataFrame: feature labels matching the feature
                                         vectors
        - home        (pandas DataFrame): home team names
        - away        (pandas DataFrame): away team names
        - predictions (numpy ndarray): predicted winners (0 for home and 1 for
                                       away)

    outputs:
        - a list of each misclassified game with the predicted and
        actual winners printed to the screen
        - misclassified (pandas Datafame): all misclassified games, predicted
                                           winners, and actual winners

    """
    cols = ['Home', 'Away', 'Predicted_winner', 'Actual_winner']
    misclassified=pd.DataFrame(columns = cols)

    for predict, label, home, away in zip(predictions, y, home, away):
        #result = predictions[item]

        if predict != label:
            if predict == 0:
                predicted_winner = home
                actual_winner = away
            else:
                predicted_winner = away
                actual_winner = home

            print( "For %s-%s, predicted winner %s, but true winner is %s" % (home, away, predicted_winner, actual_winner))
            missed_game = pd.DataFrame([home, away, predicted_winner, actual_winner])

    return


def choose_misclassified(X, home, away, game):
    """
    desired_game = choose_misclassified(X, home, away, game)

    Choose a single misclassified game to interrogate with the lime package.
    Returns an index corresponding to the desired game that can be passed to the
    function lime_explainer

    inputs:
        - X    (pandas DataFrame): feature vectors
        - home (pandas DataFrame): home team names
        - away (pandas DataFrame): away team names
        - game (list): a list containing two strings, the home team and away
                       teams for the desired game to interrogate, i.e.
                       ['Wisconsin', 'Minnesota']

    outputs:
        - i (integer): an index representing the desired game instance to
                       interrogate with the package lime

    """

    rows=list(range(len(X)))

    for i, home, away, X in zip(rows, home, away, X):
        if home == game[0] and away == game[1]:
            desired_game = i

    return desired_game


def lime_explainer(X_train, X_test, cols, game=[], home=None, away=None, classifier='mlp',
                   num_features=10, save=False):
    """
    exp = lime_explainer(X, cols, game=[], home=None, away=None, classifier='mlp',
                       num_features=10, save=False)

    explains an instance of a misclassified game. displays several figures
    related to the classification.

    inputs:
        - X_train      (pandas DataFrame): training feature vectors
        - X_test       (pandas DataFrame): testing feature vectors
        - cols         (list): list of columns from the pre-scaled data
        - game         (list): a list containing two strings, the home team and
                               away teams for the desired game to interrogate,
                               i.e. ['Wisconsin', 'Minnesota']. If a game isn't
                               chosen, the first game of the testing set will
                               be picked
        - home (pandas DataFrame): home team names
        - away (pandas DataFrame): away team names
        - classifier   (string): the classifier to use, from choices
                                 'mlp' = Multi-layer Perceptron,
                                 'svm' = Support Vector Machine,
                                 'rf'  = Random Forest
        - num_features (int): the number of features to be analyzed

    outputs:
        none
    """
    try:
        if game:
            if home != None:
                if away != None:
                    print('Acceptable game')
        elif not game:
            print('No game chosen')
    except:
        print('Unacceptable game or error was made')
        return


    if game:
        desired_game = choose_misclassified(X, home, away, game)
        target_names = np.array(game)
    elif not game:
        desired_game = 1
        target_names = np.array(['Home wins', 'Away wins'])
        print("First game in testing set will be used")

    explainer = lime.lime_tabular.LimeTabularExplainer(training_data = X_train,
                                                   mode = "classification",
                                                   feature_names=cols,
                                                   class_names = target_names,
                                                   discretize_continuous=True)

    if classifier == 'mlp':
        exp = explainer.explain_instance(X_test[desired_game], mlp.predict_proba,
                                         num_features= num_features)
    elif classifier == 'svm':
        exp = explainer.explain_instance(X_test[desired_game], svm.predict_proba,
                                         num_features= num_features)
    elif classifier == 'rf':
        exp = explainer.explain_instance(X_test[desired_game], rf.predict_proba,
                                         num_features= num_features)
    else:
        print("incorrect classifier chosen. Please select from ['mlp', 'svm', 'rf']")

    return exp

    exp.show_in_notebook(show_table=True, show_all=True)

    exp.as_pyplot_figure()

    if save:
        if game:
            exp.save_to_file('%s-%s.html' % (game[0], game[1]))
        else:
            print('You did not specify a game. No file being saved')
