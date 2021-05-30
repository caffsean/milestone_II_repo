import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import plotly.express as px
import plotly.io as pio
pio.renderers.default='notebook'
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from sklearn.feature_selection import SequentialFeatureSelector # need version 0.24 of scikit-learn


def knn(X_train1, y_train1, X_test1, y_test1, neighbors = 5):
    '''Train a k-nearest neighbor model and output the model, score, accuracy, precision, and recall'''
    knn = KNeighborsClassifier(n_neighbors = neighbors)
    knn.fit(X_train1, y_train1.ravel())
    predicted = knn.predict(X_test1)
    score = knn.score(X_train1, y_train1.ravel())
    accuracy = accuracy_score(y_test1.ravel(), predicted)
    precision = precision_score(y_test1, predicted)
    recall = recall_score(y_test1, predicted)
    print("K-Nearest Neighbors: Score - " + str(score) + " Accuracy - " + str(accuracy) + " Precision - " + str(precision) + " Recall - " + str(recall))
    return knn

def MNB(X_train1, y_train1, X_test1, y_test1, alpha = 1.5, fit_prior = True):
    '''Train a multinomial naive bayes model and output the model, score, accuracy, precision, and recall'''
    clf = MultinomialNB(alpha = alpha, fit_prior = fit_prior).fit(X_train1, y_train1.ravel())
    predicted = clf.predict(X_test1)
    score = clf.score(X_train1,y_train1.ravel())
    accuracy = accuracy_score(y_test1.ravel(), predicted)
    precision = precision_score(y_test1, predicted)
    recall = recall_score(y_test1, predicted)
    print("Multinomial NB: Score - " + str(score) + " Accuracy - " + str(accuracy) + " Precision - " + str(precision) + " Recall - " + str(recall))
    return clf

def log_reg(X_train1, y_train1, X_test1, y_test1, max_iter = 100, C=1.0, penalty = 'l2', solver = 'lbfgs'):
    '''Train a random logistic regression model and output the model, score, accuracy, precision, and recall'''
    logreg = LogisticRegression(max_iter=max_iter, C = C,random_state=655,solver=solver, penalty=penalty).fit(X_train1, y_train1.ravel())
    predicted = logreg.predict(X_test1)
    score = logreg.score(X_train1,y_train1.ravel())
    accuracy = accuracy_score(y_test1.ravel(), predicted)
    precision = precision_score(y_test1, predicted)
    recall = recall_score(y_test1, predicted)
    print("Logistic Regression: Score - " + str(score) + " Accuracy - " + str(accuracy) + " Precision - " + str(precision) + " Recall - " + str(recall))
    return logreg

def random_forest(X_train1, y_train1, X_test1, y_test1, max_depth = None, n_estimators = 100, min_samples_leaf = 1,
                 n_jobs = -1):
    '''Train a random forest model and output the model, score, accuracy, precision, and recall'''
    clf = RandomForestClassifier(max_depth=max_depth, n_estimators = n_estimators, random_state=911, 
                                 min_samples_leaf = min_samples_leaf, n_jobs = n_jobs)
    clf.fit(X_train1, y_train1.ravel())
    score = clf.score(X_train1,y_train1.ravel())
    predicted = clf.predict(X_test1)
    accuracy = accuracy_score(y_test1.ravel(), predicted)
    precision = precision_score(y_test1, predicted)
    recall = recall_score(y_test1, predicted)
    print("Random Forest: Score - " + str(score) + " Accuracy - " + str(accuracy) + " Precision - " + str(precision) + " Recall - " + str(recall))
    return clf

def create_roc(model, X_test, y_test):
    '''Create the ROC Curve visual and calculate AUC'''
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    fig = px.line(x = fpr, y = tpr, title = 'AUC: {}'.format(auc), hover_data = [thresholds])
    fig.show("notebook")


def confusion_matrix_visual(trained_model, X_test, y_test):
    '''Create a confusion matrix heatmap'''
    
    plot_confusion_matrix(trained_model, X_test, y_test, cmap = plt.cm.Blues)
    

def feature_importance(trained_model, columns):
    '''Creates a dataframe with all the features and their importance in descending order'''

    try:
        imps = trained_model.feature_importances_
    except:
        imps = trained_model.coef_[0]

    imp_df = pd.DataFrame(columns,columns=['features'])
    imp_df['importance'] = imps
    imp_df['importance_abs'] = imp_df['importance'].abs()

    return imp_df.sort_values(by='importance_abs',ascending=False)

def grid_search_features(features, X_train, y_train, X_test, y_test, n_estimators = 100, max_depth = None):
    '''Helps to see how many and what features need to be included in the training'''
    counts = range(5, len(features), 5)
    
    for count in counts:
        cols = features[:count]['features']
        X_train_ = X_train[cols]
        X_test_ = X_test[cols]
        print('Top {} features'.format(count))
        random_forest(X_train_, y_train, X_test_, y_test,n_estimators=n_estimators,max_depth=max_depth)
        print()
        
def tune_threshold(trained_model, X_test, y_test, threshold = .5):
    '''Tweak the threshold to see if can increase accuracy'''
    probs = trained_model.predict_proba(X_test)[:,1]
    predictions = []
    for each in probs:
        if each > threshold:
             predictions.append(1)
        else:
            predictions.append(0)
    accuracy = accuracy_score(y_test.ravel(), predictions)
    return accuracy