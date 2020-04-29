# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objects as go
from plotly.tools import mpl_to_plotly

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets



def gen_data():
    mean = [2.5, 1.5]
    cov = [[1.4, 1], [1.2, 0.5]]

    F, P = np.random.multivariate_normal(mean, cov, 2000).T

    idx = []
    for i in range(len(F)):
        if F[i] < 0 or P[i] < 0:
            idx.append(i)

    F = np.delete(F, idx, 0)
    P = np.delete(P, idx, 0)
    # X = np.hstack((F, P))

    X = np.zeros((len(F), 2))
    for i in range(len(F)):
        X[i, 0] = F[i]
        X[i, 1] = P[i]
    return F, P, X

def labels(F, P):
    cat = np.zeros(len(F))
    for i in range(len(P)):
        alpha_0 = 1

        if P[i] > 3:
            alpha = (P[i] - 3) * 0.3
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            #cat[i] = 1

        if F[i] > 5:
            alpha = (F[i] - 5) * 0.3
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            #cat[i] = 1

        if P[i] > 3 and F[i] > 5:
            alpha = ((F[i] - 5) * 0.1 + (P[i] - 3) * 0.1) / 2
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0] + 1
            #cat[i] = 1

        if F[i] < 0.5:
            alpha = np.abs(F[i] - 0.5) * 0.3
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            #cat[i] = 0
    return cat

F, P, X = gen_data()
y = labels(F, P)

# shuffle and split training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                    random_state=0)
F_train = X_train[:, 0]
P_train = X_train[:, 1]
F_test = X_test[:, 0]
P_test = X_test[:, 1]

cdict = {0: 'green', 1: 'orange', 2: 'red'}
labeldict = {0: 'Gutteil', 1: 'Nachbearbeiten', 2: 'Ausschuss'}

# plot training data
plt.grid(True)
for g in np.unique(y_train):
    ix = np.where(y_train == g)
    plt.scatter(F_train[ix], P_train[ix], c = cdict[g], label = labeldict[g])
legend = plt.legend(loc="lower right", title="Legende")
plt.xlabel('Kzraft in kN')
plt.ylabel('Leistung in kW')
plt.title('Trainingsdaten Station Spanen')
plt.show()

# plot test data
plt.grid(True)
for g in np.unique(y_test):
    ix = np.where(y_test == g)
    plt.scatter(F_test[ix], P_test[ix], c = cdict[g], label = labeldict[g])
legend = plt.legend(loc="lower right", title="Legende")
plt.xlabel('Kzraft in kN')
plt.ylabel('Leistung in kW')
plt.title('Testdaten Station Spanen')
plt.show()

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['green', 'orange', 'red'])
cmap_bold = ListedColormap(['darkgreen', 'darkorange', 'darkred'])

n_neighbors = 5

for weights in ['uniform']:
# for weights in ['uniform', 'distance']:
    from sklearn.multiclass import OneVsRestClassifier

    # we create an instance of Neighbours Classifier and fit the data.
    classifier = OneVsRestClassifier(neighbors.KNeighborsClassifier(n_neighbors, weights=weights))
    clf = classifier.fit(X_train, y_train)
    # clf.fit(X, y)

    # # ROC
    # y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    # # Compute ROC curve and ROC area for each class
    # fpr = dict()
    # tpr = dict()sklearn.metrics
    # roc_auc = dict()
    # n_classes = 2
    # from sklearn.metrics import roc_curve, auc
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])sklearn.metrics
    #
    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #
    # # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    #
    # # Then interpolate all ROC curves at this points
    # from scipy import interp
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(n_classes):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    #
    # # Finally average it and compute AUC
    # mean_tpr /= n_classes
    #
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #
    # # Plot all ROC curves
    # plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)
    #
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(i, roc_auc[i]))
    #
    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    # plt.legend(loc="lower right")
    # plt.show()

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    # Plot with training points
    fig_train = plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.xlabel('Prozesskraft in kN')
    plt.ylabel('Leistung in kW')
    plt.title('Klassifizierungsergebnis Station Spanen + TRAININGS')
    plt.show()

    # Plot with test points
    fig_test = plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.xlabel('Prozesskraft in kN')
    plt.ylabel('Leistung in kW')
    plt.title('Klassifizierungsergebnis Station Spanen + TESTdatan')
    plt.show()

    # Confusion Matrix on Training and on Test Data
    from sklearn.metrics import confusion_matrix
    y_pred = clf.predict(X_test)
    #confusion_train = confusion_matrix(y_train, Z)
    confusion_test = confusion_matrix(y_test, y_pred)
    print(confusion_test)

    # plot confusion matrix
    from sklearn.metrics import plot_confusion_matrix
    # titles_options = [("Konfusionsmatrix ohne Normalisierung", None),
    #                   ("Konfusionsmatrix mit Normalisierung", 'true')]
    titles_options = [("Konfusionsmatrix mit Normalisierung", 'true')]
    class_names = ['Gutteil', 'Nachbearbeiten', 'Ausschuss']
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)
        disp.ax_.set_xlabel('Vorhergesagte Klasse')
        disp.ax_.set_ylabel('Wahre Klasse')

        print(title)
        print(disp.confusion_matrix)

plt.show()

# plot a classification report
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)

# fig_test, fig_train = plotting()

plotly_fig_train = mpl_to_plotly(disp)
plotly_fig_test = mpl_to_plotly(disp)

# from plotly import offline
# plotly_fig_test = offline.plot(plotly_fig_test, filename="plotly version of an mpl figure 1")
# plotly_fig_train = offline.plot(plotly_fig_train, filename="plotly version of an mpl figure 2")

fig = go.Figure()

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = 6,
    delta = {'reference': 200},
    domain = {'x': [0.25, 1], 'y': [0.08, 0.25]},
    title = {'text': "Kraft in kN"},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [0, 7]},
        'threshold': {
            'line': {'color': "black", 'width': 2},
            'thickness': 0.75,
            'value': 5},
        'steps': [
            {'range': [0, 0.5], 'color': "lightgray"},
            {'range': [0.5, 5], 'color': "lightgreen"} ,
            {'range': [5, 7], 'color': "lightgray"}],
        'bar': {'color': "black"}}),
)

fig.add_trace(go.Indicator(
    mode = "number+gauge", value = 2,
    delta = {'reference': 200},
    domain = {'x': [0.25, 1], 'y': [0.4, 0.6]},
    title = {'text': "Leistung in kW"},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [0, 5]},
        'threshold': {
            'line': {'color': "black", 'width': 2},
            'thickness': 0.75,
            'value': 3},
        'steps': [
            {'range': [0, 3], 'color': "lightgreen"},
            {'range': [3, 5], 'color': "lightgray"}],
        'bar': {'color': "black"}})
)

fig.update_layout(height = 200 ,title="Prozessdaten an Station Spanen",  margin = {'t':0, 'b':0, 'l':0}, ) # margin = {'t':0, 'b':0, 'l':0},

# df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/c78bf172206ce24f77d6363a2d754b59/raw/c353e8ef842413cae56ae3920b8fd78468aa4cb2/usa-agricultural-exports-2011.csv')
#
# def generate_table(dataframe, max_rows=10):
#     return html.Table([
#         html.Thead(
#             html.Tr([html.Th(col) for col in dataframe.columns])
#         ),
#         html.Tbody([
#             html.Tr([
#                 html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
#             ]) for i in range(min(len(dataframe), max_rows))
#         ])
#     ])

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Datenanalyse Station Spanen'),

    html.Div(children='''
        Datenanalyse für die Station Spanen.
    '''),

    dcc.Graph(figure=fig),

    html.Div([
        html.Div([
            # html.H4('Klassifizierungsergebnis für Trainingsdaten'),
            dcc.Graph(id='matplotlib-graph1', figure=plotly_fig_train)
        ], className="six columns"),

        html.Div([
            # html.H4('Klassifizierungsergebnis für Testdaten'),
            dcc.Graph(id='matplotlib-graph2', figure=plotly_fig_test)
        ], className="six columns"),
    ], className="row"),

    html.Div([
        html.Div([
            #html.H4('Klassifizierungsergebnis für Trainingsdaten'),
            dcc.Graph(id='g1', figure={'data': [{'y': [1, 2, 3]}]})
        ], className="six columns"),

        html.Div([
            #html.H4('Klassifizierungsergebnis für Testdaten'),
            dcc.Graph(id='g2', figure={'data': [{'y': [1, 2, 3]}]})
        ], className="six columns"),
    ], className="row"),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization2'
            }
        }
    ),
    # html.H4(children='Table'),
    # generate_table(df)
])

if __name__ == '__main__':
    app.run_server(debug=True)