import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.patches as mpatches

mean = [2.5, 1.5]
cov = [[1.4, 1], [1.2, 0.5]]

F, P = np.random.multivariate_normal(mean, cov, 5000).T

idx = []
for i in range(len(F)):
    if F[i] < 0 or P[i] < 0:
        idx.append(i)

F = np.delete(F, idx, 0)
P = np.delete(P, idx, 0)
# X = np.vstack((F, P))
# X = X.reshape(len(F),2)
X = np.zeros((len(F), 2))
for i in range(len(F)):
    X[i, 0] = F[i]
    X[i, 1] = P[i]

def labels(F, P):
    cat = np.zeros(len(F))
    for i in range(len(P)):
        alpha_0 = 1

        if P[i] > 3:
            # alpha = (P[i] - 3) * 0.3
            # prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            # cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            cat[i] = 1
        if F[i] > 5:
            # alpha = (F[i] - 5) * 0.3
            # prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            # cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            cat[i] = 1
        if P[i] > 3 and F[i] > 5:
            # alpha = ((F[i] - 5) * 0.1 + (P[i] - 3) * 0.1) / 2
            # prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            # cat[i] = np.random.multinomial(1, prior, size=1)[0][0] #+ 1
            cat[i] = 1
        if F[i] < 0.5:
            # alpha = np.abs(F[i] - 0.5) * 0.3
            # prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            # cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            cat[i] = 0
    return cat
cat = labels(F, P)

colors = []
for i in range(len(P)):
    if cat[i] == 0:
        colors.append('blue')
    if cat[i] == 1:
        colors.append('orange')
    if cat[i] == 2:
        colors.append('red')

print(cat)
counts = np.unique(cat, return_counts=True)
dummy_y = np_utils.to_categorical(cat)
print(dummy_y)

plt.grid(True)
cdict = {0: 'blue', 1: 'orange', 2: 'red'}
labeldict = {0: 'Gutteil', 1: 'Nachbearbeiten', 2: 'Ausschuss'}
for g in np.unique(cat):
    ix = np.where(cat == g)
    plt.scatter(F[ix], P[ix], c = cdict[g], label = labeldict[g])

legend = plt.legend(loc="lower right", title="Klassifizierung")


plt.xlabel('Prozesskraft in kN')
plt.ylabel('Leistung in kW')
plt.title('Prozessdaten Station Spanen')
plt.show()

print(counts)

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()
history = model.fit(X, dummy_y, verbose=0, nb_epoch=100)#, #shuffle =true)
plt.plot(history.history['loss'])
plt.show()

#estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
# kfold = KFold(n_splits=2, shuffle=True)
# print(X)
# print(dummy_y)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


def plot_decision_boundary(X, y, model, steps=1000, cmap='Paired'):
    cmap = plt.get_cmap(cmap)

    # Define region of interest by data limits
    xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
    # xmin, xmax = F.min() - 1, F.max() + 1
    # ymin, ymax = P.min() - 1, P.max() + 1

    steps = 1000
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels = model.predict(np.c_[xx.ravel(), yy.ravel()])
    labels = np.amax(labels, axis=1)
    print(labels)

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    # Get predicted labels on training data and plot
    train_labels = model.predict(X)
    train_labels = np.amax(train_labels, axis=1)

    # ax.scatter(X[:,0], X[:,1], c=train_labels, cmap=cmap, lw=0)
    ax.scatter(F, P, c=train_labels, cmap=cmap, lw=0)
    return fig, ax

baseline_model = baseline_model()
plot_decision_boundary(X, cat, baseline_model, cmap = 'RdBu')
plt.show()