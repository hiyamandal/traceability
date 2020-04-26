print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 20

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
cat = labels(F, P)

plt.grid(True)
cdict = {0: 'green', 1: 'orange', 2: 'red'}
labeldict = {0: 'Gutteil', 1: 'Nachbearbeiten', 2: 'Ausschuss'}
for g in np.unique(cat):
    ix = np.where(cat == g)
    plt.scatter(F[ix], P[ix], c = cdict[g], label = labeldict[g])

legend = plt.legend(loc="lower right", title="Legende")


plt.xlabel('Kzraft in kN')
plt.ylabel('Leistung in kW')
plt.title('Trainingsdaten Station Spanen')
plt.show()

# import some data to play with
# iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
#X = iris.data[:, :2]
y = cat


h = .02  # step size in the mesh

# Create color maps
# cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
# cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

cmap_light = ListedColormap(['green', 'orange', 'red'])
cmap_bold = ListedColormap(['darkgreen', 'darkorange', 'darkred'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

    plt.xlabel('Prozesskraft in kN')
    plt.ylabel('Leistung in kW')
    plt.title('Klassifizierungsergebnis Station Spanen')
    plt.show()

plt.show()