print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 5

num_data = 3000

mean = [1.5, 6, 32]
# cov = [[1, 1, 1   0], [1, 2, 10], [1, 1, 45]] # 1, 2, 45
cov = [[1, 0, 0], [0, 2, 0], [0, 0, 75]] # 6, 45

L, D, T = np.random.multivariate_normal(mean, cov, num_data).T

idx = []
for i in range(len(L)):
    if L[i] < 0 or D[i] < 0 or T[i] < 0:
        idx.append(i)

L = np.delete(L, idx, 0)
D = np.delete(D, idx, 0)
T = np.delete(T, idx, 0)

X = np.zeros((len(L), 3))
for i in range(len(L)):
    X[i, 0] = L[i]
    X[i, 1] = D[i]
    X[i, 2] = T[i]


from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random


fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(L, D, T)
ax.set_xlabel('Leistung in kW')
ax.set_ylabel('Druck in bar')
ax.set_zlabel('Temperatur in Grad Celsius')
ax.set_title('Trainingsdaten Station Lackieren')
plt.show()


def labels(L, D, T):
    cat = np.zeros(len(L))
    for i in range(len(L)):
        alpha_0 = 1
        factor = 0.1
        factor = 0.00001
        if L[i] > 3:
            #cat[i] = 1
            alpha = np.abs((L[i] - 3)) * factor
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            if L[i] > 5:
                cat[i] += 1
            #cat[i] = 1

        if D[i] > 8:
            # cat[i] = 1
            alpha = np.abs((D[i] - 8)) * factor
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            if D[i] > 9 or T[i] > 50 or T[i] < 13:
                cat[i] += 1
            #cat[i] = 1

        if D[i] < 4:
            alpha = np.abs((D[i] - 4)) * factor
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            # cat[i] = 1
            if D[i] < 3 or T[i] < 13 or T[i] > 50:
                cat[i] += 1

        if T[i] > 45:
            alpha = np.abs((T[i] - 45)) * factor
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            # cat[i] = 1
            if T[i] > 50 or D[i] > 9:
                cat[i] += 1

        if T[i] < 18:
            # cat[i] = 1
            alpha = np.abs((T[i] - 18)) * factor
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            if T[i] < 13 or D[i] < 3:
                cat[i] += 1
            #cat[i] = 1

        # if P[i] > 3 and F[i] > 5:
        #     alpha = ((F[i] - 5) * 0.1 + (P[i] - 3) * 0.1) / 2
        #     prior = np.DArandom.dirichlet((alpha_0, alpha), 1)[0]
        #     cat[i] = np.random.multinomial(1, prior, size=1)[0][0] + 1
        #     #cat[i] = 1
        #
        # if F[i] < 0.5:
        #     alpha = np.abs(F[i] - 0.5) * 0.3
        #     prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
        #     cat[i] = np.random.multinomial(1, prior, size=1)[0][0]

            #cat[i] = 0
        if D[i] > 9 or D[i] < 2:
            cat[i] = 2

        if D[i] > 5 and D[i] < 7 and T[i]>18 and T[i]<45 and L[i]>2.5 and L[i]<6:
            if T[i] > (7.71 * L[i] - 1.28 ):
                cat[i] = 0

    return cat
cat = labels(L, D, T)

fig1 = pyplot.figure()
ax = Axes3D(fig1)
plt.grid(True)
cdict = {0: 'green', 1: 'orange', 2: 'red'}
labeldict = {0: 'Gutteil', 1: 'Nachbearbeiten', 2: 'Ausschuss'}
for g in np.unique(cat):
    ix = np.where(cat == g)
    ax.scatter(L[ix], D[ix], T[ix], c = cdict[g], label = labeldict[g])

legend = plt.legend(loc="lower right", title="Legende")


ax.set_xlabel('Leistung in kW')
ax.set_ylabel('Druck in bar')
ax.set_zlabel('Temperatur in Grad Celsius')
ax.set_title('Trainingsdaten Station Lackieren')
plt.show()

# save data for scatter in file
file = open("../" + "lackieren_knn_data_" + str(num_data) + ".csv", "w+")
file.write(str(cat.tolist()) + "\n")
file.write(str(L.tolist()) + "\n")
file.write(str(D.tolist()) + "\n")
file.write(str(T.tolist()) + "\n")
file.close()

# fig2 = pyplot.figure()
# ax = Axes3D(fig2)
# plt.grid(True)
# cdict = {0: 'green', 1: 'orange', 2: 'red'}
# labeldict = {0: 'Gutteil', 1: 'Nachbearbeiten', 2: 'Ausschuss'}
# for g in np.unique(cat):
#     ix = np.where(cat == g)
#     for i in range(len(idx)):
#         if ix[0,i] > num_data / 10:
#             np.delete(ix, i)
#     ax.scatter(L[ix], D[ix], T[ix], c = cdict[g], label = labeldict[g])
#
# legend = plt.legend(loc="lower right", title="Legende")
# ax.set_xlabel('Leistung in kW')
# ax.set_ylabel('Druck in bar')
# ax.set_zlabel('Temperatur in Grad Celsius')
# ax.set_title('Trainingsdaten Station Lackieren')
# plt.show()
#
# # import some data to play with
# # iris = datasets.load_iris()
#
# # we only take the first two features. We could avoid this ugly
# # slicing by using a two-dim dataset
# #X = iris.data[:, :2]
y = cat


h = .04  # step size in the mesh
h2 = .2
# Create color maps
# cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
# cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

cmap_light = ListedColormap(['green', 'orange', 'red'])
cmap_bold = ListedColormap(['darkgreen', 'darkorange', 'darkred'])


weights = 'uniform'
# for weights in ['uniform', 'distance']:
# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(X, y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

# _x = np.linspace(x_min, x_max, 10)
# _y = np.linspace(y_min, y_max, 10)
# _z = np.linspace(z_min, z_max, 10)

xx_planeXY, yy_planeXY= np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
xx_planeXZ, zz_planeXZ = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(z_min, z_max, h2))
yy_planeYZ, zz_planeYZ = np.meshgrid(np.arange(y_min, y_max, h),
                     np.arange(z_min, z_max, h2))
# xx, yy, zz = np.meshgrid(_x, _y, _z)

# XY prediction
xx = xx_planeXY
yy = yy_planeXY
zz = np.ones(np.shape(xx_planeXY)) * 25

V = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

V = V.reshape(xx.shape)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

plt.pcolormesh(xx, yy, V, cmap=cmap_light)

# # Plot also the training points
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=cmap_bold,
#             edgecolor='k', s=20)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title("3-Class classification (k = %i, weights = '%s')"
          % (n_neighbors, weights))

plt.xlabel('Leistung in kN')
plt.ylabel('Druck in bar')
title = 'Klassifizierung Station Lackieren, Temperatur = 32 °C'
ax2.set_title(title, weight='bold', pad=20)
plt.savefig('train_lackieren_temp.png')
plt.show()


# XZ prediction
xx = xx_planeXZ
yy = np.ones(np.shape(xx_planeXZ)) * 6
zz = zz_planeXZ

V = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

V = V.reshape(xx.shape)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
plt.pcolormesh(xx, zz, V, cmap=cmap_light)

# # Plot also the training points
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=cmap_bold,
#             edgecolor='k', s=20)

plt.xlim(xx.min(), xx.max())
plt.ylim(zz.min(), zz.max())

plt.title("3-Class classification (k = %i, weights = '%s')"
          % (n_neighbors, weights))

plt.xlabel('Leistung in kW')
plt.ylabel('Temperatur in Grad Celsius')
title = 'Klassifizierung Station Lackieren, Druck = 6 bar'
ax3.set_title(title, weight='bold', pad=20)
plt.savefig('train_lackieren_pressure.png')
plt.show()

# YZ prediction
xx = np.ones(np.shape(yy_planeYZ)) * 1.5
yy = yy_planeYZ
zz = zz_planeYZ

V = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

V = V.reshape(xx.shape)

fig4 = plt.figure()
ax4 = fig2.add_subplot(111)
plt.pcolormesh(yy, zz, V, cmap=cmap_light)

# # Plot also the training points
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=cmap_bold,
#             edgecolor='k', s=20)

plt.xlim(yy.min(), yy.max())
plt.ylim(zz.min(), zz.max())

plt.title("3-Class classification (k = %i, weights = '%s')"
          % (n_neighbors, weights))

plt.xlabel('Druck in bar')
plt.ylabel('Temperatur in Grad Celsius')
title = 'Klassifizierung Station Lackieren, Leistung = 1.5kW'
ax4.set_title(title, weight='bold', pad=20)
plt.savefig('train_lackieren_power.png')
plt.show()



