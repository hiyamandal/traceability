import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import pickle

n_train = 3000
n_neighbors = 5

# script for generating data
def gen_data(n_train):

    mean = [1.5, 6, 32]
    # cov = [[1, 1, 1   0], [1, 2, 10], [1, 1, 45]] # 1, 2, 45
    cov = [[1, 0, 0], [0, 3, 0], [0, 0, 100]] #75

    L, D, T = np.random.multivariate_normal(mean, cov, n_train).T

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

    return L, D, T, X


# assigning labels to the data (0=Gutteil,1=Nachbearbeiten,2=Ausschuss)
def labels(L, D, T):
    p_blurring = [0.90, 0.10]
    cat = np.zeros(len(L))
    for i in range(len(L)):
        alpha_0 = 1
        factor = 0.1
        factor = 0.00001
        if L[i] > 3:
            if L[i] < 3.5:
                prior = np.asarray(np.array(p_blurring))
                cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            else:
                alpha = np.abs((L[i] - 3)) * factor
                prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
                cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
                if L[i] > 5:
                    cat[i] += 1
                #cat[i] = 1

        if D[i] > 8:
            if D[i] < 8.5:
                prior = np.asarray(np.array(p_blurring))
                cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            else:
                alpha = np.abs((D[i] - 8)) * factor
                prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
                cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
                if D[i] > 9 or T[i] > 50 or T[i] < 13:
                    cat[i] += 1
                #cat[i] = 1

        if D[i] < 4:
            if D[i] > 3.5:
                prior = np.asarray(np.array(p_blurring))
                cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            else:
                alpha = np.abs((D[i] - 4)) * factor
                prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
                cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
                # cat[i] = 1
                if D[i] < 3 or T[i] < 13 or T[i] > 50:
                    cat[i] += 1

        if T[i] > 45:
            if T[i] < 50:
                prior = np.asarray(np.array(p_blurring))
                cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            else:
                alpha = np.abs((T[i] - 45)) * factor
                prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
                cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
                # cat[i] = 1
                if T[i] > 50 or D[i] > 9:
                    cat[i] += 1

        if T[i] < 18:
            if T[i] > 13:
                prior = np.asarray(np.array(p_blurring))
                cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
            else:
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
        if L[i] > 3 and L[i] < 3.5:
            cat[i] = 1
        if D[i] > 9 or D[i] < 2 or L[i] > 3.5 :
            cat[i] = 2
            #cat[i] = 2

        # Gutteil
        if D[i] > 5 and D[i] < 7 and T[i]>18 and T[i]<45 and L[i]>2.5 and L[i]<6:
            if T[i] > (7.71 * L[i] - 1.28 ):
                cat[i] = 0

    return cat

# create and save plots
def plotting(n_train, n_neighbors):

    L, D, T, X = gen_data(n_train)

    cat = labels(L, D, T)

    # shuffle and split training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, cat, test_size=.3,
                                                        random_state=0)

    # plot training data with labels
    fig1 = pyplot.figure()
    ax = Axes3D(fig1)
    plt.grid(True)

    cdict = {0: 'green', 1: 'orange', 2: 'red'}
    labeldict = {0: 'Gutteil', 1: 'Nachbearbeiten', 2: 'Ausschuss'}

    for g in np.unique(cat):
        ix = np.where(cat == g)
        ax.scatter(L[ix], D[ix], T[ix], c = cdict[g], label = labeldict[g])

    ax.set_xlabel('Leistung in kW')
    ax.set_ylabel('Druck in bar')
    ax.set_zlabel('Temperatur in °C')
    ax.set_title('Trainingsdaten Station Lackieren')
    plt.show()

    # save data for scatter in file
    file = open("lackieren_knn_data_" + str(n_train) + ".csv", "w+")
    file.write(str(cat.tolist()) + "\n")
    file.write(str(L.tolist()) + "\n")
    file.write(str(D.tolist()) + "\n")
    file.write(str(T.tolist()) + "\n")
    file.close()

    h = .04  # step size in the mesh
    h2 = .2

    # Create color maps
    cmap_light = ListedColormap(['green', 'orange', 'red'])
    cmap_bold = ListedColormap(['darkgreen', 'darkorange', 'darkred'])

    weights = 'uniform' # for weights in ['uniform', 'distance']:

    # we create an instance of Neighbours Classifier and fit the data.
    classifier = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf = classifier.fit(X_train, y_train)

    # save model to binary with pickle
    pickle.dump(clf, open("lackieren_knn_model_" + str(n_train) + ".sav", 'wb'))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    xx_planeXY, yy_planeXY= np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    xx_planeXZ, zz_planeXZ = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(z_min, z_max, h2))
    yy_planeYZ, zz_planeYZ = np.meshgrid(np.arange(y_min, y_max, h),
                         np.arange(z_min, z_max, h2))

    # XY prediction
    xx = xx_planeXY
    yy = yy_planeXY
    zz = np.ones(np.shape(xx_planeXY)) * 25

    V = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    V = V.reshape(xx.shape)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    plt.pcolormesh(xx, yy, V, cmap=cmap_light)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

    plt.xlabel('Leistung in kN')
    plt.ylabel('Druck in bar')
    title = 'Klassifizierung Station Lackieren, Temperatur = 32 °C'
    ax2.set_title(title, weight='bold', pad=20)
    plt.savefig('lackieren_train_temp.png')
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

    plt.xlim(xx.min(), xx.max())
    plt.ylim(zz.min(), zz.max())

    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

    plt.xlabel('Leistung in kW')
    plt.ylabel('Temperatur in Grad Celsius')
    title = 'Klassifizierung Station Lackieren, Druck = 6 bar'
    ax3.set_title(title, weight='bold', pad=20)
    plt.savefig('lackieren_train_pressure.png')
    plt.show()

    # YZ prediction
    xx = np.ones(np.shape(yy_planeYZ)) * 1.5
    yy = yy_planeYZ
    zz = zz_planeYZ

    V = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    V = V.reshape(xx.shape)

    plt.figure()
    ax4 = fig2.add_subplot(111)
    plt.pcolormesh(yy, zz, V, cmap=cmap_light)

    plt.xlim(yy.min(), yy.max())
    plt.ylim(zz.min(), zz.max())

    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

    plt.xlabel('Druck in bar')
    plt.ylabel('Temperatur in Grad Celsius')
    title = 'Klassifizierung Station Lackieren, Leistung = 1.5kW'
    ax4.set_title(title, weight='bold', pad=20)
    plt.savefig('lackieren_train_power.png')
    plt.show()

    # Confusion Matrix on Test Data
    from sklearn.metrics import confusion_matrix
    y_pred = clf.predict(X_test)

    confusion_test = confusion_matrix(y_test, y_pred)

    # plot confusion matrix
    from sklearn.metrics import plot_confusion_matrix
    class_names = ['Gutteil', 'Nachbear.', 'Ausschuss']

    title = 'Konfusionsmatrix (absolut)'
    normalize = None
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize,
                                 values_format='.0f')
    disp.ax_.set_title(title, weight='bold', pad=20)
    disp.ax_.set_xlabel('Vorhergesagte Klasse', weight='bold')
    disp.ax_.set_ylabel('Wahre Klasse', weight='bold')

    print(title)
    print(disp.confusion_matrix)
    plt.savefig('confusion_spanen_absolute.png')
    plt.show()

    title = 'Konfusionsmatrix (normalisiert)'
    normalize = 'true'
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title, weight='bold', pad=20)
    disp.ax_.set_xlabel('Vorhergesagte Klasse', weight='bold')
    disp.ax_.set_ylabel('Wahre Klasse', weight='bold')

    print(title)
    print(disp.confusion_matrix)
    plt.savefig('confusion_spanen_normalised.png')
    plt.show()

    # plot a classification report
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred)
    print(report)

# call plotting function
plotting(n_train, n_neighbors)

