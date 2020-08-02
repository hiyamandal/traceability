import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

import pickle

n_train = 3000
n_neighbors = 5

# script for generating data
def gen_data(n_train):

    # create data as multivariate normal distribution
    mean = [2.5, 1.5]
    cov = [[2.5, 1.5], [1.5, 1.5]]

    F, P = np.random.multivariate_normal(mean, cov, n_train).T

    idx = []
    for i in range(len(F)):
        if F[i] < 0 or P[i] < 0:
            idx.append(i)

    F = np.delete(F, idx, 0)
    P = np.delete(P, idx, 0)

    X = np.zeros((len(F), 2))
    for i in range(len(F)):
        X[i, 0] = F[i]
        X[i, 1] = P[i]

    return F, P, X

# assigning labels to the data (0=Gutteil, 1=Nachbearbeiten, 2=Ausschuss)
def labels(F, P):

    # initialize all labels with OK
    cat = np.zeros(len(F))
    for i in range(len(P)):
        alpha_0 = 1000

        # more likely to be rework if power bigger 3
        if P[i] > 3:
            alpha = (P[i] - 3) * 0.3
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]

        # more likely to be rework if force bigger 5
        if F[i] > 5:
            alpha = (F[i] - 5) * 0.3
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]

        # more likey to be scrap if power > and force > 5
        if P[i] > 3 and F[i] > 5:
            alpha = ((F[i] - 5) * 0.1 + (P[i] - 3) * 0.1) / 2
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0] + 1

        # more likely to be rework if force < 0.5
        if F[i] < 0.5:
            alpha = np.abs(F[i] - 0.5) * 0.3
            prior = np.random.dirichlet((alpha_0, alpha), 1)[0]
            cat[i] = np.random.multinomial(1, prior, size=1)[0][0]
    return cat

# make classification, create and save plots
def plotting(n_train, n_neighbors):

    # load data
    F, P, X = gen_data(n_train)
    y = labels(F, P)

    # shuffle and split training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                        random_state=0)

    from sklearn.multiclass import OneVsRestClassifier

    # we create an instance of Neighbours Classifier and fit the data.
    weights = 'uniform'
    classifier = OneVsRestClassifier(neighbors.KNeighborsClassifier(n_neighbors, weights=weights))
    clf = classifier.fit(X_train, y_train)

    # save model to binary with pickle to load it in plotly dash
    pickle.dump(clf, open("spanen_knn_model_"+str(n_train)+".sav", 'wb'))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    h = .02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    y_score = clf.predict(X_test)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    # Create color maps
    #     cmap_light = ListedColormap(['green', 'orange', 'red'])
    #     cmap_bold = ListedColormap(['darkgreen', 'darkorange', 'darkred'])
    #
    # # Plot Contour with training points
    # plt.figure()
    # plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold,
    #         edgecolor='k', s=20)
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    #
    # plt.xlabel('Prozesskraft in kN')
    # plt.ylabel('Leistung in kW')
    # plt.title('Klassifizierungsergebnis Station Spanen (Trainingsdaten)')
    # plt.savefig('train_spanen.png')
    # plt.show()
    #
    # # Plot Contour with test points
    # plt.figure()
    # plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold,
    #             edgecolor='k', s=20)
    # plt.scatter(2.9, 2.1, c='white', s=250, marker='x')
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    #
    # plt.xlabel('Prozesskraft in kN')
    # plt.ylabel('Leistung in kW')
    # plt.title('Klassifizierungsergebnis Station Spanen (Testdaten)')
    # plt.savefig('test_spanen.png')
    # plt.show()

    # do prediction on test data
    from sklearn.metrics import confusion_matrix
    y_pred = clf.predict(X_test)

    # plot a classification report
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, output_dict=True)
    print(report)
    print(report['0.0']['precision'])

    # convert contourplot data to list to save it in file
    xx_list = xx[0, :].tolist()
    yy_list = yy[:, 0].tolist()
    Z_transp = Z
    Z_list = Z_transp.tolist()

    # save data for contourplot in file
    file = open("spanen_knn_data_"+str(n_train)+".csv", "w+")
    file.write(str(y_test.tolist()) + "\n")
    file.write(str(X_test.tolist()) + "\n")
    file.write(str(xx_list) + "\n")
    file.write(str(yy_list) + "\n")
    file.write(str(Z_list) + "\n")
    file.write(str(y_train.tolist()) + "\n")
    file.write(str(X_train.tolist()) + "\n")
    file.write(str(report) + "\n")
    file.close()

    # Confusion Matrix on Test Data
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
    plt.savefig('spanen_confusion_absolute_'+str(n_train)+'.png')
    # plt.show()

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
    plt.savefig('spanen_confusion_normalised_'+str(n_train)+'.png')

# call plotting function
plotting(n_train, n_neighbors)

