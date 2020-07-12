print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

import pickle

global_list = []

n_train = 1000
def gen_data(n_train):
    mean = [2.5, 1.5]
    cov = [[2.5, 1.5], [1.5, 1.5]]
    # cov = [[1.4, 1], [1.2, 0.5]]

    F, P = np.random.multivariate_normal(mean, cov, n_train).T

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
        alpha_0 = 1000

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

def plotting(n_train):
    F, P, X = gen_data(n_train)
    y = labels(F, P)

    # shuffle and split training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                        random_state=0)
    F_train = X_train[:, 0]
    P_train = X_train[:, 1]
    F_test = X_test[:, 0]
    P_test = X_test[:, 1]

    cdict = {0: 'green', 1: 'orange', 2: 'red'}
    labeldict = {0: 'Gutteil', 1: 'Nachbearbeiten', 2: 'Ausschuss'}

    # # plot training data
    # plt.grid(True)
    # for g in np.unique(y_train):
    #     ix = np.where(y_train == g)
    #     plt.scatter(F_train[ix], P_train[ix], c = cdict[g], label = labeldict[g])
    # legend = plt.legend(loc="lower right", title="Legende")
    # plt.xlabel('Kraft in kN')
    # plt.ylabel('Leistung in kW')
    # plt.title('Trainingsdaten Station Spanen')
    # plt.savefig('data_train_spanen.png')
    # plt.show()

    # # plot test data
    # plt.grid(True)
    # for g in np.unique(y_test):
    #     ix = np.where(y_test == g)
    #     plt.scatter(F_test[ix], P_test[ix], c = cdict[g], label = labeldict[g])
    # legend = plt.legend(loc="lower right", title="Legende")
    # plt.xlabel('Kraft in kN')
    # plt.ylabel('Leistung in kW')
    # plt.title('Testdaten Station Spanen')
    # plt.savefig('data_test_spanen.png')
    # plt.show()

    h = .02  # step size in the mesh, was 0.02

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

    # save model to binary with pickle
        pickle.dump(clf, open("../spanen_knn_model_"+str(n_train)+".sav", 'wb'))

        # # ROC
        # y_score = classifier.fit(X_train, y_train).decision_function(X_test)
        # # Compute ROC curve and ROC area for each class
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # n_classes = 2
        # from sklearn.metrics import roc_curve, auc
        # for i in range(n_classes):
        #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])
        #
        # # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        #
        # # First aggregate all false positive rates
        # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        #
        # # Then interpolate all ROC curves at this points
        # from scipy import interpolate as interp
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
        y_score = clf.predict(X_test)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)

        # Plot with training points
        fig_train = plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold,
                edgecolor='k', s=20)
        # for i in range(len(y_train)):
        #     if y_train[i] == 1 or y_train[i] == 2:
        #         plt.scatter(X_train[i, 0], X_train[i, 1], c=y_train[i], cmap=cmap_bold,
        #                     edgecolor='k', s=20)
        # for i in range(len(y_train)):
        #     if y_train[i] == 0:
        #         plt.scatter(X_train[i, 0], X_train[i, 1], c=y_train[i], cmap=cmap_bold,
        #                     edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        plt.xlabel('Prozesskraft in kN')
        plt.ylabel('Leistung in kW')
        plt.title('Klassifizierungsergebnis Station Spanen (Trainingsdaten)')
        plt.savefig('train_spanen.png')
        plt.show()

        # Plot with test points
        fig_test = plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.scatter(2.9, 2.1, c='white', s=250, marker='x')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        plt.xlabel('Prozesskraft in kN')
        plt.ylabel('Leistung in kW')
        plt.title('Klassifizierungsergebnis Station Spanen (Testdaten)')
        plt.savefig('test_spanen.png')
        plt.show()

        # convert contourplot data to list
        xx_list = xx[0, :].tolist()
        yy_list = yy[:, 0].tolist()
        Z_transp = Z
        Z_list = Z_transp.tolist()

        # save data for contourplot in file
        file = open("../"+"spanen_knn_data_"+str(n_train)+".csv", "w+")
        file.write(str(y_test.tolist()) + "\n")
        file.write(str(X_test.tolist()) + "\n")  # +","+str(xx_list)+","+str(yy_list)+","+str(Z_list))
        file.write(str(xx_list) + "\n")
        file.write(str(yy_list) + "\n")
        file.write(str(Z_list) + "\n")
        file.write(str(y_train.tolist()) + "\n")
        file.write(str(X_train.tolist()) + "\n")
        file.close()

        # from sklearn.metrics import precision_recall_curve
        # from sklearn.metrics import average_precision_score
        # precision = dict()
        # recall = dict()
        # average_precision = dict()
        # n_classes = 3
        # for i in range(n_classes):
        #     precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
        #                                                         y_score[:, i])
        #     average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
        # print('precision and recall', precision, recall)

        # Confusion Matrix on Training and on Test Data
        from sklearn.metrics import confusion_matrix
        y_pred = clf.predict(X_test)

        #confusion_train = confusion_matrix(y_train, Z)
        confusion_test = confusion_matrix(y_test, y_pred)
        print(confusion_test)

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
        plt.savefig('confusion_absolute.png')
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
        plt.savefig('confusion_normalised.png')
        plt.show()

    # plot a classification report
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred)
    print(report)

plotting(n_train)

