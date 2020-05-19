# from numpy import where
# from matplotlib import pyplot
# from sklearn.datasets import make_blobs
# import random
#
# seed = random.seed()
# # generate dataset
# X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=seed, cluster_std=1)
#
# # create scatter plot for samples from each class
# for class_value in range(3):
#]
#     # get row indexes for samples with this class
#     row_ix = where(y == class_value)
#
#     # create scatter of these samples
#     pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
#
# # show the plot
# pyplot.show()

from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from keras.layers import Dense
from keras.models import Sequential


# prepare train and test dataset
def prepare_data():
    # generate 2d classification dataset
    X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
                               n_clusters_per_class=2, weights=[0.99], flip_y=0, random_state=4)
    # split into train and test
    n_train = 5000
    trainX, testX = X[:n_train, :], X[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    return trainX, trainy, testX, testy


# define the neural network model
def define_model(n_input):
    # define model
    model = Sequential()
    # define first hidden layer and visible layer
    model.add(Dense(10, input_dim=n_input, activation='relu', kernel_initializer='he_uniform'))
    # define output layer
    model.add(Dense(1, activation='sigmoid'))
    # define loss and optimizer
    model.compile(loss='binary_crossentropy', optimizer='sgd')
    return model


# prepare dataset
trainX, trainy, testX, testy = prepare_data()

# get the model
n_input = trainX.shape[1]
model = define_model(n_input)

# fit model
weights = {0: 1, 1: 100}

history = model.fit(trainX, trainy, class_weight=weights, epochs=100, verbose=0)
# evaluate model
yhat = model.predict(testX)
score = roc_auc_score(testy, yhat)
print('ROC AUC: %.3f' % score)