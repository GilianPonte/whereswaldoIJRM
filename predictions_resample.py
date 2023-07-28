# -*- coding: utf-8 -*-
"""predictions_resample.py
"""

seed=1
import os
os.environ['PYTHONHASHSEED'] = str(seed)
# For working on GPUs from "TensorFlow Determinism"
os.environ["TF_DETERMINISTIC_OPS"] = str(seed)
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)
import tensorflow as tf
tf.random.set_seed(seed)
from google.colab import drive
from google.colab import files
import tensorflow as tf
import pandas as pd
import statistics
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import average_precision_score, log_loss, roc_curve, brier_score_loss, auc, accuracy_score, f1_score, roc_auc_score, recall_score, precision_recall_curve
from scipy import stats
import math
from google.colab import drive
import pandas as pd
import numpy as np
import statistics
from sklearn import metrics
import math
import matplotlib.pyplot as plt
from __future__ import print_function, division
from keras.datasets import mnist
from tensorflow.keras.optimizers import RMSprop
from functools import partial
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import argparse
import random
import seaborn as sns
import os
import numpy as np
import keras
from tensorflow.keras import backend as K
from functools import partial
from google.colab import drive
from google.colab import files
import time
from __future__ import print_function, division
import os
from sklearn.linear_model import LinearRegression
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LSTM
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pandas as pd
import io
from keras.models import load_model
import time
from scipy.stats import pearsonr
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras import losses
import keras.backend as K
import random
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import average_precision_score, log_loss, roc_curve, brier_score_loss, auc, accuracy_score, f1_score, roc_auc_score, recall_score, precision_recall_curve

drive.mount('/content/drive')

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('train_pred.csv', sep = ',', na_values=['(NA)']).fillna(0)
data = np.array(data)
data = data.reshape(93524,9)
y_train = data[:,0]
X_train = data[:,1:9]

data_medium = pd.read_csv('train_pred_medium.csv', sep = ',', na_values=['(NA)']).fillna(0)
data_medium = np.array(data_medium)
data_medium = data_medium.reshape(46762,9)
y_train_medium = data_medium[:,0]
X_train_medium = data_medium[:,1:9]


data_small = pd.read_csv('train_pred_small.csv', sep = ',', na_values=['(NA)']).fillna(0)
data_small = np.array(data_small)
data_small = data_small.reshape(1039,9)
y_train_small = data_small[:,0]
X_train_small = data_small[:,1:9]

test = pd.read_csv('test_pred.csv', sep = ',', na_values=['(NA)']).fillna(0).sample(frac= 1)
test = np.array(test)
test = test.reshape(10392,9)
y_test = test[:,0]
X_test = test[:,1:9]

clf = RandomForestClassifier(random_state=0, n_jobs = -1)

param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'max_depth': [1,5,None]
}

#clf_grid = GridSearchCV(clf, param_grid, scoring = "f1", cv=2, n_jobs = -1)
#clf_grid.fit(X_train, y_train)

#clf_grid_medium = GridSearchCV(clf, param_grid, scoring = "f1", cv=2, n_jobs = -1)
#clf_grid_medium.fit(X_train_medium, y_train_medium)

#clf_grid_small = GridSearchCV(clf, param_grid, scoring = "f1", cv=2, n_jobs = -1)
#clf_grid_small.fit(X_train_small, y_train_small)

#clf_grid.best_params_

#clf_grid_medium.best_params_

#clf_grid_small.best_params_

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

clf_test = RandomForestClassifier(random_state=0, max_depth = 1, n_estimators = 50, n_jobs = -1)
clf_test.fit(X_train, y_train)

clf_test_medium = RandomForestClassifier(random_state=0, max_depth = 1, n_estimators = 50, max_features = 'log2', n_jobs = -1)
clf_test_medium.fit(X_train_medium, y_train_medium)

clf_test_small = RandomForestClassifier(random_state=0, class_weight = 'balanced', max_depth = 1, n_estimators = 50, n_jobs = -1)
clf_test_small.fit(X_train_small, y_train_small)


test = pd.read_csv('test_pred.csv', sep = ',', na_values=['(NA)']).fillna(0)
y_test = test.iloc[:,0]
X_test = test.iloc[:,1:9]
print(X_test)

predictions = clf_test.predict_proba(X_test)
y_test = test.iloc[:,0]
X_test = pd.DataFrame(test.iloc[:,1:9])
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions[:,1]}, columns=['truth', 'pred'])
dataset = pd.concat([X_test,dataset], axis = 1)
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])/1000
precision, recall, thresholds = precision_recall_curve(y_test, predictions[:,1])
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions[:,1])
aps_real = average_precision_score(y_test, predictions[:,1])
print(max(f1_real))
print(percentage_real*1000)
print(top.groupby("Nationality").sum())

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

predictions = clf_test_medium.predict_proba(X_test)
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions[:,1]}, columns=['truth', 'pred'])
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])/1000

precision, recall, thresholds = precision_recall_curve(y_test, predictions[:,1])
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions[:,1])
aps_real = average_precision_score(y_test, predictions[:,1])
print(max(f1_real))
print(percentage_real*1000)

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

predictions = clf_test_small.predict_proba(X_test)
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions[:,1]}, columns=['truth', 'pred'])
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])/1000

precision, recall, thresholds = precision_recall_curve(y_test, predictions[:,1])
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions[:,1])
aps_real = average_precision_score(y_test, predictions[:,1])
print(max(f1_real))
print(log_loss_real)
print(percentage_real*1000)
print(aps_real)

predictions_real = clf_test.predict(X_test)
np.savetxt("predictions_real.csv", predictions_real, delimiter=",")

predictions_real_proba = clf_test.predict_proba(X_test)
np.savetxt("predictions_real_proba.csv", predictions_real_proba, delimiter=",")

predictions_real_small_proba = clf_test_small.predict_proba(X_test)
np.savetxt("predictions_real_small_proba.csv", predictions_real_small_proba, delimiter=",")

"""## GAN data"""

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_small.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 10000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_GAN_small = data[:,0]
X_train_GAN_small = data[:,1:9]

clf_GAN_small = RandomForestClassifier(random_state=0, n_jobs = -1)

tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_GAN_tuning_small = tuning[:,0]
X_train_GAN_tuning_small = tuning[:,1:9]
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
#clf_grid_GAN_small = GridSearchCV(clf_GAN_small, param_grid, scoring = "f1", cv=2, n_jobs = -1)
#clf_grid_GAN_small.fit(X_train_GAN_tuning_small, np.around(y_train_GAN_tuning_small))

#clf_grid_GAN_small.best_params_

clf_GAN_small = RandomForestClassifier(random_state=0, class_weight = 'balanced', max_depth = 1, n_estimators = 50, n_jobs = -1)

clf_GAN_small.fit(X_train_GAN_small, np.around(y_train_GAN_small))

predictions = clf_GAN_small.predict_proba(X_test)[:,1]
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions}, columns=['truth', 'pred'])
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])/1000

precision, recall, thresholds = precision_recall_curve(y_test, predictions)
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions)
aps_real = average_precision_score(y_test, predictions)
print(max(f1_real))
print(percentage_real*1000)

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_medium.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 1000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_GAN_medium = data[:,0]
X_train_GAN_medium = data[:,1:9]

clf_GAN_medium = RandomForestClassifier(random_state=0, n_jobs = -1)

tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_GAN_tuning_medium = tuning[:,0]
X_train_GAN_tuning_medium = tuning[:,1:9]
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
clf_grid_GAN_medium = GridSearchCV(clf_GAN_medium, param_grid, scoring = "f1", cv=2, n_jobs = -1)
clf_grid_GAN_medium.fit(X_train_GAN_tuning_medium, np.around(y_train_GAN_tuning_medium))

clf_grid_GAN_medium.best_params_

clf_GAN_medium = RandomForestClassifier(random_state=0, max_depth = 1, n_estimators = 50, max_features = 'log2', n_jobs = -1)
clf_GAN_medium = clf_grid_GAN_medium.best_estimator_
clf_GAN_medium.fit(X_train_GAN_medium, np.around(y_train_GAN_medium))

predictions = clf_GAN_medium.predict_proba(X_test)[:,1]
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions}, columns=['truth', 'pred'])
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])/1000

precision, recall, thresholds = precision_recall_curve(y_test, predictions)
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions)
aps_real = average_precision_score(y_test, predictions)
print(max(f1_real))
print(log_loss_real)
print(percentage_real)
print(aps_real)

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 10000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_GAN = data[:,0]
X_train_GAN = data[:,1:9]

clf_GAN = RandomForestClassifier(random_state=0, n_jobs = -1)

tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_GAN_tuning = tuning[:,0]
X_train_GAN_tuning = tuning[:,1:9]
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
#clf_grid_GAN = GridSearchCV(clf_GAN, param_grid, scoring = "f1", cv=2, n_jobs = -1)#
#clf_grid_GAN.fit(X_train_GAN_tuning, np.around(pd.DataFrame(y_test).sample(1000)))
clf_GAN = RandomForestClassifier(random_state=0, class_weight = 'balanced_subsample', criterion = 'log_loss', n_estimators = 1, n_jobs = -1)

clf_GAN.fit(X_train_GAN, np.around(y_train_GAN))

test = pd.read_csv('test_pred.csv', sep = ',', na_values=['(NA)']).fillna(0)
y_test = test.iloc[:,0]
X_test = pd.DataFrame(test.iloc[:,1:9])
predictions = clf_GAN.predict_proba(X_test)

dataset = pd.DataFrame({'truth': y_test, 'pred': predictions[:,1]}, columns=['truth', 'pred'])
dataset = pd.concat([X_test,dataset], axis = 1)
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])

print(percentage_real)
print(top.groupby("Nationality").sum())
precision, recall, thresholds = precision_recall_curve(y_test, predictions[:,1])
f1_real = (2*recall*precision)/(precision + recall)
print(max(f1_real))

predictions_GAN = clf_GAN.predict(X_test)
np.savetxt("predictions_GAN.csv", predictions_GAN, delimiter=",")

predictions_GAN_small = clf_GAN_small.predict(X_test)
np.savetxt("predictions_GAN_small.csv", predictions_GAN_small, delimiter=",")

predictions_GAN_proba = clf_GAN.predict_proba(X_test)
np.savetxt("predictions_GAN_proba.csv", predictions_GAN_proba, delimiter=",")

predictions_GAN_small_proba = clf_GAN_small.predict_proba(X_test)
np.savetxt("predictions_GAN_small_proba.csv", predictions_GAN_small_proba, delimiter=",")

"""## epsilon = 13"""

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_e_13_small.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 1000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_13_small = data[:,0]
X_train_13_small = data[:,1:9]
clf_13_small = RandomForestClassifier(random_state=0, n_jobs = -1)

tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_13_tuning_small = tuning[:,0]
X_train_13_tuning_small = tuning[:,1:9]
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500,1000],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
#clf_grid_13_small = GridSearchCV(clf_13_small, param_grid, scoring = "f1", cv=2, n_jobs = -1)
#clf_grid_13_small.fit(X_train_13_tuning_small, np.around(y_train_13_tuning_small))
#clf_grid_13_small.best_params_
clf_13_small = RandomForestClassifier(random_state=0, n_jobs = -1, criterion = "log_loss", n_estimators= 200, max_features = 'log2')
clf_13_small.fit(X_train_13_small, np.around(y_train_13_small))

predictions = clf_13_small.predict_proba(X_test)[:,1]
print(predictions)
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions}, columns=['truth', 'pred'])
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])

precision, recall, thresholds = precision_recall_curve(y_test, predictions)
f1_real = (2*recall*precision)/(precision + recall)
print(max(f1_real))
print(percentage_real)

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_medium_e_13.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 10000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_13_medium = data[:,0]
X_train_13_medium = data[:,1:9]
clf_13_medium = RandomForestClassifier(random_state=0, n_jobs = -1)

tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_13_tuning_medium = tuning[:,0]
X_train_13_tuning_medium = tuning[:,1:9]
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500,1000],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
clf_grid_13_medium = GridSearchCV(clf_13_medium, param_grid, scoring = "f1", cv=2, n_jobs = -1)
clf_grid_13_medium.fit(X_train_13_tuning_medium, np.around(y_train_13_tuning_medium))

clf_grid_13_medium.best_params_

clf_13_medium = RandomForestClassifier(random_state=0, n_jobs = -1, class_weight = 'balanced', criterion = "gini", n_estimators= 100, max_features = 'log2')
clf_13_medium.fit(X_train_13_medium, np.around(y_train_13_medium))

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

predictions = clf_13_medium.predict_proba(X_test)[:,1]
print(predictions)
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions}, columns=['truth', 'pred'])
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])/1000

precision, recall, thresholds = precision_recall_curve(y_test, predictions)
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions)
aps_real = average_precision_score(y_test, predictions)
print(max(f1_real))
print(log_loss_real)
print(percentage_real)
print(aps_real)

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_e_13.csv', sep = ',', na_values=['(NA)']).fillna(0) #3
tuning = data.sample(n = 10000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_13 = data[:,0]
X_train_13 = data[:,1:9]
clf_13 = RandomForestClassifier(random_state=0, n_jobs = -1)

tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_13_tuning = tuning[:,0]
X_train_13_tuning = tuning[:,1:9]
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
clf_grid_13 = GridSearchCV(clf_13, param_grid, scoring = "f1", cv=2, n_jobs = -1)
clf_grid_13.fit(X_train_13_tuning, np.around(y_train_13_tuning))

clf_grid_13.best_params_

clf_13 =RandomForestClassifier(random_state=0, n_estimators = 500, criterion = 'log_loss', n_jobs = -1)
clf_13.fit(X_train_13, np.around(y_train_13))

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

predictions = clf_13.predict_proba(X_test)
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions[:,1]}, columns=['truth', 'pred'])
dataset = pd.concat([X_test,dataset], axis = 1)
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])

precision, recall, thresholds = precision_recall_curve(y_test, predictions[:,1])
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions[:,1])
aps_real = average_precision_score(y_test, predictions[:,1])
print(max(f1_real))
print(log_loss_real)
print(percentage_real)
print(aps_real)
print(top.groupby("Nationality").sum())

predictions_13 = clf_13.predict(X_test)
np.savetxt("predictions_13.csv", predictions_13, delimiter=",")

predictions_13_small = clf_13_small.predict(X_test)
np.savetxt("predictions_13_small.csv", predictions_13_small, delimiter=",")

predictions_13_proba = clf_13.predict_proba(X_test)
np.savetxt("predictions_13_proba.csv", predictions_13_proba, delimiter=",")

predictions_13_small_proba = clf_13_small.predict_proba(X_test)
np.savetxt("predictions_13_small_proba.csv", predictions_13_small_proba, delimiter=",")

"""## epsilon = 3"""

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_e_3_small.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 10000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_3_small = data[:,0]
X_train_3_small = data[:,1:9]

clf_3_small = RandomForestClassifier(random_state=0, n_jobs = -1)

tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_3_tuning_small = tuning[:,0]
X_train_3_tuning_small = tuning[:,1:9]
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
clf_grid_3_small = GridSearchCV(clf_3_small, param_grid, scoring = "f1", cv=2, n_jobs = -1)
clf_grid_3_small.fit(X_train_3_tuning_small, np.around(y_train_3_tuning_small))

clf_grid_3_small.best_params_

clf_3_small = RandomForestClassifier(random_state=0, n_jobs = -1, criterion = "gini", n_estimators= 50, max_features = 'log2', max_depth = 1)
clf_3_small.fit(X_train_3_small, np.around(y_train_3_small))

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

test = pd.read_csv('test_pred.csv', sep = ',', na_values=['(NA)']).fillna(0).sample(frac= 1)
test = np.array(test)
test = test.reshape(10392,9)
y_test = test[:,0]
X_test = test[:,1:9]

predictions = clf_3_small.predict_proba(X_test)
predictions = 1 - predictions[:,0]
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions}, columns=['truth', 'pred'])
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])

precision, recall, thresholds = precision_recall_curve(y_test, predictions)
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions)
aps_real = average_precision_score(y_test, predictions)
print(max(f1_real))
print(log_loss_real)
print(percentage_real)
print(aps_real)

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_medium_e_3.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 10000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_3_medium = data[:,0]
X_train_3_medium = data[:,1:9]

clf_3_medium = RandomForestClassifier(random_state=0, n_jobs = -1)

tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_3_tuning_medium = tuning[:,0]
X_train_3_tuning_medium = tuning[:,1:9]
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
clf_grid_3_medium = GridSearchCV(clf_3_medium, param_grid, scoring = "f1", cv=2, n_jobs = -1)
clf_grid_3_medium.fit(X_train_3_tuning_medium, np.around(y_train_3_tuning_medium))

clf_grid_3_medium.best_params_

clf_3_medium = RandomForestClassifier(random_state=0, n_jobs = -1, class_weight = 'balanced_subsample', criterion = "log_loss", n_estimators= 500)
clf_3_medium.fit(X_train_3_medium, np.around(y_train_3_medium))

predictions = clf_3_medium.predict_proba(X_test)
predictions = 1 - predictions[:,0]
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions}, columns=['truth', 'pred'])
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])/1000

precision, recall, thresholds = precision_recall_curve(y_test, predictions)
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions)
aps_real = average_precision_score(y_test, predictions)
print(max(f1_real))
print(log_loss_real)
print(percentage_real)
print(aps_real)

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_e_3.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 10000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_3 = data[:,0]
X_train_3 = data[:,1:9]

clf_3 = RandomForestClassifier(random_state=0, n_jobs = -1)

tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_3_tuning = tuning[:,0]
X_train_3_tuning = tuning[:,1:9]
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
clf_grid_3 = GridSearchCV(clf_3, param_grid, scoring = "f1", cv=2, n_jobs = -1)
clf_grid_3.fit(X_train_3_tuning, np.around(y_train_3_tuning))

clf_3 = RandomForestClassifier(random_state=0, n_jobs = -1, class_weight = None,
 criterion = 'gini', max_depth = 1, max_features = 'log2', n_estimators = 10)
clf_3.fit(X_train_3, np.around(y_train_3))

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

test = pd.read_csv('test_pred.csv', sep = ',', na_values=['(NA)']).fillna(0).sample(frac= 1)
y_test = test.iloc[:,0]
X_test = test.iloc[:,1:9]

predictions = clf_3.predict_proba(X_test)
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions[:,1]}, columns=['truth', 'pred'])
dataset = pd.concat([X_test,dataset], axis = 1)
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])

precision, recall, thresholds = precision_recall_curve(y_test, predictions[:,1])
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions[:,1])
aps_real = average_precision_score(y_test, predictions[:,1])
print(max(f1_real))
print(log_loss_real)
print(percentage_real)
print(aps_real)
print(top.groupby("Nationality").sum())

predictions_3 = clf_3.predict(X_test)
np.savetxt("predictions_3.csv", predictions_3, delimiter=",")

predictions_3_small = clf_3_small.predict(X_test)
np.savetxt("predictions_3_small.csv", predictions_3_small, delimiter=",")

predictions_3_proba = clf_3.predict_proba(X_test)
np.savetxt("predictions_3_proba.csv", predictions_3_proba, delimiter=",")

predictions_3_small_proba = clf_3_small.predict_proba(X_test)
np.savetxt("predictions_3_small_proba.csv", predictions_3_small_proba, delimiter=",")

"""## epsilon = 1"""

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_e_1_small.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 1000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_1_small = data[:,0]
X_train_1_small = data[:,1:9]

clf_1_small = RandomForestClassifier(random_state=0, n_jobs = -1)
tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_1_small_tuning = tuning[:,0]
X_train_1_small_tuning = tuning[:,1:9]
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
clf_grid_1_small = GridSearchCV(clf_1_small, param_grid, scoring = "f1", cv=2, n_jobs = -1)
clf_grid_1_small.fit(X_train_1_small_tuning, np.around(y_train_1_small_tuning))

clf_grid_1_small.best_params_

clf_1_small = RandomForestClassifier(random_state=0, n_jobs = -1, criterion = "gini", n_estimators= 50, max_features = 'sqrt', max_depth = 1)
clf_1_small.fit(X_train_1_small, np.around(y_train_1_small))

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

test = pd.read_csv('test_pred.csv', sep = ',', na_values=['(NA)']).fillna(0).sample(frac= 1)
y_test = test.iloc[:,0]
X_test = test.iloc[:,1:9]

predictions = clf_1_small.predict_proba(X_test)
print(predictions)
predictions = 1 - predictions[:,0]
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions}, columns=['truth', 'pred'])
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])/1000

precision, recall, thresholds = precision_recall_curve(y_test, predictions)
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions)
aps_real = average_precision_score(y_test, predictions)
print(max(recall[1:len(recall)]))
print(max(recall[1:len(precision)]))
print(max(f1_real))
print(log_loss_real)
print(percentage_real)
print(aps_real)

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_medium_e_1.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 10000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_1_medium = data[:,0]
X_train_1_medium = data[:,1:9]

clf_1_medium = RandomForestClassifier(random_state=0, n_jobs = -1)

tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_1_tuning_medium = tuning[:,0]
X_train_1_tuning_medium = tuning[:,1:9]
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
clf_grid_1_medium = GridSearchCV(clf_1_medium, param_grid, scoring = "f1", cv=2, n_jobs = -1)
clf_grid_1_medium.fit(X_train_1_tuning_medium, np.around(y_train_1_tuning_medium))

clf_grid_1_medium.best_params_

clf_1_medium = RandomForestClassifier(random_state=0, n_jobs = -1, class_weight = 'balanced', criterion = "log_loss", n_estimators= 200, max_features = 'log2')
clf_1_medium.fit(X_train_1_medium, np.around(y_train_1_medium))

predictions = clf_1_medium.predict_proba(X_test)
predictions = 1 - predictions[:,0]
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions}, columns=['truth', 'pred'])
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])/1000

precision, recall, thresholds = precision_recall_curve(y_test, predictions)
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions)
aps_real = average_precision_score(y_test, predictions)
print(max(f1_real))
print(log_loss_real)
print(percentage_real)
print(aps_real)

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_e_1.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 10000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_1 = data[:,0]
X_train_1 = data[:,1:9]

clf_1 = RandomForestClassifier(random_state=0, n_jobs = -1)
tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_1_tuning = tuning[:,0]
X_train_1_tuning = tuning[:,1:9]
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
clf_grid_1 = GridSearchCV(clf_1, param_grid, scoring = "f1", cv=2, n_jobs = -1)
clf_grid_1.fit(X_train_1_tuning, np.around(y_train_1_tuning))

clf_grid_1.best_params_

clf_1 = RandomForestClassifier(random_state=0, n_jobs = -1, class_weight = 'balanced', criterion = "log_loss", n_estimators= 200, max_features = 'log2')
clf_1.fit(X_train_1, np.around(y_train_1))

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

test = pd.read_csv('test_pred.csv', sep = ',', na_values=['(NA)']).fillna(0).sample(frac= 1)
y_test = test.iloc[:,0]
X_test = test.iloc[:,1:9]

predictions = clf_1.predict_proba(X_test)
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions[:,1]}, columns=['truth', 'pred'])
dataset = pd.concat([X_test,dataset], axis = 1)
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])

precision, recall, thresholds = precision_recall_curve(y_test, predictions[:,1])
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions[:,1])
aps_real = average_precision_score(y_test, predictions[:,1])
print(max(f1_real))
print(log_loss_real)
print(percentage_real)
print(aps_real)
print(top.groupby("Nationality").sum())

predictions_1 = clf_1.predict(X_test)
np.savetxt("predictions_1.csv", predictions_1, delimiter=",")

predictions_1_small = clf_1_small.predict(X_test)
np.savetxt("predictions_1_small.csv", predictions_1_small, delimiter=",")

predictions_1_proba = clf_1.predict_proba(X_test)
np.savetxt("predictions_1_proba.csv", predictions_1_proba, delimiter=",")

predictions_1_small_proba = clf_1_small.predict_proba(X_test)
np.savetxt("predictions_1_small_proba.csv", predictions_1_small_proba, delimiter=",")

"""## epsilon = 0.5"""

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_e_05_small.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 10000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_05_small = data[:,0]
X_train_05_small = data[:,1:9]

clf_05_small = RandomForestClassifier(random_state=0, n_jobs = -1)
tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_05_small_tuning = tuning[:,0]
X_train_05_small_tuning = tuning[:,1:9]
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
clf_grid_05_small = GridSearchCV(clf_05_small, param_grid, scoring = "f1", cv=2, n_jobs = -1)
clf_grid_05_small.fit(X_train_05_small_tuning, np.around(y_train_05_small_tuning))

clf_grid_05_small.best_params_

clf_05_small = RandomForestClassifier(random_state=0, n_jobs = -1, criterion = "gini", n_estimators= 50, max_features = 'sqrt', max_depth = 1)
clf_05_small.fit(X_train_05_small, np.around(y_train_05_small))

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

test = pd.read_csv('test_pred.csv', sep = ',', na_values=['(NA)']).fillna(0).sample(frac= 1)
y_test = test.iloc[:,0]
X_test = test.iloc[:,1:9]

predictions = clf_05_small.predict_proba(X_test)
print(predictions)
predictions = 1 - predictions[:,0]
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions}, columns=['truth', 'pred'])
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])

precision, recall, thresholds = precision_recall_curve(y_test, predictions)
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions)
aps_real = average_precision_score(y_test, predictions)
print(max(recall[1:len(recall)]))
print(max(f1_real))
print(log_loss_real)
print(percentage_real)
print(aps_real)

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_e_05.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 10000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_05 = data[:,0]
X_train_05 = data[:,1:9]

clf_05 = RandomForestClassifier(random_state=0, n_jobs = -1)

tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_05_tuning = tuning[:,0]
X_train_05_tuning = tuning[:,1:9]
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
clf_grid_05 = GridSearchCV(clf_05, param_grid, scoring = "f1", cv=2, n_jobs = -1)
clf_grid_05.fit(X_train_05_tuning, np.around(y_train_05_tuning))

clf_grid_05.best_params_

clf_05 = RandomForestClassifier(random_state=0, n_jobs = -1, class_weight = 'balanced_subsample', criterion = "log_loss", n_estimators= 500, max_features = 'sqrt')
clf_05.fit(X_train_05, np.around(y_train_05))

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

test = pd.read_csv('test_pred.csv', sep = ',', na_values=['(NA)']).fillna(0).sample(frac= 1)
y_test = test.iloc[:,0]
X_test = test.iloc[:,1:9]

predictions = clf_05.predict_proba(X_test)
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions[:,1]}, columns=['truth', 'pred'])
dataset = pd.concat([X_test,dataset], axis = 1)
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])

precision, recall, thresholds = precision_recall_curve(y_test, predictions[:,1])
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions[:,1])
aps_real = average_precision_score(y_test, predictions[:,1])
print(max(f1_real))
print(log_loss_real)
print(percentage_real)
print(aps_real)
print(top.groupby("Nationality").sum())

predictions_05 = clf_05.predict(X_test)
np.savetxt("predictions_05.csv", predictions_05, delimiter=",")

predictions_05_small = clf_05_small.predict(X_test)
np.savetxt("predictions_05_small.csv", predictions_05_small, delimiter=",")

predictions_05_proba = clf_05.predict_proba(X_test)
np.savetxt("predictions_05_proba.csv", predictions_05_proba, delimiter=",")

predictions_05_small_proba = clf_05_small.predict_proba(X_test)
np.savetxt("predictions_05_small_proba.csv", predictions_05_small_proba, delimiter=",")

"""## epsilon = 0.05"""

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_e_005_small.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 10000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_005_small = data[:,0]
X_train_005_small = data[:,1:9]

tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_005_small_tuning = tuning[:,0]
X_train_005_small_tuning = tuning[:,1:9]
clf_005_small = RandomForestClassifier(random_state=0, n_jobs = -1)
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
clf_grid_005_small = GridSearchCV(clf_005_small, param_grid, scoring = "f1", cv=2, n_jobs = -1)
clf_grid_005_small.fit(X_train_005_small_tuning, np.around(y_train_005_small_tuning))

clf_grid_005_small.best_params_

clf_005_small = RandomForestClassifier(random_state=0, n_jobs = -1, criterion = "gini", n_estimators= 50, max_features = 'sqrt', max_depth = 1)
clf_005_small.fit(X_train_005_small, np.around(y_train_005_small))

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

test = pd.read_csv('test_pred.csv', sep = ',', na_values=['(NA)']).fillna(0).sample(frac= 1)
y_test = test.iloc[:,0]
X_test = test.iloc[:,1:9]

predictions = clf_005_small.predict_proba(X_test)
print(predictions)
predictions = 1 - predictions[:,0]
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions}, columns=['truth', 'pred'])
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])

precision, recall, thresholds = precision_recall_curve(y_test, predictions)
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions)
aps_real = average_precision_score(y_test, predictions)
print(max(recall[1:len(recall)]))
print(max(f1_real))
print(log_loss_real)
print(percentage_real)
print(aps_real)

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_e_005.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 1000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_005 = data[:,0]
X_train_005 = data[:,1:9]

tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_005_tuning = tuning[:,0]
X_train_005_tuning = tuning[:,1:9]
clf_005 = RandomForestClassifier(random_state=0, n_jobs = -1)
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
clf_grid_005 = GridSearchCV(clf_005, param_grid, scoring = "f1", cv=2, n_jobs = -1)
clf_grid_005.fit(X_train_005_tuning, np.around(y_train_005_tuning))

clf_grid_005.best_params_

clf_005 = RandomForestClassifier(random_state=0, n_jobs = -1, criterion = "gini", n_estimators= 1, max_features = 'sqrt')
clf_005.fit(X_train_005, np.around(y_train_005))

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

test = pd.read_csv('test_pred.csv', sep = ',', na_values=['(NA)']).fillna(0).sample(frac= 1)
y_test = test.iloc[:,0].sample(frac = 1)
X_test = test.iloc[:,1:9]

predictions = clf_005.predict_proba(X_test)
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions[:,1]}, columns=['truth', 'pred'])
dataset = pd.concat([X_test,dataset], axis = 1)
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])

precision, recall, thresholds = precision_recall_curve(y_test, predictions[:,1])
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions[:,1])
aps_real = average_precision_score(y_test, predictions[:,1])
print(max(f1_real))
print(log_loss_real)
print(percentage_real)
print(aps_real)
print(top.groupby("Nationality").sum())

predictions_005 = clf_005.predict(X_test)
np.savetxt("predictions_005.csv", predictions_005, delimiter=",")

predictions_005_small = clf_005_small.predict(X_test)
np.savetxt("predictions_005_small.csv", predictions_005_small, delimiter=",")

predictions_005_proba = clf_005.predict_proba(X_test)
np.savetxt("predictions_005_proba.csv", predictions_005_proba, delimiter=",")

predictions_005_small_proba = clf_005_small.predict_proba(X_test)
np.savetxt("predictions_005_small_proba.csv", predictions_005_small_proba, delimiter=",")

"""# epsilon = 0.01"""

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_e_001_small.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 10000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_001_small = data[:,0]
X_train_001_small = data[:,1:9]

tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_001_small_tuning = tuning[:,0]
X_train_001_small_tuning = tuning[:,1:9]
clf_001_small = RandomForestClassifier(random_state=0, n_jobs = -1)
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
clf_grid_001_small = GridSearchCV(clf_001_small, param_grid, scoring = "f1", cv=2, n_jobs = -1)
clf_grid_001_small.fit(X_train_001_small_tuning, np.around(y_train_001_small_tuning))

clf_grid_001_small.best_params_

clf_001_small = RandomForestClassifier(random_state=0, n_jobs = -1, criterion = "gini", n_estimators= 50, max_features = 'sqrt')
clf_001_small.fit(X_train_001_small, np.around(y_train_001_small))

predictions = clf_001_small.predict_proba(X_test)
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions[:,0]}, columns=['truth', 'pred'])
dataset = pd.concat([X_test,dataset], axis = 1)
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])

precision, recall, thresholds = precision_recall_curve(y_test, predictions[:,0])
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions[:,0])
aps_real = average_precision_score(y_test, predictions[:,0])
print(max(f1_real))
print(log_loss_real)
print(percentage_real)
print(aps_real)
print(top.groupby("Nationality").sum())

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data_without_privacy_train_e_001.csv', sep = ',', na_values=['(NA)']).fillna(0)
tuning = data.sample(n = 10000, random_state = 0)
data = np.array(data)
data = data.reshape(len(data),9)
y_train_001 = data[:,0]
X_train_001 = data[:,1:9]

clf_001 = RandomForestClassifier(random_state=0, n_jobs = -1)

tuning = np.array(tuning)
tuning = tuning.reshape(len(tuning),9)
y_train_001_tuning = tuning[:,0]
X_train_001_tuning = tuning[:,1:9]
param_grid = {
    'class_weight': [None,"balanced", "balanced_subsample"],
    'n_estimators': [50, 100, 200,500],
    'max_features' : ['sqrt', 'log2', None],
    'criterion' : ['gini','log_loss', 'entropy'],
    'max_depth': [1,5,None]
}
clf_grid_001 = GridSearchCV(clf_001, param_grid, scoring = "f1", cv=2, n_jobs = -1)
clf_grid_001.fit(X_train_001_tuning, np.around(y_train_001_tuning))

clf_grid_001.best_params_

clf_001 = RandomForestClassifier(random_state=0, n_jobs = -1, criterion = "gini", n_estimators= 50, max_features = 'sqrt', max_depth = 1)
clf_001.fit(X_train_001, np.around(y_train_001))

predictions = clf_001.predict_proba(X_test)
dataset = pd.DataFrame({'truth': y_test, 'pred': predictions[:,0]}, columns=['truth', 'pred'])
dataset = pd.concat([X_test,dataset], axis = 1)
sorted = dataset.sort_values('pred', ascending = False)
top = sorted.nlargest(n = 1000, columns = 'pred')
percentage_real = sum(top['truth'])/1000

precision, recall, thresholds = precision_recall_curve(y_test, predictions[:,0])
f1_real = (2*recall*precision)/(precision + recall)
log_loss_real = log_loss(y_test, predictions[:,0])
aps_real = average_precision_score(y_test, predictions[:,0])
print(max(f1_real))
print(log_loss_real)
print(percentage_real)
print(aps_real)
print(top.groupby("Nationality").sum())

predictions_001 = clf_001.predict(X_test)
np.savetxt("predictions_001.csv", predictions_001, delimiter=",")

predictions_001_small = clf_001_small.predict(X_test)
np.savetxt("predictions_001_small.csv", predictions_001_small, delimiter=",")

predictions_001_proba = clf_001.predict_proba(X_test)
np.savetxt("predictions_001_proba.csv", predictions_001_proba, delimiter=",")

predictions_001_small_proba = clf_001_small.predict_proba(X_test)
np.savetxt("predictions_001_small_proba.csv", predictions_001_small_proba, delimiter=",")
