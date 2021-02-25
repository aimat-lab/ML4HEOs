import os
import sys
import logging
logging.disable(logging.WARNING)
import tensorflow as tf
import tensorflow.keras as ks
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utils import r2_metric, lr_lin_reduction, lr_log_reduction
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import random
import shap
import sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from gplearn.genetic import SymbolicClassifier, SymbolicTransformer

random.seed(43)
np.random.seed(43)



# real data as a final TEST
features_unscaled = np.loadtxt("xrd/data_ml/data_xrd_intensities_red.txt")
targets_unscaled = np.loadtxt("xrd/data_ml/phase_pure_or_not.txt")
xs_unscaled = np.loadtxt("xrd/data_ml/data_xrd_wavelengths_red.txt")
names = []
phases = []
for line in open("xrd/data_ml/experimental_phases.txt","r"):
    names.append(line.split()[0])
    phases.append(line.split()[2])
compositions_unscaled = np.loadtxt("xrd/data_ml/compositions.txt")



elements=["Y", "Sm", "La", "Ce", "Pr"]

for idx in range(len(phases)):
    if targets_unscaled[idx]==1:
        print("mixed phase: sample %s, %3.0f %s %3.0f %s %3.0f %s %3.0f %s %3.0f %s (%s)"%(names[idx], compositions_unscaled[idx][0], elements[0], compositions_unscaled[idx][1], elements[1], compositions_unscaled[idx][2], elements[2], compositions_unscaled[idx][3], elements[3], compositions_unscaled[idx][4], elements[4], phases[idx]))


######################
# additional features
######################

for i in range(4):
    for j in range(i+1,5):
        # AND
        #new = (compositions_unscaled[:,i]*compositions_unscaled[:,j]).reshape((106,1))
        #compositions_unscaled = np.hstack((compositions_unscaled, new))
        #elements.append("%s_AND_%s"%(elements[i], elements[j]))
        # AND2
        new = np.zeros((106))
        for k in range(106):
            if compositions_unscaled[k][i]>0 and compositions_unscaled[k][j]>0:
                new[k] = np.mean([compositions_unscaled[k][i] + compositions_unscaled[k][j]])
            else:
                new[k] = 0.0
        new = new.reshape((106,1))
        compositions_unscaled = np.hstack((compositions_unscaled, new))
        elements.append("%s_AND_%s"%(elements[i], elements[j]))

        # OR
        #new = (compositions_unscaled[:,i]+compositions_unscaled[:,j]).reshape((106,1))
        #compositions_unscaled = np.hstack((compositions_unscaled, new))
        #elements.append("%s_OR_%s"%(elements[i], elements[j]))
        # XOR
        #new = np.zeros((106))
        #for k in range(106):
        #    if compositions_unscaled[k][i]>0 and compositions_unscaled[k][j]>0:
        #        new[k] = 0
        #    else:
        #        new[k] = max(compositions_unscaled[k][i], compositions_unscaled[k][j])
        #new = new.reshape((106,1))
        #compositions_unscaled = np.hstack((compositions_unscaled, new))
        #elements.append("%s_XOR_%s"%(elements[i], elements[j]))







#####################
# data preprocessing
#####################
targets_unscaled = np.expand_dims(targets_unscaled, axis=-1)
targets = targets_unscaled



# scaling
feature_scaler = StandardScaler()
features = feature_scaler.fit_transform(compositions_unscaled)
features = np.array(features, dtype=np.float)



num_samples = int(features.shape[0])
num_features = int(features.shape[1])

train_features, test_features, train_labels, test_labels = train_test_split(features, targets, test_size = 0.6, random_state = 43, shuffle=True)
valid_features, test_features, valid_labels, test_labels = train_test_split(test_features, test_labels, test_size = 0.5, random_state = 43, shuffle=True)



print('Training Features Shape:                 ', train_features.shape)
print('Training Labels Shape:                   ', train_labels.shape)
print('Validation Features Shape:               ', valid_features.shape)
print('Validation Labels Shape:                 ', valid_labels.shape)
print('Testing Features Shape:                  ', test_features.shape)
print('Testing Labels Shape:                    ', test_labels.shape)
num_pure_all, num_mixed_all = len(targets)-np.sum(targets), np.sum(targets)
num_pure_train, num_mixed_train = len(train_labels)-np.sum(train_labels), np.sum(train_labels)
num_pure_valid, num_mixed_valid = len(valid_labels)-np.sum(valid_labels), np.sum(valid_labels)
num_pure_test, num_mixed_test = len(test_labels)-np.sum(test_labels), np.sum(test_labels)

print("Total set:    %i are pure, %i are mixed"%(num_pure_all, num_mixed_all))
print("Training set: %i are pure, %i are mixed"%(num_pure_train, num_mixed_train))
print("Training set: %i are pure, %i are mixed"%(num_pure_valid, num_mixed_valid))
print("Training set: %i are pure, %i are mixed"%(num_pure_test, num_mixed_test))

if not os.path.exists("xrd/models_composition"):
    os.makedirs("xrd/models_composition")

model_dir = "xrd/models_composition/model_test"
if not os.path.exists(model_dir):
    reg_l1 = 0.001
    reg_l2 = 0.001


    # sequential model
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(num_features,)))
    #model.add(tf.keras.layers.Dense(2, activation="relu",
    #                                kernel_regularizer = tf.keras.regularizers.L1L2(reg_l1,reg_l2),
    #                                bias_regularizer =  tf.keras.regularizers.L1L2(reg_l1,reg_l2)
    #                                ))
    #model.add(tf.keras.layers.Dropout(rate=0.05))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))


    model.summary()

    learning_rate_start = 1e-2
    learning_rate_stop = 1e-4
    epo = 200
    epomin = 100
    epostep = 1

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate_start)
    cbks = tf.keras.callbacks.LearningRateScheduler(lr_log_reduction(learning_rate_start, learning_rate_stop, epomin = epomin, epo = epo))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print(model.summary())

    hist = model.fit(train_features, train_labels,
              epochs=epo,
              batch_size=1000,
              validation_freq=epostep,
              callbacks= [cbks],
              validation_data=(valid_features, valid_labels)
              )

    trainlossall = np.array(hist.history['accuracy'])
    validlossall = np.array(hist.history['val_accuracy'])
    trainaccall = np.array(hist.history['accuracy'])
    validaccall = np.array(hist.history['val_accuracy'])


    #Plot loss vs epochs    
    plt.figure()
    plt.plot(np.arange(trainlossall.shape[0]),trainlossall,label='Training Loss',c='blue')
    plt.plot(np.arange(epostep,epo+epostep,epostep),validlossall,label='Valid Loss',c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim([0, epo])
    plt.ylim([0.5, 1.0])
    plt.legend(loc='lower right',fontsize='x-large')
    plt.savefig('training_curve_composition.png')
    plt.close()



    if not os.path.exists("xrd/models_composition"):
        os.makedirs("xrd/models_composition")
    model.save(model_dir)

else:
    model = ks.models.load_model(model_dir)



# random forest

feature_importances_logreg = np.zeros((compositions_unscaled.shape[1]))
feature_importances_GBR = np.zeros((compositions_unscaled.shape[1]))
conf_matrix_train_total_logreg = np.zeros((2,2))
conf_matrix_valid_total_logreg = np.zeros((2,2))
conf_matrix_train_total_GBR = np.zeros((2,2))
conf_matrix_valid_total_GBR = np.zeros((2,2))
conf_matrix_train_total_SR = np.zeros((2,2))
conf_matrix_valid_total_SR = np.zeros((2,2))
num_splits = 10
kf = KFold(n_splits=num_splits, random_state=None, shuffle=True)
counter = 0
SR_programs = []
for train_index, test_index in kf.split(features):
    print("   ###   iteration %i of %i"%(counter+1, num_splits))
    X_train, X_valid = features[train_index], features[test_index]
    y_train, y_valid = targets[train_index], targets[test_index]
    y_train = y_train.reshape((len(y_train),))
    y_valid = y_valid.reshape((len(y_valid),))

    #print("   ---   print %i validation samples"%(len(y_valid)))
    #clf1 = RandomForestClassifier(n_estimators = 100, max_depth = 2)
    clf1 = GradientBoostingClassifier(n_estimators = 100, max_depth = 1)
    clf2 = LogisticRegression()#n_estimators = 100, max_depth = 5)
    #clf = MLPClassifier(hidden_layer_sizes = [10,10], activation = "relu", random_state=0, learning_rate_init=0.01, max_iter = 1000)
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    feature_importances_GBR += clf1.feature_importances_
    feature_importances_logreg += clf2.coef_[0]
    #kernel = RBF() + ConstantKernel()
    #clf = GaussianProcessClassifier(kernel=kernel, random_state=0)
    #clf.fit(X_train, y_train)


    # symbolic regression
    #est = SymbolicTransformer(parsimony_coefficient=.01,
    #est = SymbolicClassifier(parsimony_coefficient=.01,
    #                     feature_names=elements[:5],
    #                     random_state=43)
    #est.fit(X_train[:,:5], y_train)
    #print(est._program)
    #SR_programs.append(est._program)




    # logreg
    pred_train = clf2.predict(X_train)
    pred_valid = clf2.predict(X_valid)

    pred_train_int = np.rint(pred_train)
    pred_valid_int = np.rint(pred_valid)

    conf_matrix_train = sklearn.metrics.confusion_matrix(y_train, pred_train_int)
    conf_matrix_valid = sklearn.metrics.confusion_matrix(y_valid, pred_valid_int)
    if len(conf_matrix_train)==1:
        s = np.sum(conf_matrix_train)
        conf_matrix_train = np.zeros((2,2))
        if pred_train_int[0]==0:
            conf_matrix_train[0][0]+=s
        else:
            conf_matrix_train[1][1]+=s
    if len(conf_matrix_valid)==1:
        s = np.sum(conf_matrix_valid)
        conf_matrix_valid = np.zeros((2,2))
        if pred_valid_int[0]==0:
            conf_matrix_valid[0][0]+=s
        else:
            conf_matrix_valid[1][1]+=s
    conf_matrix_train_total_logreg += conf_matrix_train
    conf_matrix_valid_total_logreg += conf_matrix_valid




    # GBR
    pred_train = clf1.predict(X_train)
    pred_valid = clf1.predict(X_valid)

    pred_train_int = np.rint(pred_train)
    pred_valid_int = np.rint(pred_valid)

    conf_matrix_train = sklearn.metrics.confusion_matrix(y_train, pred_train_int)
    conf_matrix_valid = sklearn.metrics.confusion_matrix(y_valid, pred_valid_int)
    if len(conf_matrix_train)==1:
        s = np.sum(conf_matrix_train)
        conf_matrix_train = np.zeros((2,2))
        if pred_train_int[0]==0:
            conf_matrix_train[0][0]+=s
        else:
            conf_matrix_train[1][1]+=s
    if len(conf_matrix_valid)==1:
        s = np.sum(conf_matrix_valid)
        conf_matrix_valid = np.zeros((2,2))
        if pred_valid_int[0]==0:
            conf_matrix_valid[0][0]+=s
        else:
            conf_matrix_valid[1][1]+=s
    conf_matrix_train_total_GBR += conf_matrix_train
    conf_matrix_valid_total_GBR += conf_matrix_valid




    '''
    # symbolic regression
    pred_train = est.predict(X_train[:,:5])
    pred_valid = est.predict(X_valid[:,:5])

    pred_train_int = np.rint(pred_train)
    pred_valid_int = np.rint(pred_valid)

    conf_matrix_train = sklearn.metrics.confusion_matrix(y_train, pred_train_int)
    conf_matrix_valid = sklearn.metrics.confusion_matrix(y_valid, pred_valid_int)
    if len(conf_matrix_train)==1:
        s = np.sum(conf_matrix_train)
        conf_matrix_train = np.zeros((2,2))
        if pred_train_int[0]==0:
            conf_matrix_train[0][0]+=s
        else:
            conf_matrix_train[1][1]+=s
    if len(conf_matrix_valid)==1:
        s = np.sum(conf_matrix_valid)
        conf_matrix_valid = np.zeros((2,2))
        if pred_valid_int[0]==0:
            conf_matrix_valid[0][0]+=s
        else:
            conf_matrix_valid[1][1]+=s
    #print("   ---   sum of confustion matrix %i"%(np.sum(conf_matrix_valid)))
    #print("   ---   sum of total confusion matrix before summation: %i"%(np.sum(conf_matrix_valid_total)))
    conf_matrix_train_total_SR += conf_matrix_train
    conf_matrix_valid_total_SR += conf_matrix_valid
    #print("   ---   sum of total confusion matrix before summation: %i"%(np.sum(conf_matrix_valid_total)))

    '''


    counter+=1

print("Gradient Boosting Regression")
for i in np.argsort(np.abs(feature_importances_GBR))[::-1]:
    print("Feature %i: %.3f %s"%(i, feature_importances_GBR[i], elements[i]))

    xmin = 0.0
    xmax = np.max(feature_scaler.inverse_transform(features)[:,i])
    border = 0.05*xmax
    plt.figure(figsize=(3,3))
    mask_pure = np.where(targets.T[0]<0.5)[0]
    mask_mixed = np.where(targets.T[0]>0.5)[0]
    plt.scatter(feature_scaler.inverse_transform(features)[:,i][mask_pure], targets.T[0][mask_pure]+np.random.normal(size=(len(mask_pure)))*0.013, marker = "o", s=40, facecolor = "b", edgecolor ="k")
    plt.scatter(feature_scaler.inverse_transform(features)[:,i][mask_mixed], targets.T[0][mask_mixed]+np.random.normal(size=(len(mask_mixed)))*0.013, marker = "o", s=40, facecolor = "r", edgecolor ="k")
    plt.xlabel("%s concentration"%(elements[i].replace("_AND_"," & ")))
    #plt.ylabel("Pure:0, Mixed: 1")
    plt.yticks([0, 1.0], ["Pure", "Mixed"])
    plt.xlim([xmin-border, xmax+border])
    #plt.xticks([xmin, xmax],[0.0,1.0])
    plt.ylim([-0.15,1.15])
    plt.subplots_adjust(left=0.18, bottom=0.18, right=0.95, top=0.95)
    plt.savefig("xrd/compositions/correlation_concentration_%s.png"%(elements[i]), dpi=300)
    plt.close()

print(conf_matrix_train_total_GBR)
print(conf_matrix_valid_total_GBR)


print("Logistic Regression")
for i in np.argsort(np.abs(feature_importances_logreg))[::-1]:
    print("Feature %3i: %7.3f %s"%(i, feature_importances_logreg[i], elements[i]))
print(conf_matrix_train_total_logreg)
print(conf_matrix_valid_total_logreg)


colours =  plt.get_cmap('bwr')
values = feature_importances_logreg-np.min(feature_importances_logreg)
values/=np.max(values)
colours_here = [colours(x) for x in values]
order_logreg = np.argsort(np.abs(feature_importances_logreg))[::-1]
num=15
plt.figure()
plt.plot([-100,100],[0.0,0.0],"k-", linewidth=1)
for i in range(num):
    sign = feature_importances_logreg[order_logreg][i]/np.abs(feature_importances_logreg[order_logreg][i])
    plt.bar([i], [feature_importances_logreg[order_logreg][i]], width=0.8, edgecolor="k", facecolor=colours_here[order_logreg[i]])
    if i<=8:
        if sign>0:
            plt.text(i+0.05, 0.5, elements[order_logreg[i]].replace("_AND_"," & "), rotation=90, ha='center', va='bottom')
        else:
            if i<=3:
                plt.text(i+0.05, -0.2, elements[order_logreg[i]].replace("_AND_"," & "), rotation=90, ha='center', va='top', color="white")
            else:
                plt.text(i+0.05, -0.2, elements[order_logreg[i]].replace("_AND_"," & "), rotation=90, ha='center', va='top')
    else:
        if sign>0:
            plt.text(i+0.05, feature_importances_logreg[order_logreg][i]+sign*0.2, elements[order_logreg[i]].replace("_AND_"," & "), rotation=90, ha='center', va='bottom')
        else:
            plt.text(i+0.05, feature_importances_logreg[order_logreg][i]+sign*0.2, elements[order_logreg[i]].replace("_AND_"," & "), rotation=90, ha='center', va='top')
plt.xlim([-0.9,num-1.0+0.9])
plt.ylim([-11,11])
plt.gca().set_xticks([])
plt.yticks([-10.0, 10.0], ["pure", "mixed"])
plt.ylabel("Feature impact")
plt.savefig("feature_importances.png", dpi=300)
plt.close()





