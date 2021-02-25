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
from tensorflow.keras.callbacks import Callback

random.seed(43)
np.random.seed(43)


if not os.path.exists("xrd"):
    os.makedirs("xrd")


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current > self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True




num_samples = 5000
model_todo="CNN"

# synthetic data to train
targets_unscaled = np.loadtxt("xrd/data_ml/phase_pure_or_not_synthetic.txt")
features_unscaled = np.loadtxt("xrd/data_ml/data_xrd_intensities_synthetic_red.txt")
xs_unscaled = np.loadtxt("xrd/data_ml/data_xrd_wavelengths_synthetic_red.txt")
compositions_unscaled = np.loadtxt("xrd/data_ml/compositions.txt")
compositions_int_unscaled = np.loadtxt("xrd/data_ml/compositions_int.txt")
elements=["Y", "Sm", "La", "Ce", "Pr"]

if len(targets_unscaled)>num_samples:
    targets_unscaled = targets_unscaled[:num_samples]
    features_unscaled = features_unscaled[:num_samples]
    xs_unscaled = xs_unscaled[:num_samples]

# real data as a final TEST
features_TEST_unscaled = np.loadtxt("xrd/data_ml/data_xrd_intensities_red.txt")
targets_TEST_unscaled = np.loadtxt("xrd/data_ml/phase_pure_or_not.txt")
xs_TEST_unscaled = np.loadtxt("xrd/data_ml/data_xrd_wavelengths_red.txt")
names_TEST = []
phases_TEST = []
for line in open("xrd/data_ml/experimental_phases.txt","r"):
    names_TEST.append(line.split()[0])
    phases_TEST.append(line.split()[2])

# reference spectra of pure crystals, just for plotting
data_x_ref = np.loadtxt("xrd/data_ml/data_xrd_wavelengths_ref.txt")
data_y_ref = np.loadtxt("xrd/data_ml/data_xrd_intensities_ref.txt")
names_ref = []
for line in open("xrd/data_ml/data_xrd_names_ref.txt","r"):
    names_ref.append(line.split()[0])





#####################
# data preprocessing
#####################
targets_unscaled = np.expand_dims(targets_unscaled, axis=-1)
targets_TEST_unscaled = np.expand_dims(targets_TEST_unscaled, axis=-1)


# scaling
do_scaling = True
if do_scaling:
    feature_scaler = StandardScaler()
    features = feature_scaler.fit_transform(features_unscaled)
    features_TEST = feature_scaler.transform(features_TEST_unscaled)
else:
    features_scaling = np.zeros((len(features_unscaled)))
    features = np.zeros((features_unscaled.shape))
    for idx in range(len(features_unscaled)):
        features_scaling[idx] = np.max(features_unscaled[idx])
        features[idx] = features_unscaled[idx]/features_scaling[idx]

    features_scaling_TEST = np.zeros((len(features_TEST_unscaled)))
    features_TEST = np.zeros((features_TEST_unscaled.shape))
    for idx in range(len(features_TEST_unscaled)):
        features_scaling_TEST[idx] = np.max(features_TEST_unscaled[idx])
        features_TEST[idx] = features_TEST_unscaled[idx]/features_scaling_TEST[idx]


for idx in range(len(features)):
    features[idx]/=np.max(features[idx])


targets = targets_unscaled
targets_TEST = targets_TEST_unscaled

features = np.array(features, dtype=np.float)
features_TEST = np.array(features_TEST, dtype=np.float)




num_samples = int(features.shape[0])
num_features = int(features.shape[1])

train_index, test_index = train_test_split(list(range(num_samples)), test_size = 0.2, random_state = 0, shuffle=True)


train_xs, test_xs, train_features, test_features, train_labels, test_labels = train_test_split(xs_unscaled, features, targets, test_size = 0.20, random_state = 43, shuffle=True)
valid_xs, test_xs, valid_features, test_features, valid_labels, test_labels = train_test_split(test_xs, test_features, test_labels, test_size = 0.5, random_state = 43, shuffle=True)


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

if not os.path.exists("xrd/models"):
    os.makedirs("xrd/models")
model_dir = "xrd/models/model_test"
shap_file_TEST = "xrd/model_analysis/shap_values_TEST.txt"
shap_file_test = "xrd/model_analysis/shap_values_test.txt"
if not os.path.exists(model_dir):
    if os.path.exists(shap_file_TEST):
        os.system("rm %s"%(shap_file_TEST))
    if os.path.exists(shap_file_test):
        os.system("rm %s"%(shap_file_test))

    reg_l1 = 0.001
    reg_l2 = 0.001

    # NN model
    # sequential model
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(num_features,)))
    #model.add(tf.keras.layers.Dense(150,activation=tf.keras.layers.LeakyReLU(0.03),
    model.add(tf.keras.layers.Dense(150,activation="relu",
                                    kernel_regularizer = tf.keras.regularizers.L1L2(reg_l1,reg_l2),
                                    bias_regularizer =  tf.keras.regularizers.L1L2(reg_l1,reg_l2)
                                    ))
    model.add(tf.keras.layers.Dropout(rate=0.05))
    #model.add(tf.keras.layers.Dense(150,activation=tf.keras.layers.LeakyReLU(0.03),
    model.add(tf.keras.layers.Dense(150,activation="relu",
                                    kernel_regularizer = tf.keras.regularizers.L1L2(reg_l1,reg_l2),
                                    bias_regularizer =  tf.keras.regularizers.L1L2(reg_l1,reg_l2)
                                    ))
    model.add(tf.keras.layers.Dropout(rate=0.05))
    #model.add(tf.keras.layers.Dense(150,activation=tf.keras.layers.LeakyReLU(0.03),
    model.add(tf.keras.layers.Dense(20,activation="relu",
                                    kernel_regularizer = tf.keras.regularizers.L1L2(reg_l1,reg_l2),
                                    bias_regularizer =  tf.keras.regularizers.L1L2(reg_l1,reg_l2)
                                    ))
    model.add(tf.keras.layers.Dropout(rate=0.05))
    model.add(tf.keras.layers.Dense(20,activation="relu",
                                    kernel_regularizer = tf.keras.regularizers.L1L2(reg_l1,reg_l2),
                                    bias_regularizer =  tf.keras.regularizers.L1L2(reg_l1,reg_l2)
                                    ))
    model.add(tf.keras.layers.Dropout(rate=0.05))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))



    # CNN model
    # sequential model
    if model_todo=="CNN":
        train_features = np.expand_dims(train_features, axis=-1)
        valid_features = np.expand_dims(valid_features, axis=-1)
        test_features = np.expand_dims(test_features, axis=-1)
        features_TEST = np.expand_dims(features_TEST, axis=-1)
    model2 = tf.keras.Sequential()
    model2.add(tf.keras.Input(shape=(num_features,1)))
    padding="same"
    model2.add(tf.keras.layers.Conv1D(filters=10, kernel_size=5, padding=padding, activation='relu', input_shape=(267,1)))
    model2.add(tf.keras.layers.Dropout(rate=0.05))
    model2.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model2.add(tf.keras.layers.Conv1D(filters=10, kernel_size=5, padding=padding, activation='relu'))
    model2.add(tf.keras.layers.Dropout(rate=0.05))
    model2.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model2.add(tf.keras.layers.Conv1D(filters=10, kernel_size=5, padding=padding, activation='relu'))
    model2.add(tf.keras.layers.Dropout(rate=0.05))
    model2.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model2.add(tf.keras.layers.Flatten())
    model2.add(tf.keras.layers.Dense(150,activation="relu",
                                    kernel_regularizer = tf.keras.regularizers.L1L2(reg_l1,reg_l2),
                                    bias_regularizer =  tf.keras.regularizers.L1L2(reg_l1,reg_l2)
                                    ))
    model2.add(tf.keras.layers.Dropout(rate=0.05))
    model2.add(tf.keras.layers.Dense(1, activation="sigmoid"))


    if model_todo=="CNN":
        model = model2
    model.summary()

    learning_rate_start = 1e-2
    learning_rate_stop = 1e-4
    epo = 200
    epomin = 100
    epostep = 1

    #x = lr_log_reduction(learning_rate_start, learning_rate_stop, epomin = epomin, epo = epo)
    #for i in range(epo):
    #    print(i, x(i))

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate_start)
    cbk1 = tf.keras.callbacks.LearningRateScheduler(lr_log_reduction(learning_rate_start, learning_rate_stop, epomin = epomin, epo = epo))
    #cbk2 = EarlyStoppingByLossVal(monitor='val_accuracy', value=0.91, verbose=1)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print(model.summary())

    hist = model.fit(train_features,train_labels,
              epochs=epo,
              batch_size=1000,
              validation_freq=epostep,
              callbacks= [cbk1],
              validation_data=(valid_features, valid_labels)
              )

    trainlossall = np.array(hist.history['accuracy'])
    validlossall = np.array(hist.history['val_accuracy'])
    trainaccall = np.array(hist.history['accuracy'])
    validaccall = np.array(hist.history['val_accuracy'])


    #Plot loss vs epochs    
    plt.figure()
    plt.plot(np.arange(trainlossall.shape[0])[:len(trainlossall)],trainlossall,label='Training accuracy',c='blue')
    plt.plot(np.arange(epostep,epo+epostep,epostep)[:len(validlossall)],validlossall,label='Valid accuracy',c='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xlim([0, epo])
    plt.ylim([0.0, 1.0])
    plt.legend(loc='lower right',fontsize='x-large')
    plt.savefig('training_curve.png')
    plt.close()
    if not os.path.exists("xrd/training_curve"):
        os.makedirs("xrd/training_curve")
    np.savetxt("xrd/training_curve/trainlossall.txt", trainlossall)
    np.savetxt("xrd/training_curve/validlossall.txt", validlossall)
    np.savetxt("xrd/training_curve/train_x.txt", np.arange(trainlossall.shape[0]))
    np.savetxt("xrd/training_curve/valid_x.txt", np.arange(epostep,epo+epostep,epostep))


    model.save(model_dir)

else:
    if model_todo=="CNN":
        train_features = np.expand_dims(train_features, axis=-1)
        valid_features = np.expand_dims(valid_features, axis=-1)
        test_features = np.expand_dims(test_features, axis=-1)
        features_TEST = np.expand_dims(features_TEST, axis=-1)
    model = ks.models.load_model(model_dir)
    print(model.summary())

if not os.path.exists(shap_file_TEST) or not os.path.exists(shap_file_test):
    print("generate shap DeepExplainer")
    background = train_features[np.random.choice(train_features.shape[0], min(1000,train_features.shape[0]), replace=False)]
    explainer = shap.DeepExplainer(model, background)
    print("compute shap values")
    shap_values_TEST = explainer.shap_values(features_TEST)
    shap_values_TEST = shap_values_TEST[0]
    if model_todo=="CNN":
        shap_values_TEST = shap_values_TEST[:,:,0]
    print(shap_values_TEST.shape)
    np.savetxt(shap_file_TEST,shap_values_TEST)
    shap_values_test = explainer.shap_values(test_features)
    shap_values_test = shap_values_test[0]
    if model_todo=="CNN":
        shap_values_test = shap_values_test[:,:,0]
    np.savetxt(shap_file_test,shap_values_test)
else:
    shap_values_TEST = np.loadtxt(shap_file_TEST)
    shap_values_test = np.loadtxt(shap_file_test)



pred_valid = model.predict(valid_features)
pred_train = model.predict(train_features)
pred_test = model.predict(test_features)
pred_TEST = model.predict(features_TEST)

pred_valid_int = np.rint(pred_valid)
pred_train_int = np.rint(pred_train)
pred_test_int = np.rint(pred_test)
pred_TEST_int = np.rint(pred_TEST)

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(targets_TEST, pred_TEST, pos_label=None)


gmeans = (tpr * (1-fpr))**0.5
best_threshold_idx = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[best_threshold_idx], gmeans[best_threshold_idx]))

pred_TEST_int2 = np.array([[0] if x<thresholds[best_threshold_idx] else [1] for x in pred_TEST])
pred_test_int2 = np.array([[0] if x<thresholds[best_threshold_idx] else [1] for x in pred_test])

plt.figure()
plt.plot(fpr, tpr, "k-", label="ROC curve")
plt.scatter(fpr[best_threshold_idx], tpr[best_threshold_idx], marker='o', color='black', label='Best')
plt.xlabel("False positive rate")
plt.ylabel("True pusitive rate")
plt.xlim([0,1])
plt.ylim([0,1])
plt.legend(loc="lower right")
plt.savefig("ROC_curve.png", dpi=120)
plt.close()



import sklearn
conf_matrix_train = sklearn.metrics.confusion_matrix(train_labels, pred_train_int)
conf_matrix_valid = sklearn.metrics.confusion_matrix(valid_labels, pred_valid_int)
conf_matrix_test = sklearn.metrics.confusion_matrix(test_labels, pred_test_int)
conf_matrix_TEST = sklearn.metrics.confusion_matrix(targets_TEST, pred_TEST_int)
conf_matrix_TEST2 = sklearn.metrics.confusion_matrix(targets_TEST, pred_TEST_int2)
conf_matrix_test2 = sklearn.metrics.confusion_matrix(test_labels, pred_test_int2)

print("Training confusion matrix")
print(conf_matrix_train)
print("Validation confusion matrix")
print(conf_matrix_valid)
print("Test confusion matrix")
print(conf_matrix_test)
print("Real TEST confusion matrix")
print(conf_matrix_TEST)
print("Real TEST confusion matrix with ideal threshold")
print(conf_matrix_TEST2)
print("Test confusion matrix with ideal threshold")
print(conf_matrix_test2)
pred_TEST_int = pred_TEST_int2


# try different thresholds
#for t in np.linspace(0.99990,1.0,100):
#    print("Threshold is %.3f"%(t))
#    pred_TEST_int_here = np.array([0 if x<t else 1 for x in pred_TEST])
#    conf_matrix_TEST_here = sklearn.metrics.confusion_matrix(targets_TEST, pred_TEST_int_here)
#    print(conf_matrix_TEST_here)

labels_names = ["pure", "mixed"]
outdir_conf = "xrd/confusion_matrix"
if not os.path.exists("xrd/confusion_matrix"):
    os.makedirs("xrd/confusion_matrix")
os.system("rm %s/*"%(outdir_conf))
for idx in range(len(features_TEST)):
    name = names_TEST[idx]
    label_true = labels_names[int(targets_TEST[idx][0])]
    label_pred = labels_names[int(pred_TEST_int[idx][0])]
    phase_experiment = phases_TEST[idx]
    certainty = pred_TEST[idx]*100.0
    if certainty<50.0:
        certainty = 100.0-certainty
    if label_true!=label_pred:
        print("true: %s  predicted (%.3f%%): %s   name: %s %s"%(label_true, certainty, label_pred, name, phase_experiment))
        if label_pred=="mixed":
            target_filename = "pred_mixed_%s"%(name)
        else:
            target_filename = "pred_pure_%s"%(name)
        if not os.path.exists("xrd/all_spectra2"):
            os.makedirs("xrd/all_spectra2")
        os.system("cp xrd/all_spectra2/%s.png %s/%s.png"%(name, outdir_conf, target_filename))

for test in ["TEST", "test"]:
#for test in ["TEST"]:
    if not os.path.exists("xrd/model_analysis"):
        os.makedirs("xrd/model_analysis")
    if not os.path.exists("xrd/model_analysis/%s"%(test)):
        os.makedirs("xrd/model_analysis/%s"%(test))
    print("   ---   start plotting %s data"%(test))
    if not os.path.exists("xrd/model_analysis/%s/correct"%(test)):
        os.makedirs("xrd/model_analysis/%s/correct"%(test))
    else:
        os.system("rm xrd/model_analysis/%s/correct/*"%(test))
    if not os.path.exists("xrd/model_analysis/%s/wrong"%(test)):
        os.makedirs("xrd/model_analysis/%s/wrong"%(test))
    else:
        os.system("rm xrd/model_analysis/%s/wrong/*"%(test))

    if test=="TEST":
        features = features_TEST
        shap_values = shap_values_TEST
        xs_all = xs_TEST_unscaled
        ys_all = features_TEST_unscaled
        pred_int = pred_TEST_int
        pred = pred_TEST
        targets = targets_TEST
        names = names_TEST
        phases_experiment = phases_TEST

    elif test=="test":
        features = test_features
        shap_values = shap_values_test
        xs_all = test_xs
        if model_todo=="CNN":
            if do_scaling:
                ys_all = feature_scaler.inverse_transform(test_features[:,:,0])
            else:
                ys_all = test_features[:,:,0]
        else:
            if do_scaling:
                ys_all = feature_scaler.inverse_transform(test_features)
            else:
                ys_all = test_features
        pred_int = pred_test_int
        pred = pred_test
        targets = test_labels
        names = ["test_sample_%i"%(i) for i in range(len(features))]
        phases_experiment = ["Fake data" for i in range(len(features))]

    print("some statistics on the %s set"%(test))
    certainties = []
    certainties_right = []
    certainties_wrong = []
    for idx in range(len(features)):
        certainty = pred[idx][0]*100.0
        if certainty<50.0:
            certainty = 100.0-certainty
        certainties.append(certainty)
        label_true = labels_names[int(targets[idx][0])]
        label_pred = labels_names[int(pred_int[idx][0])]
        if label_true == label_pred:
            certainties_right.append(certainty)
        else:
            certainties_wrong.append(certainty)
    print("mean of certainties:                       %.1f +- %.1f"%(np.mean(certainties), np.std(certainties)))
    print("mean of certainties (correct predictions): %.1f +- %.1f"%(np.mean(certainties_right), np.std(certainties_right)))
    print("mean of certainties (wrong predictions):   %.1f +- %.1f"%(np.mean(certainties_wrong), np.std(certainties_wrong)))
    print("smallest certaintes for correct predicions:")
    print(np.sort(certainties_right)[:10])
    print("smallest certaintes for wrong predicions:")
    print(np.sort(certainties_wrong)[:10])

    if test=="test":
        continue

    for idx in range(len(features)):
        if (idx+1)%20==0 or (idx+1)==len(features):
            print("   ---   plotting %s data, spectrum %i of %i"%(test, idx+1, len(features)))
        name = names[idx]
        phase_experiment = phases_experiment[idx]
        labels_names = ["pure", "mixed"]
        label_true = labels_names[int(targets[idx][0])]
        label_pred = labels_names[int(pred_int[idx][0])]
        if label_true!=label_pred:
            #print("   ---   %s sample %i has a wrong prediction"%(test,idx))
            pass
        else:
            #print("   ---   %s sample %i has the correct prediction"%(test,idx))
            if test=="test" and np.random.random()<0.9:
                continue
        certainty = pred[idx]*100.0
        if certainty<50.0:
            certainty = 100.0-certainty
        xs = xs_all[idx]
        ys = ys_all[idx]
        shap_values_rel_here = shap_values[idx]/np.max(np.abs(shap_values))

        # sample composition
        sample_composition_string="$("
        cs = compositions_int_unscaled[idx]
        cs /= np.sum(cs)
        for i in range(5):
            c = cs[i]
            if abs(c)>0.01:
                sample_composition_string+="%s_{%.2f}"%(elements[i], c)
                #else:
                #    sample_composition_string+="%s_{%.1f}"%(elements[i], c)

        sample_composition_string+=")O_{2-\delta}$"

        fig, axes = plt.subplots(4,1, figsize=[2/3*6.4, 2/3*8.1], gridspec_kw={'height_ratios': [5, 1, 1, 1]})
        axes[0].set_title("True label: %s - Predicted label: %s (%.2f%%) %s"%(label_true, label_pred, certainty, phase_experiment), fontsize=5)
        if test=="TEST":
            #axes[0].plot(xs, ys, "k-", linewidth=12, label="True label: %s\nPredicted label: %s  (%.2f%%)\n%s"%(label_true, label_pred, certainty, phase_experiment))
            axes[0].plot(xs, ys, "k-", linewidth=2, label="Sample %s\n%s"%(name.split("_")[1], sample_composition_string))
        else:
            #axes[0].plot(xs, ys, "k-", linewidth=12, label="True label: %s\nPredicted label: %s  (%.2f%%)"%(label_true, label_pred, certainty))
            axes[0].plot(xs, ys, "k-", linewidth=2, label="Sample %s\n%s"%(name.split("_")[1], sample_composition_string))
        for i in range(len(xs)-1):
            s = shap_values_rel_here[i]
            if s>0:
                alpha = s
                axes[0].fill_between([xs[i], xs[i+1]], 0, [ys[i],ys[i+1]], facecolor='red', interpolate=False, alpha=alpha)
            elif s<0:
                alpha = -s
                axes[0].fill_between([xs[i], xs[i+1]], 0, [ys[i],ys[i+1]], facecolor='blue', interpolate=False, alpha=alpha)
        #axes[1].plot(xs, shap_values[idx],"k.-")
        #axes[1].plot(xs, np.zeros((len(xs))), "k-")

        colors=["k","k","k"]
        if test=="TEST":
            if label_true=="pure":
                if "Fm-3m" in phase_experiment:
                    colors[0]="b"
                elif "Ia-3" in phase_experiment:
                    colors[1]="b"
                elif "P63" in phase_experiment:
                    colors[2]="b"
        for i in range(3):
            axes[i+1].plot(data_x_ref[i], data_y_ref[i], "-", color=colors[i], label="Reference: %s"%(names_ref[i]))
            axes[i+1].legend(loc="upper right")

        for i in range(3):
            axes[i].get_xaxis().set_ticks([])

        axes[3].set_xlabel("Angle [degree]")
        axes[1].set_ylabel("XRD intensity [a.u.]")
        axes[0].set_ylabel("")
        axes[2].set_ylabel("")
        axes[3].set_ylabel("")
        axes[0].legend(loc="upper right")
        xmin=10.0
        xmax=50.0
        ymin = 0.0
        #ymax = 4500.0
        ymax = [np.max(ys), np.max(data_y_ref[0]), np.max(data_y_ref[1]), np.max(data_y_ref[2])]
        for i in range(4):
            axes[i].set_xlim([xmin, xmax])
        for i in [0,1,2,3]:
            axes[i].set_ylim([ymin, 1.2*ymax[i]])
        plt.subplots_adjust(hspace=0, wspace=0, left=0.2, bottom=0.1, top=0.95, right=0.95)
        if label_true!=label_pred:
            plt.savefig("xrd/model_analysis/%s/wrong/%s.png"%(test, name), dpi=600)
        else:
            plt.savefig("xrd/model_analysis/%s/correct/%s.png"%(test, name), dpi=600)
        plt.close()

        #print("   ---   %s data, plot of spectrum %i of %i done"%(test, idx+1, len(features)))




