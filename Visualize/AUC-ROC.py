# -*-coding:utf-8-*-

import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import precision_recall_curve
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
# input_fn = 'y_pred_SPEI6_MGRU_FCN_GCN.npy'
# input_fn1 = 'y_true_SPEI6_MGRU_FCN_GCN.npy'
# probas = 'test_probas_SPEI6_MGRU_FCN_GCN.npy'
# pred = np.load(open(input_fn,'rb'))
# true = np.load(open(input_fn1,'rb'))
# probas2 = np.load(open(probas,'rb'))
# #
# pred = pd.DataFrame(pred)
# true = pd.DataFrame(true)
# probas3 = pd.DataFrame(probas2)
# pred.to_excel('AUC-ROC/SPEI6_MGRU_FCN_GCN_pred.xls',index=None)
# true.to_excel('AUC-ROC/SPEI6_MGRU_FCN_GCN_true.xls',index=None)
# probas3.to_excel('AUC-ROC/SPEI6_MGRU_FCN_GCN_probas.xls',index=None)



data1 = pd.read_excel('AUC-ROC/real_result.xlsx')
data = pd.read_excel('AUC-ROC/MSTSN.xlsx')
true_y = data1.iloc[1:, 0]
true_y=true_y.to_numpy()
true_y=to_categorical(true_y,num_classes=5)
PM_y = data.iloc[1:,1:].to_numpy()
n_classes=PM_y.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(true_y[:, i], PM_y[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
from scipy import interp

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.ticker import FuncFormatter

lw = 2
# Plot all ROC curves
plt.figure()
labels = ['Category 0', 'Category 1', 'Category 2', 'Category 3', 'Category 4']
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blue', 'yellow'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label=labels[i] + '(area = {0:0.4f})'.format(roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('True Positive Rate (%)')
plt.ylabel('False Positive Rate (%)')
plt.title('ROC-AUC curves of different drought categorization classes')


def to_percent(temp, position):
    return '%1.0f' % (100 * temp)


plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
plt.legend(loc="lower right")
plt.show()
