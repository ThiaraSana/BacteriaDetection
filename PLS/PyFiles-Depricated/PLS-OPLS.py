import glob
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/sanaahmed/Desktop/mb-master')
import mb
from pylab import *
from matplotlib import pyplot
from sklearn import decomposition
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_consistent_length
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import r2_score, accuracy_score
sys.path.append('/Users/sanaahmed/Desktop/pyopls-master')
import pyopls
from pyopls import OPLS

training_folder_conc = '/Users/sanaahmed/Desktop/test/Data/Training_FreshDMEMFBSNoPS/'
testing_folder_pH = '/Users/sanaahmed/Desktop/test/Data/NApH/'
# training_folder_pH = '/Users/sanaahmed/Desktop/test/Data/Training_pH/'
testing_folder_msc_21 = '/Users/sanaahmed/Desktop/test/Data/Testing_SpikedMSCs_0421/'
testing_folder_msc_09 = '/Users/sanaahmed/Desktop/test/Data/Testing_SpikedMSCs_0409/'

def getData(files, footer, header, j, l):

    data = []
    na_conc=[]
    label = []
    for i in range(0,len(files)):
        a=np.genfromtxt(files[i], skip_header=header, delimiter=',', skip_footer=footer)
        conc_na = a[:,6][0]
        w1 = a[:,0]
        a1 = a[:,1]
        a2 = a[:,3]
        a3 = a[:,5]
        b1 = (a1 + a2 + a3)/3
        na_conc.append(conc_na)
        label.append(conc_na)
        data.append(b1)
        
        data1 = np.array(data)
        na_conc1 = np.array(na_conc)

        plot(w1, b1)
        plt.show

    xlabel("wavelength [nm]")
    ylabel("Absorbance")
    title(l)
    legend(label)
    plt.savefig('./Graphs/UV Vis/UV_VIS_' + j + '.png')
    plt.clf()

    return data1, na_conc1, label, w1

def Predict_pH(files, footer, header, j):
    data = []
    pH_phenolred=[]
    label = []
    for i in range(0,len(files)):
        a=np.genfromtxt(files[i], skip_header=header, delimiter=',', skip_footer=footer)
        pH = a[:,6][0]
        w1 = a[:,0]

        a1 = a[:,1]
        a2 = a[:,3]
        a3 = a[:,5]

        b1 = (a1 + a2 + a3)/3

        pH_phenolred.append(pH)
        data.append(b1)
        label.append(pH)

        plot(w1, b1)
        plt.show

    xlabel("wavelength [nm]")
    ylabel("Absorbance")
    legend(label)
    plt.savefig('./Graphs/UV Vis/UV_VIS_' + j + '.png')
    plt.clf()

    return data, pH_phenolred, label

def plotGraphPLS(actual, predicted, j, l):
    figure('PLS Graph')
    plot(actual,'+')
    plot(predicted,'.')
    title(l)
    plt.savefig('./Graphs/PLS/PLS_Graph'+ j +'.png')
    plt.clf()

def plotGraphPLS1(actual, predicted, j, l):
    figure('PLS Graph')
    # plot(actual,'+')
    plot(predicted,'.')
    title(l)
    plt.savefig('./Graphs/PLS/PLS_Graph'+ j +'.png')
    plt.clf()

def plotGraphTraining1(predicted, label, l, position):
    plot(position[0], predicted[0], '.')
    plot(position[1], predicted[1], '.')
    plot(position[2], predicted[2], '.')
    plot(position[3], predicted[3], '.')
    plot(position[4], predicted[4], '.')
    plot(position[5], predicted[5], '.')
    # plot(position[6], predicted[6], '.')
    # plot(position[7], predicted[7], '.')
    # plot(position[8], predicted[8], '.')
    # plot(position[9], predicted[9], '.')
    # legend(label)
    plt.savefig('./Graphs/PLS/PLS_Graph'+ l +'.png')
    plt.clf()

def plotGraphTraining2(predicted, label, l, position):
    plot(position[0], predicted[0], '.')
    # plot(position[1], predicted[1], '.')
    # plot(position[2], predicted[2], '.')
    # plot(position[3], predicted[3], '.')
    # plot(position[4], predicted[4], '.')
    # plot(position[5], predicted[5], '.')
    # legend(label)
    plt.savefig('./Graphs/PLS/PLS_Graph'+ l +'.png')
    plt.clf()

if __name__ == "__main__":

    training_files_conc = sorted(glob.glob(training_folder_conc+"*csv"),key=os.path.getmtime)
    trainX_conc, trainY_conc, label, w1 = getData(training_files_conc, 93, 2002, "Training_FreshDEMMFBS", "UV Vis Spectrum Training_FreshDEMMFBS")
    opls = OPLS(3)
    Z = opls.fit_transform(trainX_conc, trainY_conc)
    plot(Z)
    show()
    # pls_conc,sc = mb.raman.pls_x(Z, trainY_conc, n_components=3)
    train_predicted_conc = opls.predict(Z)
    plotGraphPLS(trainY_conc, train_predicted_conc, "Training_FreshDEMMFBS", "Predicted NA Concentration Training_FreshDEMMFBS")

    testing_files_pH = sorted(glob.glob(testing_folder_pH +"*csv"),key=os.path.getmtime)
    testX, testY, label, w1 = getData(testing_files_pH, 93, 2002, "Testing_FreshDEMMFBS_VaryingpH", "UV Vis Spectrum Testing_FreshDEMMFBS_VaryingpH")
    pp  = opls.transform(testX)
    print(testY)
    plot(pp)
    show()
    conc_pred = pls_conc.predict(pp)
    plotGraphPLS(testY, conc_pred, "Testing_FreshDEMMFBS_VaryingpH", "Predicted NA Concentration Testing_FreshDEMMFBS_VaryingpH")

    testing_files_msc = sorted(glob.glob(testing_folder_msc_21 +"*csv"),key=os.path.getmtime)
    testX_msc, testY_msc, label, w1 = getData(testing_files_msc, 93, 2002, "Testing_SpikedMSCs_0421", "UV Vis Spectrum Testing_SpikedMSCs_0421")
    qq  = opls.transform(testX_msc)
    plot(qq)
    show()
    conc_pred = pls_conc.predict(qq)
    print(conc_pred)
    plotGraphPLS1(testY_msc, conc_pred, "Testing_SpikedMSCs_0421", "Predicted NA Concentration Testing_SpikedMSCs_0421")