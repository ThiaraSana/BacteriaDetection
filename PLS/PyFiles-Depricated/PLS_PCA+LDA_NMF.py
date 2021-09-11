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
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

training_folder_conc = '/Users/sanaahmed/Desktop/test/Data/Training_FreshDMEMFBSNoPS/'
testing_folder_pH = '/Users/sanaahmed/Desktop/test/Data/NApH/'
training_folder_pH = '/Users/sanaahmed/Desktop/test/Data/Training_pH/'
testing_folder_msc_21 = '/Users/sanaahmed/Desktop/test/Data/Testing_SpikedMSCs_0421/'

def getData(files, footer, header, j):

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
    legend(label)
    plt.savefig('./Graphs/UV Vis/UV_VIS_' + j + '.png')
    plt.clf()

    return data1, na_conc1, label

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

def SetNMF(absorbance, y, j):
    nmf = decomposition.NMF(n_components=16)
    nmf.fit(absorbance)
    absorbance_nmf = nmf.transform(absorbance)

    return absorbance_nmf, nmf

def SetPCA(absorbance, concentration, j):
    LDA = LinearDiscriminantAnalysis()
    concentration=concentration.astype('int')
    LDA.fit(absorbance, concentration)
    absorbance_LDA = LDA.transform(absorbance)

    plt.savefig('./Graphs/' + 'PCATraining.png')
    show()

    return absorbance_LDA, LDA

def plotGraphPLS(actual, predicted, j):
    figure('PLS Graph')
    plot(actual,'+')
    plot(predicted,'.')
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
    trainX_conc, trainY_conc, label = getData(training_files_conc, 93, 2, "3")
    #TRAIN NMF
    nmf_train, nmf = SetNMF(trainX_conc, trainY_conc, "train_n5")
    #TRAIN PLS_CONC
    pls_conc,sc = mb.raman.pls_x(nmf_train, trainY_conc, n_components=3)
    train_predicted_conc = pls_conc.predict(nmf_train)
    plot(pls_conc.x_loadings_)
    show()
    plotGraphPLS(trainY_conc, train_predicted_conc, "3")

    training_files_pH = sorted(glob.glob(training_folder_pH+"*csv"),key=os.path.getmtime)
    trainX_pH, trainY_pH, label = Predict_pH(training_files_pH, 1093, 336, "4")
    #TRAIN PLS_pH
    pls_pH,sc = mb.raman.pls_x(trainX_pH, trainY_pH, n_components=3)
    train_predicted_pH = pls_pH.predict(trainX_pH)
    plotGraphPLS(trainY_pH, train_predicted_pH, "4")

    ######################################################
    testing_files_pH = sorted(glob.glob(testing_folder_pH +"*csv"),key=os.path.getmtime)
    testX_conc, testY_conc, label = getData(testing_files_pH, 93, 2, "13")
    trainX_conc_nmf = nmf.transform(testX_conc)
    conc_pred = pls_conc.predict(trainX_conc_nmf)
    plotGraphPLS(testY_conc, conc_pred, "13")
    
    # testX_pH, testY_pH, label = Predict_pH(testing_files_pH, 1093, 2, "14")
    # test_predicted_pH = pls_pH.predict(testX_pH)
    # print(test_predicted_pH)
    # print(conc_pred)

    testing_files_msc = sorted(glob.glob(testing_folder_msc_21 +"*csv"),key=os.path.getmtime)
    testX_msc, testY_msc, label = getData(testing_files_msc, 93, 2, "14")
    trainX_msc_nmf = nmf.transform(testX_msc)
    conc_pred = pls_conc.predict(trainX_msc_nmf)
    plotGraphPLS(testY_msc, conc_pred, "14")



### PCA ################################################
if __name__ == "__main__":

    training_files_conc = sorted(glob.glob(training_folder_conc+"*csv"),key=os.path.getmtime)
    trainX_conc, trainY_conc, label = getData(training_files_conc, 93, 2, "1")
    #TRAIN PCA
    pca_train, pca = SetPCA(trainX_conc, trainY_conc, "train_n5")
    #TRAIN PLS_CONC
    pls_conc,sc = mb.raman.pls_x(pca_train, trainY_conc, n_components=3)
    train_predicted_conc = pls_conc.predict(pca_train)
    plot(pls_conc.x_loadings_)
    show()
    plotGraphPLS(trainY_conc, train_predicted_conc, "1")

    training_files_pH = sorted(glob.glob(training_folder_pH+"*csv"),key=os.path.getmtime)
    trainX_pH, trainY_pH, label = Predict_pH(training_files_pH, 1093, 336, "2")
    #TRAIN PLS_pH
    pls_pH,sc = mb.raman.pls_x(trainX_pH, trainY_pH, n_components=3)
    train_predicted_pH = pls_pH.predict(trainX_pH)
    plotGraphPLS(trainY_pH, train_predicted_pH, "2")

    ######################################################
    testing_files_pH = sorted(glob.glob(testing_folder_msc_21 +"*csv"),key=os.path.getmtime)
    testX, testY, label = getData(testing_files_pH, 93, 2, "10")
    testX_pca = pca.transform(testX)
    conc_pred = pls_conc.predict(testX_pca)
    plotGraphPLS(testY, conc_pred, "10")
    
    # testX_pH, testY_pH, label = Predict_pH(testing_files_pH, 1093, 2, "2")
    # test_predicted_pH = pls_pH.predict(testX_pH)

    testing_files_msc = sorted(glob.glob(testing_folder_msc_21 +"*csv"),key=os.path.getmtime)
    testX_msc, testY_msc, label = getData(testing_files_msc, 93, 2, "12")
    trainX_msc_pca = pca.transform(testX_msc)
    conc_pred = pls_conc.predict(trainX_msc_pca)
    plotGraphPLS(testY_msc, conc_pred, "12")