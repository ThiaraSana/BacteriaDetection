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
    trainX_conc, trainY_conc, label = getData(training_files_conc, 93, 2002, "PLS_Training_FreshDEMMFBS", "UV Vis Spectrum Training_FreshDEMMFBS")
    pls_conc,sc = mb.raman.pls_x(trainX_conc, trainY_conc, n_components=3)
    train_predicted_conc = pls_conc.predict(trainX_conc)
    plot(pls_conc.x_loadings_)
    show()
    plotGraphPLS(trainY_conc, train_predicted_conc, "PLS_Training_FreshDEMMFBS", "Predicted NA Concentration Training_FreshDEMMFBS")

    # training_files_pH = sorted(glob.glob(training_folder_pH+"*csv"),key=os.path.getmtime)
    # trainX_pH, trainY_pH, label = Predict_pH(training_files_pH, 93, 336, "Training_FreshDEMMFBS_VaryingpH")
    # pls_pH,sc = mb.raman.pls_x(trainX_pH, trainY_pH, n_components=3)
    # train_predicted_pH = pls_pH.predict(trainX_pH)
    # plotGraphPLS(trainY_pH, train_predicted_pH, "Training_FreshDEMMFBS_VaryingpH")

    testing_files_pH = sorted(glob.glob(testing_folder_pH +"*csv"),key=os.path.getmtime)
    testX, testY, label = getData(testing_files_pH, 93, 2002, "PLS_Testing_FreshDEMMFBS_VaryingpH", "UV Vis Spectrum Testing_FreshDEMMFBS_VaryingpH")
    conc_pred = pls_conc.predict(testX)
    plotGraphPLS(testY, conc_pred, "PLS_Testing_FreshDEMMFBS_VaryingpH", "Predicted NA Concentration Testing_FreshDEMMFBS_VaryingpH")

    # testX_pH, testY_pH, label = Predict_pH(testing_files_pH, 93, 2, "2")
    # test_predicted_pH = pls_pH.predict(testX_pH)
    # print(test_predicted_pH)

    testing_files_msc = sorted(glob.glob(testing_folder_msc_21 +"*csv"),key=os.path.getmtime)
    testX_msc, testY_msc, label = getData(testing_files_msc, 93, 2002, "PLS_Testing_SpikedMSCs_0421", "UV Vis Spectrum Testing_SpikedMSCs_0421")
    conc_pred = pls_conc.predict(testX_msc)
    plotGraphPLS1(testY_msc, conc_pred, "PLS_Testing_SpikedMSCs_0421", "Predicted NA Concentration Testing_SpikedMSCs_0421")
    
    testing_files_msc09 = sorted(glob.glob(testing_folder_msc_09 +"*csv"),key=os.path.getmtime)
    testX_msc09, testY_msc09, label = getData(testing_files_msc09, 93, 1002, "PLS_Testing_SpikedMSCs_0409", "UV Vis Spectrum Testing_SpikedMSCs_0409")
    conc_pred = pls_conc.predict(testX_msc09)
    plotGraphPLS1(testY_msc09, conc_pred, "PLS_Testing_SpikedMSCs_0409", "Predicted NA Concentration Testing_SpikedMSCs_0409")