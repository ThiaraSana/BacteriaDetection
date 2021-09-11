import glob
import os
import numpy as np
import sys
sys.path.append('/Users/sanaahmed/Desktop/mb-master')
import mb
from pylab import *
import pandas as pd
from matplotlib import pyplot
from sklearn import decomposition

training_folder = '/Users/sanaahmed/Desktop/test/Data/Training_FreshDMEMFBSNoPS/'
# testing_folderBlank = '/Users/sanaahmed/Desktop/test/Data/Testing_FreshDMEMBlank/'
# testing_folderBacCul = '/Users/sanaahmed/Desktop/test/Data/Testing_BacCulture/'
# MSC09_folder = '/Users/sanaahmed/Desktop/test/Data/Testing_SpikedMSCs_0409/'
MSC06_folder = '/Users/sanaahmed/Desktop/SMARTCAMP/UV Vis/Big Machine/04192021/'
MSC21_folder = '/Users/sanaahmed/Desktop/test/Data/Testing_SpikedMSCs_0421/'

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

def SetPCA(absorbance, y, j):
    pca = decomposition.PCA(n_components=6)
    absorbance_centered = absorbance - absorbance.mean(axis=0)
    pca.fit(absorbance_centered)
    absorbance_pca = pca.transform(absorbance_centered)
    loadings = pca.components_.T*np.sqrt(pca.explained_variance_)
    loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC4', 'PC6'])
    plot(loading_matrix)
    plt.savefig('./Graphs/' + 'PCATraining.png')
    show()

    return absorbance_pca, pca

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
    training_files = sorted(glob.glob(training_folder+"*csv"),key=os.path.getmtime)
    trainX, trainY, label = getData(training_files, 93, 2, "1")
    pca_absorbance_train, pca = SetPCA(trainX, trainY, "train_n5")
    pls,sc = mb.raman.pls_x(pca_absorbance_train, trainY, n_components=3)
    train_predicted = pls.predict(pca_absorbance_train)
    # print(train_predicted)
    plotGraphPLS(trainY, train_predicted, "1")

    MSC21_files = sorted(glob.glob(MSC06_folder+"*csv"),key=os.path.getmtime)
    testMSCX_21, testMSCY_21, label_21 = getData(MSC21_files, 90, 2, "5")
    testMSC_21_predicted = pca.transform(testMSCX_21)
    # test_predicted = pls.predict(testMSC_21_predicted)
    print(testMSC_21_predicted)
    # position = [1, 2, 3, 4, 5, 6]
    # plotGraphTraining1(test_predicted, label_21, "5", position)

    # MSC21_files = sorted(glob.glob(MSC21_folder+"*csv"),key=os.path.getmtime)
    # testMSCX_21, testMSCY_21, label_21 = getData(MSC21_files, 93, 2, "5")
    # testMSC_21_predicted = pca.transform(testMSCX_21)
    # test_predicted = pls.predict(testMSC_21_predicted)
    # print(testMSC_21_predicted)
    # position = [1, 2, 3, 4, 5, 6]
    # plotGraphTraining1(test_predicted, label_21, "5", position)
