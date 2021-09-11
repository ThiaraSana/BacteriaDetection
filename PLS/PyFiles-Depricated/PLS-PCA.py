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
        
        data1 = np.log(array(data))
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
    title(l)
    legend(label)
    plt.savefig('./Graphs/UV Vis/UV_VIS_' + j + '.png')
    plt.clf()

    return data, pH_phenolred, label

def SetPCA(absorbance, concentration, j):
    absorbance_centered = absorbance - absorbance.mean(axis=0)
    pca = decomposition.PCA(n_components= 45, svd_solver='auto')
    absorbance_pca = pca.fit_transform(absorbance_centered)

    loadings = pca.components_.T*np.sqrt(pca.explained_variance_)
    loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC4', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20', 'PC21', 'PC22 ', 'PC23', 'PC24', 'PC25', 'PC26', 'PC27', 'PC1', 'PC2', 'PC3', 'PC4', 'PC4', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18'])
    # loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3'])
    plot(loading_matrix)
    show()
    
    # plt.savefig('./Graphs/' + 'PCATraining.png')
    # plt.clf()
    # show()

    return absorbance_pca, pca

def plotGraphPLS(actual, predicted, j, l):
    figure('PLS Graph')
    plot(actual,'+')
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

def plotGraphTraining3(predicted, label, l, position, heading):
    plot(position[0], predicted[0], '.')
    plot(position[1], predicted[1], '.')
    plot(position[2], predicted[2], '.')
    plot(position[3], predicted[3], '.')
    plot(position[4], predicted[4], '.')
    plot(position[5], predicted[5], '.')
    title(heading)
    plt.savefig('./Graphs/PLS/PLS_Graph'+ l +'.png')
    plt.clf()


### PCA ################################################
if __name__ == "__main__":

    training_files_conc = sorted(glob.glob(training_folder_conc+"*csv"),key=os.path.getmtime)
    trainX_conc, trainY_conc, label = getData(training_files_conc, 93, 2, "45PCA_Training_FreshDEMMFBS", "UV Spectrum Training_FreshDEMMFBS_PCA")
    #TRAIN PCA
    pca_train, pca = SetPCA(trainX_conc, trainY_conc, "PCATraining")
    #TRAIN PLS_CONC
    pls_conc,sc = mb.raman.pls_x(pca_train, trainY_conc, n_components=3)
    # pls_conc,sc = mb.raman.pls_x(trainX_conc, trainY_conc, n_components=3)
    train_predicted_conc = pls_conc.predict(pca_train)
    # train_predicted_conc = pls_conc.predict(trainX_conc)
    plot(pls_conc.x_loadings_)
    show()
    plotGraphPLS(trainY_conc, train_predicted_conc, "45PCA_Training_FreshDEMMFBS", "NA Concentration Prediction Training_FreshDEMMFBS_PCA")

    # training_files_pH = sorted(glob.glob(training_folder_pH+"*csv"),key=os.path.getmtime)
    # trainX_pH, trainY_pH, label = Predict_pH(training_files_pH, 1093, 336, "2")
    # #TRAIN PLS_pH
    # pls_pH,sc = mb.raman.pls_x(trainX_pH, trainY_pH, n_components=3)
    # train_predicted_pH = pls_pH.predict(trainX_pH)
    # plotGraphPLS(trainY_pH, train_predicted_pH, "2")

    testing_files_pH = sorted(glob.glob(testing_folder_pH +"*csv"),key=os.path.getmtime)
    testX, testY, label = getData(testing_files_pH, 93, 2, "45PCA_Testing_FreshDEMMFBS_VaryingpH", "UV Spectrum Testing_FreshDEMMFBS_VaryingpH_PCA")
    testX_pca = pca.transform(testX)
    conc_pred = pls_conc.predict(testX_pca)
    # conc_pred = pls_conc.predict(testX)
    plotGraphPLS(testY, conc_pred, "45PCA_Testing_FreshDEMMFBS_VaryingpH", "NA Concentration Prediction Testing_FreshDEMMFBS_VaryingpH_PCA")
    # testX_pH, testY_pH, label = Predict_pH(testing_files_pH, 1093, 2, "2")
    # test_predicted_pH = pls_pH.predict(testX_pH)

    testing_files_msc = sorted(glob.glob(testing_folder_msc_21 +"*csv"),key=os.path.getmtime)
    testX_msc, testY_msc, label = getData(testing_files_msc, 93, 2, "45PCA_Testing_SpikedMSCs_0421", "UV Spectrum Testing_SpikedMSCs_0421_PCA")
    trainX_msc_pca = pca.transform(testX_msc)
    conc_pred = pls_conc.predict(trainX_msc_pca)
    print(conc_pred)
    # conc_pred = pls_conc.predict(testX_msc)
    position = [1, 2, 3, 4, 5, 6]
    plotGraphTraining3(conc_pred, label, "45PCA_Testing_SpikedMSCs_0421", position, "NA Concentration Prediction Testing_SpikedMSCs_0421_PCA")