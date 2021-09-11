import glob
import os
import numpy as np
import sys
sys.path.append('/Users/sanaahmed/Desktop/mb-master')
import mb
from pylab import *
from scipy.optimize import curve_fit
from matplotlib import pyplot
from sklearn import decomposition
from sklearn.decomposition import PCA
import pandas as pd
import scipy.signal

training_folder = '/Users/sanaahmed/Desktop/test/Data/Training_FreshDMEMFBSNoPS/'
testing_folder_pH = '/Users/sanaahmed/Desktop/test/Data/NApH/'
testing_folderBacCul = '/Users/sanaahmed/Desktop/test/Data/Testing_BacCulture/'
MSC09_folder = '/Users/sanaahmed/Desktop/test/Data/Testing_SpikedMSCs_0409/'
MSC06_folder = '/Users/sanaahmed/Desktop/test/Data/Testing_SpikedMSCs_0406/'
testing_folder_msc_21 = '/Users/sanaahmed/Desktop/test/Data/Testing_SpikedMSCs_0421/'

def getData(files, footer, header, j, divisor):

    def objective(x, p1, p2, p3, p4):
    	return p1 * np.log( (x-p3)/p4 ) + p2

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
        # data.append(b1)
        label.append(conc_na)
        
        popt, _ = curve_fit(objective, w1, b1)
        p1, p2, p3, p4 = popt
        # print('y = %.5f * x + %.5f * x^2 + %.5f + %.5f * x^3 + %.5f * x^4 ' % (q, r, s, t,u))
        x_line = arange(min(w1), max(w1), divisor)
        y_line = objective(x_line, p1, p2, p3, p4)
        pyplot.plot(x_line, y_line, '--', color='red')
        pyplot.show()

        x = np.array(w1)
        y = np.array(b1)
        Na_y = y - y_line
        data.append(Na_y)
        data1 = np.array(data)

        yhat = scipy.signal.savgol_filter(data1, 11, 3) # window size 51, polynomial order 3

        plot(w1, Na_y)
        plt.show

    xlabel("wavelength [nm]")
    ylabel("Absorbance")
    legend(label)
    plt.savefig('./Graphs/UV Vis/UV_VIS_' + j + '.png')
    plt.clf()

    return yhat, na_conc, label

# def SetPCA(absorbance, concentration, j):
#     absorbance_centered = absorbance - absorbance.mean(axis=0)
#     pca = decomposition.PCA(n_components= 45, svd_solver='auto')
#     absorbance_pca = pca.fit_transform(absorbance_centered)

#     loadings = pca.components_.T*np.sqrt(pca.explained_variance_)
#     loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC4', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20', 'PC21', 'PC22 ', 'PC23', 'PC24', 'PC25', 'PC26', 'PC27','PC28', 'PC29', 'PC30', 'PC31', 'PC32', 'PC33 ', 'PC34', 'PC35', 'PC36', 'PC37', 'PC38', 'PC39', 'PC40 ', 'PC41', 'PC42', 'PC43', 'PC44', 'PC45'])
#     # loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
#     plot(loading_matrix)
#     show()
    
#     plt.savefig('./Graphs/' + 'PCATraining.png')
#     plt.clf()
#     show()

#     return absorbance_pca, pca

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
    plot(position[6], predicted[6], '.')
    plot(position[7], predicted[7], '.')
    plot(position[8], predicted[8], '.')
    plot(position[9], predicted[9], '.')
    # plot(position[10], predicted[10], '.')
    # plot(position[11], predicted[11],'.')
    legend(label)
    plt.savefig('./Graphs/PLS/PLS_Graph'+ l +'.png')
    plt.clf()

def plotGraphTraining2(predicted, label, l, position):
    plot(position[0], predicted[0], '.')
    plot(position[1], predicted[1], '.')
    plot(position[2], predicted[2], '.')
    plot(position[3], predicted[3], '.')
    plot(position[4], predicted[4], '.')
    plot(position[5], predicted[5], '.')
    legend(label)
    plt.savefig('./Graphs/PLS/PLS_Graph'+ l +'.png')
    plt.clf()

if __name__ == "__main__":    
    training_files = sorted(glob.glob(training_folder+"*csv"),key=os.path.getmtime)
    trainX, trainY, label = getData(training_files, 93, 2002, "CurveFitting_Training_FreshDMEMFBS", 0.14995)
    # pca_train, pca = SetPCA(trainX, trainY, "PCATraining")
    # plot(pca_train)
    # show()
    # pls,sc = mb.raman.pls_x(pca_train, trainY, n_components=1)
    pls,sc = mb.raman.pls_x(trainX, trainY, n_components=3)
    train_predicted = pls.predict(trainX)
    plotGraphPLS(trainY, train_predicted, "CurveFitting_Training_FreshDMEMFBS")

    testing_files = sorted(glob.glob(testing_folder_pH+"*csv"),key=os.path.getmtime)
    testX, testY, label = getData(testing_files, 93, 2002,  "CurveFitting_Testing_FreshDMEMpH", 0.1498)
    # testX_pca = pca.transform(testX)
    # plot(testX_pca)
    # show()
    test_predicted = pls.predict(testX)
    plotGraphPLS(testY, test_predicted, "CurveFitting_Testing_FreshDMEMpH")
    plot(pls.x_loadings_)
    show()

    MSC06_files = sorted(glob.glob(MSC06_folder+"*csv"),key=os.path.getmtime)
    testMSCX_06, testMSCY_06, label_06 = getData(MSC06_files, 93, 1002, "CurveFitting_Testing_SpikedMSCs_0406", 0.1498)
    # testMSCX_pca = pca.transform(testMSCX_06)
    # plot(testMSCX_pca)
    # show()
    testMSC_06_predicted = pls.predict(testMSCX_06)
    position = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plotGraphTraining1(testMSC_06_predicted, label_06, "CurveFitting_Testing_SpikedMSCs_0406", position)
        
    MSC09_files = sorted(glob.glob(MSC09_folder+"*csv"),key=os.path.getmtime)
    testMSCX_09, testMSCY_09, label_09 = getData(MSC09_files, 93,1002, "CurveFitting_Testing_SpikedMSCs_0409", 0.1498)
    # testMSCX_pca09 = pca.transform(testMSCX_09)
    # plot(testMSCX_pca09)
    # show()
    testMSC_09_predicted = pls.predict(testMSCX_09)
    position = [1, 2, 3, 4, 5, 6]
    plotGraphTraining2(testMSC_09_predicted, label_09, "CurveFitting_Testing_SpikedMSCs_0409", position)

    testing_msc_21_files = sorted(glob.glob(testing_folder_msc_21+"*csv"),key=os.path.getmtime)
    testMSCX_21, testMSCY_21, label_21 = getData(testing_msc_21_files, 93,2002, "CurveFitting_Testing_SpikedMSCs_0421", 0.1498)
    # testMSCX_pca21 = pca.transform(testMSCX_21)
    # plot(testMSCX_pca21)
    # show()
    testMSC_21_predicted = pls.predict(testMSCX_21)
    position = [1, 2, 3, 4, 5, 6]
    plotGraphTraining2(testMSC_21_predicted, label_21, "CurveFitting_esting_SpikedMSCs_0421", position)