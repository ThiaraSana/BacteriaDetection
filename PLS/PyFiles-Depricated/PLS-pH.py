import glob
import os
import numpy as np
import sys
sys.path.append('/Users/sanaahmed/Desktop/mb-master')
import mb
from pylab import *
import scipy.signal

training_folder = '/Users/sanaahmed/Desktop/test/Data_0.5mm/Training_pH/'
testing_folder = '/Users/sanaahmed/Desktop/test/Data_0.5mm/NApH/'
testing_folder2 = '/Users/sanaahmed/Desktop/test/Data_2mm/Training_FreshDMEMFBSNoPS_0601/'
training_pca_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_BacCulture_0525/ec/'

def getData(files, footer, header, j, k):

    data = []
    pH_phenolred=[]
    label = []
    for i in range(0,len(files)):
        a=np.genfromtxt(files[i], skip_header=header, delimiter=',', skip_footer=footer)
        pH = a[:,7][0]
        w1 = a[:,0]

        # a1 = (a[:,1] - min(a[:,1])) / (max(a[:,1]) - min(a[:,1]))
        # a2 = (a[:,3] - min(a[:,3])) / (max(a[:,3]) - min(a[:,3]))
        # a3 = (a[:,5] - min(a[:,5])) / (max(a[:,5]) - min(a[:,5]))
        a1 = a[:,1]
        a2 = a[:,3]
        a3 = a[:,5]

        b1 = (a1 + a2 + a3)/3
        pH_phenolred.append(pH)
        data.append(b1)
        label.append(pH)

        # yhat = scipy.signal.savgol_filter(data, 201, 3) # window size 51, polynomial order 3

        data1 = np.log(array(data))
        pH_phenolred1 = np.array(pH_phenolred)
        

        plot(w1, b1)
        plt.show

    xlabel("wavelength [nm]")
    ylabel("Absorbance")
    legend(label)
    title(k)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/UV_VIS' + j + '.png')
    plt.clf()

    return data1, pH_phenolred1, label

def getData1(files, footer, header, j, k):
    
    data = []
    # pH_phenolred=[]
    label = []
    for i in range(0,len(files)):
        a=np.genfromtxt(files[i], skip_header=header, delimiter=',', skip_footer=footer)
        # pH = a[:,7][0]
        w1 = a[:,0]

        # a1 = (a[:,1] - min(a[:,1])) / (max(a[:,1]) - min(a[:,1]))
        # a2 = (a[:,3] - min(a[:,3])) / (max(a[:,3]) - min(a[:,3]))
        # a3 = (a[:,5] - min(a[:,5])) / (max(a[:,5]) - min(a[:,5]))
        a1 = a[:,1]
        a2 = a[:,3]
        a3 = a[:,5]

        b1 = (a1 + a2 + a3)/3
        # pH_phenolred.append(pH)
        data.append(b1)
        # label.append(pH)

        # yhat = scipy.signal.savgol_filter(data, 201, 3) # window size 51, polynomial order 3

        data1 = np.log(array(data))
        # pH_phenolred1 = np.array(pH_phenolred)
        

        plot(w1, b1)
        plt.show

    xlabel("wavelength [nm]")
    ylabel("Absorbance")
    legend(label)
    title(k)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/UV_VIS' + j + '.png')
    plt.clf()

    return data1, label

def plotGraphPLS(actual, predicted, j, k):
    figure('PLS Graph')
    plot(actual,'+')
    plot(predicted,'.')
    title(k)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS_Graph'+ j +'.png')
    plt.clf()

def plotGraphTraining(predicted, label, l, position):
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
    plot(position[10], predicted[10], '.')
    plot(position[11], predicted[11], '.')
    plot(position[12], predicted[12], '.')

    legend(label)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS_Graph'+ l +'.png')
    plt.clf()

def plotGraphTraining1(predicted, label, l, position):
    plot(position[0], predicted[0], '.')
    plot(position[1], predicted[1], '.')
    plot(position[2], predicted[2], '.')
    plot(position[3], predicted[3], '.')
    plot(position[4], predicted[4], '.')
    plot(position[5], predicted[5], '.')
    plot(position[6], predicted[6], '.')

    legend(label)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS_Graph'+ l +'.png')
    plt.clf()

if __name__ == "__main__":    
    training_files = sorted(glob.glob(training_folder+"*csv"),key=os.path.getmtime)
    trainX, trainY, label = getData(training_files, 1000, 336, "pHTraining_FreshDMEMFBS", "UV Spectrum Training FreshDMEMFBS Varying pH")
    pls,sc = mb.raman.pls_x(trainX, trainY, n_components=3)
    train_predicted = pls.predict(trainX)
    plotGraphPLS(trainY, train_predicted, "pHTraining_FreshDMEMFBS", "UV Spectrum Training FreshDMEMFBS Varying pH")
    plot(pls.x_loadings_)
    show()

    testing_files = sorted(glob.glob(testing_folder+"*csv"),key=os.path.getmtime)
    testX, testY, label = getData(testing_files, 1000, 2, "pHTesting_FreshDMEMFBS", "UV Spectrum Testing FreshDMEMFBS Varying pH")
    test_predicted = pls.predict(testX)
    plotGraphPLS(testY, test_predicted,"pHTesting_FreshDMEMFBS", "UV Spectrum Testing FreshDMEMFBS Varying pH")
    print(test_predicted)

    testing_files = sorted(glob.glob(testing_folder2+"*csv"),key=os.path.getmtime)
    testX, label = getData1(testing_files, 1000, 2, "2pHTesting_FreshDMEMFBS", "UV Spectrum Testing FreshDMEMFBS Varying pH")
    test_predicted = pls.predict(testX)
    position = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    plotGraphTraining(test_predicted, label, "2pHTesting_FreshDMEMFBS", position)
    # print(test_predicted)

    testing_files = sorted(glob.glob(training_pca_folder+"*csv"),key=os.path.getmtime)
    testX, label = getData1(testing_files, 1000, 2, "3pHTesting_FreshDMEMFBS", "UV Spectrum Testing FreshDMEMFBS Varying pH")
    test_predicted = pls.predict(testX)
    position = [1,2,3,4,5,6,7]
    plotGraphTraining1(test_predicted, label, "3pHTesting_FreshDMEMFBS", position)
    print(test_predicted)