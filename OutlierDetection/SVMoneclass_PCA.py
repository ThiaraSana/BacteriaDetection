from sklearn.svm import OneClassSVM
import optunity
import optunity.metrics
import glob
import os
import numpy as np
import pandas as pd
import sys
from pylab import *
from sklearn.decomposition import PCA
import xlwt
from xlwt import Workbook

train_folder = '/Users/sanaahmed/Desktop/FlowCell/TrainingData/' # Change to appropriate file path
test_folder = '/Users/sanaahmed/Desktop/FlowCell/TestingData/' # Change to appropriate file path

def getData(folder, footer, header, graphtitle, legendtitle, filename, labelnumber):
    data = []
    name = []
    wave = []
    fixlabel = []
    blankcount = []
    baccount = []

    for i in range (0, len(folder)):

        a=np.genfromtxt(folder[i], skip_header=header, delimiter=',', skip_footer=footer, encoding='latin-1') # Try removine encoding = 'latin-1'... it only works if i use this on my laptop (after my dataset number increased) but im not sure why

        label = a[:,6][0]
        if label ==1:
            blankcount.append(label)
        else: 
            label == -1
            baccount.append(label)

        w1 = a[:,0]

        a1 = a[:,1]
        a2 = a[:,3]
        a3 = a[:,5]
        b1 = (a1 + a2 + a3)/3

        plot(w1, b1)

        name.append(folder[i][labelnumber:])
        fixlabel.append(label)
        data.append(b1)
        wave.append(w1)
        
    data1 = np.array(data, dtype=object)
    wavelength = np.array(wave, dtype = object)
    blankcount = len(blankcount)
    baccount = len(baccount)
    
    xlabel("wavelength [nm]")
    ylabel("Absorbance")
    title(graphtitle)
    legend(name, fontsize = 'xx-small', fancybox = True, title= legendtitle)
    plt.savefig('/Users/sanaahmed/Desktop/FlowCell/UVVis_wPCA_' + filename + '.png') # Change to appropriate file path
    plt.clf()
    
    return data1, wavelength, fixlabel, blankcount, baccount

def SetPCA(absorbance):
    absorbance_centered = absorbance - absorbance.mean(axis=0)
    pca = PCA(n_components= 3, svd_solver='auto')
    pca.fit(absorbance_centered)
    absorbance_pca = pca.transform(absorbance_centered)

    loadings = pca.components_.T*np.sqrt(pca.explained_variance_)
    loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3'])
    plot(loading_matrix)
    xlabel("Wavelength Datapoints")
    title('PCA Loading Matrix')
    legend(['PC1', 'PC2', 'PC3'], fontsize = 'small', fancybox = True)
    
    plt.savefig('/Users/sanaahmed/Desktop/FlowCell/xloadings' + 'PCALoadingMatrix.png') # Change to appropriate file path
    plt.clf()

    return absorbance_pca, pca

def Results(correct, incorrect, pred):
    samplenumber = correct.size + incorrect.size
    correctpred = correct.size
    correctpredpercentage = 100*correct.size/pred.size
    incorrectpred = incorrect.size
    return samplenumber, correctpred, correctpredpercentage, incorrectpred

def VisualizeClustering(data, prediction, filename):
    scores = svm.score_samples(data)
    anom_index = np.where(prediction==-1)
    values = data[anom_index]
    plt.scatter(data[:,0], data[:,1])
    plt.scatter(values[:,0], values[:,1], color='r')
    plt.savefig('/Users/sanaahmed/Desktop/FlowCell/' + filename +'.png') # Change to appropriate file path
    plt.clf()

def SaveResultsToExcel(SaveName, SheetTitle, correcttest, correcttestpercentage, incorrecttest, correcttrain, correcttrainpercentage, incorrecttrain, trainsamplenumb, testsamplenumb, bacteriasamples_train, blanksamples_train, bacteriasamples_test, blanksamples_test):
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(SheetTitle)
    style_header = xlwt.easyxf('font: bold 1')
    sheet.write(0, 1, 'SVM with PCA', style_header)
    sheet.write(1, 1, 'Test', style_header)
    sheet.write(1, 2, 'Train', style_header)
    sheet.write(0, 5, 'Number of Training Samples:', style_header)
    sheet.write(1, 5, trainsamplenumb)
    sheet.write(2, 5, 'Bacteria Samples:', style_header)
    sheet.write(3, 5, bacteriasamples_train)
    sheet.write(2, 6, 'Blank Samples:', style_header)
    sheet.write(3, 6, blanksamples_train)
    sheet.write(0, 7, 'Number of Testing Samples:', style_header)
    sheet.write(1, 7, testsamplenumb)
    sheet.write(2, 7, 'Bacteria Samples:', style_header)
    sheet.write(3, 7, bacteriasamples_test)
    sheet.write(2, 8, 'Blank Samples:', style_header)
    sheet.write(3, 8, blanksamples_test)
    sheet.write(2, 0, 'Correct Predictions:', style_header)
    sheet.write(2, 1, correcttest)
    sheet.write(2, 2, correcttrain)
    sheet.write(3, 0, 'Incorrect Predictions:', style_header)
    sheet.write(3, 1, incorrecttest)
    sheet.write(3, 2, incorrecttrain)
    sheet.write(4, 0, 'True Positive (%):', style_header)
    sheet.write(4, 1, correcttestpercentage)
    sheet.write(4, 2, correcttrainpercentage)
  
    workbook.save('/Users/sanaahmed/Desktop/FlowCell/'+ SaveName)

if __name__ == "__main__":
    training_files = sorted(glob.glob(train_folder+"*csv"),key=os.path.getmtime)
    trainX, wavelength, label_train, blankcount_train, baccount_train = getData(training_files, 173, 2002,  "Training Data UV Vis Spectrum w PCA", 'File', "TrainingDatawPCA", 69)
    # When Running sheet with new data set change the name of the file to prevent it replacing the previous
    trainX_pca, pca = SetPCA(trainX)
    svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.5)
    svm.fit(trainX_pca)
    pred = svm.predict(trainX_pca)
    VisualizeClustering(trainX_pca, pred, "Training_withPCA_08182021")
    # When Running sheet with new data set change the name of the file to prevent it replacing the previous
    correct_train = np.where(pred==label_train)[0]
    incorrect_train = np.where(pred!=label_train)[0]
    print("Training Data")
    print("Correct Predictions:", correct_train.size,"-----",100*correct_train.size/pred.size,"%")
    print("Incorrect Predictions:", incorrect_train.size,"-----",100*incorrect_train.size/pred.size,"%")
    trainsamplenumb, correcttrain, correcttrainpercentage, incorrecttrain = Results(correct_train, incorrect_train, pred)

    testing_files = sorted(glob.glob(test_folder+"*csv"),key=os.path.getmtime)
    testX, wavelength, label_test, blankcount_test, baccount_test = getData(testing_files, 173, 2002,  "Testing Data UV Vis Spectrum w PCA", 'File', "TestingDatawPCA", 68)
    # When Running sheet with new data set change the name of the file to prevent it replacing the previous
    testX_pca = pca.transform(testX)
    pred = svm.predict(testX_pca)
    VisualizeClustering(testX_pca, pred, "Testing_withPCA_08182021")
    # When Running sheet with new data set change the name of the file to prevent it replacing the previous
    correct_test = np.where(pred==label_test)[0]
    incorrect_test = np.where(pred!=label_test)[0]
    print("Testing Data")
    print("Correct Predictions:", correct_test.size,"-----",100*correct_test.size/pred.size,"%")
    print("Incorrect Predictions:", incorrect_test.size,"-----",100*incorrect_test.size/pred.size,"%")
    testsamplenumb, correcttest, correcttestpercentage, incorrecttest = Results(correct_test, incorrect_test, pred)

    SaveResultsToExcel("SVMResultsLog_SVM+PCA_08182021.xls", "Date_08182021", correcttest, correcttestpercentage, incorrecttest, correcttrain, correcttrainpercentage, incorrecttrain, trainsamplenumb, testsamplenumb, baccount_train, blankcount_train, baccount_test, blankcount_test)
    #Change Name of log sheet to avoid the sheet replacing one with the same name