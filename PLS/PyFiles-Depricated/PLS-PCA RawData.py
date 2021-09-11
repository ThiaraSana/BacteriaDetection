import glob
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/sanaahmed/Desktop/mb-master')
import mb
from pylab import *
from matplotlib import pyplot
import scipy.signal
from sklearn import decomposition
from sklearn.decomposition import PCA


pls_training_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Training_FreshDMEMFBSNoPS_0601/'
#Testing with ec culture (Is the complexity of the bacteria culture problem solved with PCA?)
pca_training_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_BacCulture_0525/ec/'
#Testing with Fresh DMEM spiked with NA at low concentration and a different pH (Is detection at a low concentration of NA possible and does pH matter like hypothesized?)
oldsample_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Old_FreshDMEMFBSNoPS_0601/'
#Testing with other types of bacteria cultured with varying CFUs (Does the type of bacteria matter?)
ab_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_BacCulture_0525/ab/'
pa_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_BacCulture_0525/pa/'
sa_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_BacCulture_0525/sa/'
se_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_BacCulture_0525/se/'
kp_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_BacCulture_0525/kp/'
#Testing with MSC cultures (should not have any NA but will have NAM and other metabolites, do these affect the PLS?)
msc_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_MSC_0525/'
#Testing with samples that have a 0.5mm path length (Can we still use these samples?)
shortpath_folder = '/Users/sanaahmed/Desktop/test/Data_0.5mm/Training_FreshDMEMFBSNoPS/'

def getData(files, footer, header, j, l):
    
    data = []
    na_conc=[]
    label = []
    wave = []

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
        wave.append(w1)

        data1 = np.array(data)
        na_conc1 = np.array(na_conc)
        wavelength = np.array(wave)

        plot(w1, b1)
        plt.show

    xlabel("wavelength [nm]")
    ylabel("Absorbance")
    title(l)
    legend(label, fontsize = 'small', fancybox = True, title= 'ug/ml')
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/RawData/UVvis/UVvis' + j + '.png')
    plt.clf()

    return data1, na_conc1, wavelength

def SetPCA(absorbance):
    absorbance_centered = absorbance - absorbance.mean(axis=0)
    pca = decomposition.PCA(n_components= 3, svd_solver='auto')
    absorbance_pca = pca.fit_transform(absorbance_centered)

    loadings = pca.components_.T*np.sqrt(pca.explained_variance_)
    loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3'])
    plot(loading_matrix)
    
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/RawData/xloadings/' + 'PCALoadingMatrix.png')
    plt.clf()

    return absorbance_pca, pca

def plotGraphPLS(actual, predicted, j, l):
    figure('PLS Graph')
    plot(actual,'+')
    plot(predicted,'.')
    title(l)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/RawData/PLS/PLSprediction'+ j +'.png')
    plt.clf()

def plotGraphTraining7(predicted, label, l, position, heading):
    plot(position[0], predicted[0], '.')
    plot(position[1], predicted[1], '.')
    plot(position[2], predicted[2], '.')
    plot(position[3], predicted[3], '.')
    plot(position[4], predicted[4], '.')
    plot(position[5], predicted[5], '.')
    plot(position[6], predicted[6], '.')

    legend(label, fontsize = 'small', fancybox = True, title= 'CFU')
    title(heading)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/RawData/PLS/PLSprediction'+ l +'.png')
    plt.clf()

def plotGraphTraining4(predicted, label, l, position, heading):
    plot(position[0], predicted[0], '.')
    plot(position[1], predicted[1], '.')
    plot(position[2], predicted[2], '.')
    plot(position[3], predicted[3], '.')

    legend(label, fontsize = 'small', fancybox = True, title= 'MSC')
    title(heading)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/RawData/PLS/PLSprediction'+ l +'.png')
    plt.clf()

if __name__ == "__main__":
    #Testing with ec culture (Is the complexity of the bacteria culture problem solved with PCA?)
    testing_files_bac = sorted(glob.glob(pca_training_folder +"*csv"),key=os.path.getmtime)
    testX_bac, testY_bac, wavelength_bac = getData(testing_files_bac, 93, 2002, "_ec", "UV Vis Spectrum ec")
    pca_train, pca = SetPCA(testX_bac)
    
    #PLS Training
    training_files = sorted(glob.glob(pls_training_folder+"*csv"),key=os.path.getmtime)
    trainX, trainY, wavelength = getData(training_files, 93, 2002, "_FreshDMEMFBS", "UV Spectrum Fresh DMEM w FBS")
    trainX_pca = pca.transform(trainX)
    pls,sc = mb.raman.pls_x(trainX_pca, trainY, n_components=3)
    train_predicted = pls.predict(trainX_pca)
    plot(pls.x_scores_)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/RawData/xloadings/' + 'PLSLoadings.png')
    plt.clf()
    plotGraphPLS(trainY, train_predicted, "_FreshDMEMFBS_Training", "NA Concentration Prediction Fresh DMEM w FBS")

    #CONTINUED: Testing with ec culture (Is the complexity of the bacteria culture problem solved with PCA?)
    conc_pred = pls.predict(pca_train)
    position = [1, 2, 3, 4, 5, 6, 7]
    plotGraphTraining7(conc_pred, testY_bac, "_ec_Testing", position, "Predicted NA Concentration ec")

    #Testing with Fresh DMEM spiked with NA at low concentration and a different pH (Is detection at a low concentration of NA possible and does pH matter like hypothesized?)
    testing_files = sorted(glob.glob(oldsample_folder +"*csv"),key=os.path.getmtime)
    testX, testY, wavelength = getData(testing_files, 93, 2002, "_oldFreshDMEMFBS", "UV Spectrum old Fresh DMEM w FBS")
    testX_pca = pca.transform(testX)
    conc_pred = pls.predict(testX_pca)
    plotGraphPLS(testY, conc_pred, "_oldFreshDMEMFBS_Testing", "NA Concentration Prediction Old Fresh DMEM w FBS")

    #Testing with other types of bacteria cultured with varying CFUs (Does the type of bacteria matter?)
    ab_files = sorted(glob.glob(ab_folder +"*csv"),key=os.path.getmtime)
    abX, abY, wavelength = getData(ab_files, 93, 2002, "_ab", "UV Spectrum ab")
    abX_pca = pca.transform(abX)
    conc_pred = pls.predict(abX_pca)
    position = [1, 2, 3, 4, 5, 6, 7]
    plotGraphTraining7(conc_pred, abY, "_ab_Testing", position, "Predicted NA Concentration ab")

    pa_files = sorted(glob.glob(pa_folder +"*csv"),key=os.path.getmtime)
    paX, paY, wavelength = getData(pa_files, 93, 2002, "_pa", "UV Spectrum pa")
    paX_pca = pca.transform(paX)
    conc_pred = pls.predict(paX_pca)
    position = [1, 2, 3, 4, 5, 6, 7]
    plotGraphTraining7(conc_pred, paY, "_pa_Testing", position, "Predicted NA Concentration pa")

    sa_files = sorted(glob.glob(sa_folder +"*csv"),key=os.path.getmtime)
    saX, saY, wavelength = getData(sa_files, 93, 2002, "_sa", "UV Spectrum sa")
    saX_pca = pca.transform(saX)
    conc_pred = pls.predict(saX_pca)
    position = [1, 2, 3, 4, 5, 6, 7]
    plotGraphTraining7(conc_pred, saY, "_sa_Testing", position, "Predicted NA Concentration sa")

    se_files = sorted(glob.glob(se_folder +"*csv"),key=os.path.getmtime)
    seX, seY, wavelength = getData(se_files, 93, 2002, "_se", "UV Spectrum se")
    seX_pca = pca.transform(seX)
    conc_pred = pls.predict(seX_pca)
    position = [1, 2, 3, 4, 5, 6, 7]
    plotGraphTraining7(conc_pred, seY, "_se_Testing", position, "Predicted NA Concentration se")

    kp_files = sorted(glob.glob(kp_folder +"*csv"),key=os.path.getmtime)
    kpX, kpY, wavelength = getData(kp_files, 93, 2002, "_kp_Testing", "UV Spectrum kp")
    kpX_pca = pca.transform(kpX)
    conc_pred = pls.predict(kpX_pca)
    position = [1, 2, 3, 4, 5, 6, 7]
    plotGraphTraining7(conc_pred, kpY, "_kp_Testing", position, "Predicted NA Concentration kp")

    # #Testing with MSC cultures (should not have any NA but will have NAM and other metabolites, do these affect the PLS?)
    msc_files = sorted(glob.glob(msc_folder +"*csv"),key=os.path.getmtime)
    mscX, mscY, wavelength = getData(msc_files, 93, 2002, "_MSC", "UV Spectrum MSC")
    mscX_pca = pca.transform(mscX)
    conc_pred = pls.predict(mscX_pca)
    position = [1, 2, 3, 4]
    plotGraphTraining4(conc_pred, mscY, "_MSC_Testing", position, "NA Concentration Prediction MSC")

    #Testing with samples that have a 0.5mm path length (Can we still use these samples?)
    shortpath_files = sorted(glob.glob(shortpath_folder +"*csv"),key=os.path.getmtime)
    shortpathX, shortpathY, wavelength = getData(shortpath_files, 93, 2002, "_shortpathFreshDMEMFBS", "UV Spectrum Short Path Fresh DMEM w FBS")
    shortpathX_pca = pca.transform(shortpathX)
    conc_pred = pls.predict(shortpathX_pca)
    plotGraphPLS(shortpathY, conc_pred, "_shortpathFreshDMEMFBS_Testing", "NA Concentration Prediction Short Path Fresh DMEM w FBS")
    