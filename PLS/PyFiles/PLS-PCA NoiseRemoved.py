import glob
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/sanaahmed/Desktop/mb-master')
import mb
from pylab import *
import scipy.signal
from sklearn import decomposition
from sklearn.decomposition import PCA

np.set_printoptions(threshold=sys.maxsize)

pls_training_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Training_FreshDMEMFBSNoPS_0601/'
testingspikeddata_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_SpikedFreshDMEMFBSNoPS_0624/'
#Testing with ec culture (Is the complexity of the bacteria culture problem solved with PCA?)
ec_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_BacCulture_0525/ec/'
#Testing with Fresh DMEM spiked with NA at low concentration and a different pH (Is detection at a low concentration of NA possible and does pH matter like hypothesized?)
oldsample_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Old_FreshDMEMFBSNoPS_0601/'
#Testing with other types of bacteria cultured with varying CFUs (Does the type of bacteria matter?)
ab_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_BacCulture_0525/ab/'
pa_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_BacCulture_0525/pa/'
sa_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_BacCulture_0525/sa/'
se_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_BacCulture_0525/se/'
kp_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_BacCulture_0525/kp/'
all_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_BacCulture_0525/all/'
#Testing with MSC cultures (should not have any NA but will have NAM and other metabolites, do these affect the PLS?)
msc_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_MSC_0525/'
#Testing with samples that have a 0.5mm path length (Can we still use these samples?)
shortpath_folder = '/Users/sanaahmed/Desktop/test/Data_0.5mm/Training_FreshDMEMFBSNoPS/'
#Infected MSC Culture
infectedMSC_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_BacCulture_0615/'
InfectedMSCs_0629_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_InfectedMSCs_0629/'
InfectedMSCs_0629_diluted_folder = '/Users/sanaahmed/Desktop/test/Data_2mm/Testing_InfectedMSCs_0629_diluted/'


def getData_forML(path, files, footer, header, j, l, m, p):
    data = []
    na_conc=[]
    label = []
    wave = []
    k=1
    
    for kk in range(0,len(files),m):

        for kkk in range(m):

            a=np.genfromtxt(files[kk+kkk], skip_header=header, delimiter=',', skip_footer=footer)
            
            conc_na = a[:,6][0]
            w1 = a[:,0]
            a1 = a[:,1]
            a2 = a[:,3]
            a3 = a[:,5]
            b1 = (a1 + a2 + a3)/3

            plot(w1, b1)
            plt.show
            
            na_conc.append(conc_na)
            label.append(conc_na)
            data.append(b1)
            wave.append(w1)

            k +=1 
        
    data1 = np.array(data)
    na_conc1 = np.array(na_conc)
    wavelength = np.array(wave)

    xlabel("wavelength [nm]")
    ylabel("Absorbance")
    title(l)
    legend(label, fontsize = 'small', fancybox = True, title= p)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/PCAwAllBac/UVvis/UVvis' + j + '.png')
    # plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/UVvis/UVvis' + j + '.png')
    plt.clf()

    return data1, na_conc1, wavelength

def getData_forML_filtered(path, files, footer, header, j, l, m, p):
    data = []
    na_conc=[]
    label = []
    wave = []
    k=1
    
    for kk in range(0,len(files),m):

        for kkk in range(m):

            a=np.genfromtxt(files[kk+kkk], skip_header=header, delimiter=',', skip_footer=footer)
            
            conc_na = a[:,6][0]
            w1 = a[:,0]
            a1 = a[:,1]
            a2 = a[:,3]
            a3 = a[:,5]
            b1 = (a1 + a2 + a3)/3
            
            noise_data = scipy.signal.savgol_filter(b1, 51, 3)

            plot(w1, noise_data)
            plt.show
            
            na_conc.append(conc_na)
            label.append(conc_na)
            data.append(noise_data)
            wave.append(w1)

            k +=1 
        
    data1 = np.array(data)
    na_conc1 = np.array(na_conc)
    wavelength = np.array(wave)

    xlabel("wavelength [nm]")
    ylabel("Absorbance")
    title(l)
    legend(label, fontsize = 'small', fancybox = True, title= p)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/PCAwAllBac/UVvis/UVvis' + j + '.png')
    # plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/UVvis/UVvis' + j + '.png')
    plt.clf()

    return data1, na_conc1, wavelength

def SetPCA(absorbance):
    absorbance_centered = absorbance - absorbance.mean(axis=0)
    pca = PCA(n_components= 3, svd_solver='auto')
    pca.fit(absorbance_centered)
    absorbance_pca = pca.transform(absorbance_centered)
    print("original shape:   ", absorbance_centered.shape)
    print("transformed shape:", absorbance_pca.shape)

    # X_new = pca.inverse_transform(absorbance_pca)
    # plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.2)
    # plt.scatter(absorbance_centered[:, 0], absorbance_centered[:, 1], alpha=0.8)
    # plt.axis('equal'); 
    # show()

    loadings = pca.components_.T*np.sqrt(pca.explained_variance_)
    loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3'])
    plot(loading_matrix)
    xlabel("wavelength datapoints")
    ylabel("Correlation")
    title('PCA Loading Matrix')
    legend(['PC1', 'PC2', 'PC3'], fontsize = 'small', fancybox = True)
    
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/PCAwAllBac/xloadings/' + 'PCALoadingMatrix.png')
    # plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/xloadings/' + 'PCALoadingMatrix.png')
    plt.clf()

    return absorbance_pca, pca

def plotGraphPLS(actual, predicted, j, l):
    # figure('PLS Graph', figsize=(6,6), dpi=80)
    plot(actual,'+', label = 'Actual NA Concentration')
    plot(predicted,'.', label = 'Predicted NA Concentration', color = "crimson")
    xlabel("Sample Number", fontsize = 12)
    ylabel("Concentration (ug/ml)", fontsize = 12)
    legend(loc='upper right', fontsize =  'x-large')
    title(l, fontsize = 12)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/PCAwAllBac/PLS/PLSprediction'+ j +'.png')
    # plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/PLS/PLSprediction'+ j +'.png')
    plt.clf()

def plotGraphTraining8(predicted, label, l, position, heading):
    plot(position[0], predicted[0], '.')
    plot(position[1], predicted[1], '.')
    plot(position[2], predicted[2], '.')
    plot(position[3], predicted[3], '.')
    plot(position[4], predicted[4], '.')
    plot(position[5], predicted[5], '.')
    plot(position[6], predicted[6], '.')
    plot(position[7], predicted[7], '.')

    legend(label, fontsize = 'small', fancybox = True, title= 'CFU')
    title(heading)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/PCAwAllBac/PLS/PLSprediction'+ l +'.png')
    # plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/PLS/PLSprediction'+ l +'.png')
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
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/PCAwAllBac/PLS/PLSprediction'+ l +'.png')
    # plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/PLS/PLSprediction'+ l +'.png')
    plt.clf()

def plotGraphTraining6(predicted, label, l, position, heading):
    plot(position[0], predicted[0], '.')
    plot(position[1], predicted[1], '.')
    plot(position[2], predicted[2], '.')
    plot(position[3], predicted[3], '.')
    plot(position[4], predicted[4], '.')
    plot(position[5], predicted[5], '.')

    legend(label, fontsize = 'small', fancybox = True, title= 'CFU')
    title(heading)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/PCAwAllBac/PLS/PLSprediction'+ l +'.png')
    # plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/PLS/PLSprediction'+ l +'.png')
    plt.clf()


def plotGraphTraining4(predicted, label, l, position, heading):
    plot(position[0], predicted[0], '.')
    plot(position[1], predicted[1], '.')
    plot(position[2], predicted[2], '.')
    plot(position[3], predicted[3], '.')

    legend(label, fontsize = 'small', fancybox = True, title= 'MSC')
    title(heading)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/PCAwAllBac/PLS/PLSprediction'+ l +'.png')
    # plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/PLS/PLSprediction'+ l +'.png')
    plt.clf()

def plotGraphTraining3(predicted, label, l, position, heading):
    plot(position[0], predicted[0], '.')
    plot(position[1], predicted[1], '.')
    plot(position[2], predicted[2], '.')

    legend(label, fontsize = 'small', fancybox = True, title= 'MSC')
    title(heading)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/PCAwAllBac/PLS/PLSprediction'+ l +'.png')
    # plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/PLS/PLSprediction'+ l +'.png')
    plt.clf()

if __name__ == "__main__":
    # #Testing with ec culture (Is the complexity of the bacteria culture problem solved with PCA?)
    # testing_files_bac = sorted(glob.glob(ec_folder +"*csv"),key=os.path.getmtime)
    # testX_bac, testY_bac, wavelength_bac = getData_forML(ec_folder, testing_files_bac, 173, 2002, "_ec", "UV Vis Spectrum ec", 7, 'CFU')
    # pca_train, pca = SetPCA(testX_bac)

    # # Testing with all bacteria cultures (Is the complexity of the bacteria culture problem solved with PCA?)
    testing_files_bac = sorted(glob.glob(all_folder +"*csv"),key=os.path.getmtime)
    testX_bac, testY_bac, wavelength_bac = getData_forML(all_folder, testing_files_bac, 173, 2002, "_all", "UV Vis Spectrum All Bacteria", 23, 'CFU')
    pca_train, pca = SetPCA(testX_bac)
    
    #PLS Training
    training_files = sorted(glob.glob(pls_training_folder+"*csv"),key=os.path.getmtime)
    trainX, trainY, wavelength = getData_forML(pls_training_folder, training_files, 173, 2002, "_FreshDMEMFBS", "UV Spectrum Fresh DMEM w FBS", 10, 'ug/ml')
    trainX_pca = pca.transform(trainX)
    pls,sc = mb.raman.pls_x(trainX_pca, trainY, n_components=3)
    train_predicted = pls.predict(trainX_pca)
    plot(pls.x_loadings_)
    plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/PCAwAllBac/xloadings/' + 'PLSLoadings.png')
    # plt.savefig('/Users/sanaahmed/Desktop/test/Graphs_2mm/PLS/NoiseRemoved/xloadings/' + 'PLSLoadings.png')
    plt.clf()
    plotGraphPLS(trainY, train_predicted, "_FreshDMEMFBS_Training", "NA Concentration Prediction Fresh DMEM w FBS")

    #Testing with Fresh DMEM spiked with NA at low concentration and a different pH (Is detection at a low concentration of NA possible and does pH matter like hypothesized?)
    testing_files = sorted(glob.glob(oldsample_folder +"*csv"),key=os.path.getmtime)
    testX, testY, wavelength = getData_forML(oldsample_folder, testing_files, 173, 2002, "_oldFreshDMEMFBS", "UV Spectrum old Fresh DMEM w FBS", 3, 'ug/ml')
    testX_pca = pca.transform(testX)
    conc_pred = pls.predict(testX_pca)
    plotGraphPLS(testY, conc_pred, "_oldFreshDMEMFBS_Testing", "NA Concentration Prediction Old Fresh DMEM w FBS")

    #Testing with other types of bacteria cultured with varying CFUs (Does the type of bacteria matter?)
    ec_files = sorted(glob.glob(ec_folder +"*csv"),key=os.path.getmtime)
    ecX, ecY, wavelength = getData_forML(ec_folder, ec_files, 173, 2002, "_ec", "UV Spectrum ec", 7, 'CFU')
    ecX_pca = pca.transform(ecX)
    conc_pred = pls.predict(ecX_pca)
    position = [1, 2, 3, 4, 5, 6, 7]
    plotGraphTraining7(conc_pred, ecY, "_ec_Testing", position, "Predicted NA Concentration ec")

    ab_files = sorted(glob.glob(ab_folder +"*csv"),key=os.path.getmtime)
    abX, abY, wavelength = getData_forML(ab_folder, ab_files, 173, 2002, "_ab", "UV Spectrum ab", 7, 'CFU')
    abX_pca = pca.transform(abX)
    conc_pred = pls.predict(abX_pca)
    position = [1, 2, 3, 4, 5, 6, 7]
    plotGraphTraining7(conc_pred, abY, "_ab_Testing", position, "Predicted NA Concentration ab")

    pa_files = sorted(glob.glob(pa_folder +"*csv"),key=os.path.getmtime)
    paX, paY, wavelength = getData_forML(pa_folder, pa_files, 173, 2002, "_pa", "UV Spectrum pa", 7, 'CFU')
    paX_pca = pca.transform(paX)
    conc_pred = pls.predict(paX_pca)
    position = [1, 2, 3, 4, 5, 6, 7]
    plotGraphTraining7(conc_pred, paY, "_pa_Testing", position, "Predicted NA Concentration pa")

    sa_files = sorted(glob.glob(sa_folder +"*csv"),key=os.path.getmtime)
    saX, saY, wavelength = getData_forML(sa_folder, sa_files, 173, 2002, "_sa", "UV Spectrum sa", 7, 'CFU')
    saX_pca = pca.transform(saX)
    conc_pred = pls.predict(saX_pca)
    position = [1, 2, 3, 4, 5, 6, 7]
    plotGraphTraining7(conc_pred, saY, "_sa_Testing", position, "Predicted NA Concentration sa")

    se_files = sorted(glob.glob(se_folder +"*csv"),key=os.path.getmtime)
    seX, seY, wavelength = getData_forML(se_folder, se_files, 173, 2002, "_se", "UV Spectrum se", 7, 'CFU')
    seX_pca = pca.transform(seX)
    conc_pred = pls.predict(seX_pca)
    position = [1, 2, 3, 4, 5, 6, 7]
    plotGraphTraining7(conc_pred, seY, "_se_Testing", position, "Predicted NA Concentration se")

    kp_files = sorted(glob.glob(kp_folder +"*csv"),key=os.path.getmtime)
    kpX, kpY, wavelength = getData_forML(kp_folder, kp_files, 173, 2002, "_kp_Testing", "UV Spectrum kp", 7, 'CFU')
    kpX_pca = pca.transform(kpX)
    conc_pred = pls.predict(kpX_pca)
    position = [1, 2, 3, 4, 5, 6, 7]
    plotGraphTraining7(conc_pred, kpY, "_kp_Testing", position, "Predicted NA Concentration kp")

    infected_files = sorted(glob.glob(infectedMSC_folder +"*csv"),key=os.path.getmtime)
    infectedX, infectedY, wavelength = getData_forML(infectedMSC_folder, infected_files, 173, 2002, "_infected_Testing", "UV Spectrum Infected MSC Cultures", 8, 'CFU')
    infectedX_pca = pca.transform(infectedX)
    conc_pred = pls.predict(infectedX_pca)
    position = [1, 2, 3, 4, 5, 6, 7, 8]
    plotGraphTraining8(conc_pred, infectedY, "_infected_Testing", position, "Predicted NA Concentration Infected MSC Cultures")

    # #Testing with MSC cultures (should not have any NA but will have NAM and other metabolites, do these affect the PLS?)
    msc_files = sorted(glob.glob(msc_folder +"*csv"),key=os.path.getmtime)
    mscX, mscY, wavelength = getData_forML(msc_folder, msc_files, 173, 2002, "_MSC", "UV Spectrum MSC", 4, 'MSCs')
    mscX_pca = pca.transform(mscX)
    conc_pred = pls.predict(mscX_pca)
    position = [1, 2, 3, 4]
    plotGraphTraining4(conc_pred, mscY, "_MSC_Testing", position, "NA Concentration Prediction MSC")

    #Testing with samples that have a 0.5mm path length (Can we still use these samples?)
    shortpath_files = sorted(glob.glob(shortpath_folder +"*csv"),key=os.path.getmtime)
    shortpathX, shortpathY, wavelength = getData_forML(shortpath_folder, shortpath_files, 173, 2002, "_shortpathFreshDMEMFBS", "UV Spectrum Short Path Fresh DMEM w FBS", 19, 'ug/ml')
    shortpathX_pca = pca.transform(shortpathX)
    conc_pred = pls.predict(shortpathX_pca)
    plotGraphPLS(shortpathY, conc_pred, "_shortpathFreshDMEMFBS_Testing", "NA Concentration Prediction Short Path Fresh DMEM w FBS")

    spiked_files = sorted(glob.glob(testingspikeddata_folder +"*csv"),key=os.path.getmtime)
    spikedX, spikedY, wavelength = getData_forML(testingspikeddata_folder, spiked_files, 173, 2002, "_spiked", "UV Spectrum Fresh DMEM w FBS", 6, 'ug/ml')
    spikedX_pca = pca.transform(spikedX)
    conc_pred = pls.predict(spikedX_pca)
    plotGraphPLS(spikedY, conc_pred, "_spikedFreshDMEMwFBS_Testing", "NA Concentration Prediction of Fresh DMEM w FBS spiked with NA")
    
    InfectedMSCs_0629_files = sorted(glob.glob(InfectedMSCs_0629_folder +"*csv"),key=os.path.getmtime)
    infectedX, infectedY, wavelength = getData_forML(InfectedMSCs_0629_folder, InfectedMSCs_0629_files, 173, 2002, "_InfectedMSCs_0629_Testing", "UV Spectrum Infected MSCs 0629", 6, 'CFU')
    infectedX_pca = pca.transform(infectedX)
    conc_pred = pls.predict(infectedX_pca)
    position = [1, 2, 3, 4, 5, 6]
    plotGraphTraining6(conc_pred, infectedY, "_InfectedMSCs_0629_Testing", position, "Predicted NA Concentration Infected MSCs 0629")
    print(conc_pred)

    InfectedMSCs_0629_diluted_files = sorted(glob.glob(InfectedMSCs_0629_diluted_folder +"*csv"),key=os.path.getmtime)
    infecteddilX, infecteddilY, wavelength = getData_forML(InfectedMSCs_0629_diluted_folder, InfectedMSCs_0629_diluted_files, 173, 2002, "_InfectedMSCs_0629_diluted_Testing", "UV Spectrum Diluted Infected MSCs 0629", 6, 'CFU')
    infecteddilX_pca = pca.transform(infecteddilX)
    conc_pred = pls.predict(infecteddilX_pca)
    position = [1, 2, 3, 4, 5, 6]
    plotGraphTraining6(conc_pred, infecteddilY, "_InfectedMSCs_0629_diluted_Testing", position, "Predicted NA Concentration Diluted Infected MSCs 0629")
    print(conc_pred)