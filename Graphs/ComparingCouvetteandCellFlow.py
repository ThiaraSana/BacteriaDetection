import pandas as pd
import numpy as np
from pylab import *
from scipy.optimize import curve_fit

#Plot Comparing Path Length
NA_100_CF = pd.read_csv('/Users/sanaahmed/Desktop/ConferenceGraphs/DataForPLS/FlowCell/Training/FreshDMEM_100NA.csv', header=1700, skipfooter=133, engine='python',names=["Wavelength Sample 1", "Sample 1", "Wavelength Sample 2", "Sample 2", "Wavelength Sample 3", "Sample 3", "NAconc"])
NA_100_CF = NA_100_CF.drop(['Wavelength Sample 2','Wavelength Sample 3', 'NAconc'], axis = 1)
NA_100_CF['Average_Abs_100NA_CF'] = NA_100_CF[['Sample 1', 'Sample 2', 'Sample 3']].mean(axis=1)
NA_100_average_CF = NA_100_CF[['Wavelength Sample 1','Average_Abs_100NA_CF']]

NA_50_CF = pd.read_csv('/Users/sanaahmed/Desktop/ConferenceGraphs/DataForPLS/FlowCell/Training/FreshDMEM_50NA.csv', header=1700, skipfooter=133, engine='python',names=["Wavelength Sample 1", "Sample 1", "Wavelength Sample 2", "Sample 2", "Wavelength Sample 3", "Sample 3", "NAconc"])
NA_50_CF = NA_50_CF.drop(['Wavelength Sample 2','Wavelength Sample 3', 'NAconc'], axis = 1)
NA_50_CF['Average_Abs_50NA_CF'] = NA_50_CF[['Sample 1', 'Sample 2', 'Sample 3']].mean(axis=1)
NA_50_average_CF = NA_50_CF[['Average_Abs_50NA_CF']]

NA_25_CF = pd.read_csv('/Users/sanaahmed/Desktop/ConferenceGraphs/DataForPLS/FlowCell/Training/FreshDMEM_25NA.csv', header=1700, skipfooter=133, engine='python',names=["Wavelength Sample 1", "Sample 1", "Wavelength Sample 2", "Sample 2", "Wavelength Sample 3", "Sample 3", "NAconc"])
NA_25_CF = NA_25_CF.drop(['Wavelength Sample 2','Wavelength Sample 3', 'NAconc'], axis = 1)
NA_25_CF['Average_Abs_25NA_CF'] = NA_25_CF[['Sample 1', 'Sample 2', 'Sample 3']].mean(axis=1)
NA_25_average_CF = NA_25_CF[['Average_Abs_25NA_CF']]

NA_12_5_CF = pd.read_csv('/Users/sanaahmed/Desktop/ConferenceGraphs/DataForPLS/FlowCell/Training/FreshDMEM_12,5NA.csv', header=1700, skipfooter=133, engine='python',names=["Wavelength Sample 1", "Sample 1", "Wavelength Sample 2", "Sample 2", "Wavelength Sample 3", "Sample 3", "NAconc"])
NA_12_5_CF = NA_12_5_CF.drop(['Wavelength Sample 2','Wavelength Sample 3', 'NAconc'], axis = 1)
NA_12_5_CF['Average_Abs_12.5NA_CF'] = NA_12_5_CF[['Sample 1', 'Sample 2', 'Sample 3']].mean(axis=1)
NA_12_5_average_CF = NA_12_5_CF[['Average_Abs_12.5NA_CF']]

NA_6_25_CF = pd.read_csv('/Users/sanaahmed/Desktop/ConferenceGraphs/DataForPLS/FlowCell/Training/FreshDMEM_6,25NA.csv', header=1700, skipfooter=133, engine='python',names=["Wavelength Sample 1", "Sample 1", "Wavelength Sample 2", "Sample 2", "Wavelength Sample 3", "Sample 3", "NAconc"])
NA_6_25_CF = NA_6_25_CF.drop(['Wavelength Sample 2','Wavelength Sample 3', 'NAconc'], axis = 1)
NA_6_25_CF['Average_Abs_6.25NA_CF'] = NA_6_25_CF[['Sample 1', 'Sample 2', 'Sample 3']].mean(axis=1)
NA_6_25_average_CF = NA_6_25_CF[['Average_Abs_6.25NA_CF']]

NA_3_125_CF = pd.read_csv('/Users/sanaahmed/Desktop/ConferenceGraphs/DataForPLS/FlowCell/Training/FreshDMEM_Repeated3,125.csv', header=1700, skipfooter=133, engine='python',names=["Wavelength Sample 1", "Sample 1", "Wavelength Sample 2", "Sample 2", "Wavelength Sample 3", "Sample 3", "NAconc"])
NA_3_125_CF = NA_3_125_CF.drop(['Wavelength Sample 2','Wavelength Sample 3', 'NAconc'], axis = 1)
NA_3_125_CF['Average_Abs_3.125NA_CF'] = NA_3_125_CF[['Sample 1', 'Sample 2', 'Sample 3']].mean(axis=1)
NA_3_125_average_CF = NA_3_125_CF[['Average_Abs_3.125NA_CF']]

NA_0_CF = pd.read_csv('/Users/sanaahmed/Desktop/ConferenceGraphs/DataForPLS/FlowCell/Training/FreshDMEM_Blank1.csv', header=1700, skipfooter=133, engine='python',names=["Wavelength Sample 1", "Sample 1", "Wavelength Sample 2", "Sample 2", "Wavelength Sample 3", "Sample 3", "NAconc"])
NA_0_CF = NA_0_CF.drop(['Wavelength Sample 2','Wavelength Sample 3', 'NAconc'], axis = 1)
NA_0_CF['Average_Abs_0NA_CF'] = NA_0_CF[['Sample 1', 'Sample 2', 'Sample 3']].mean(axis=1)
NA_0_average_CF = NA_0_CF[['Average_Abs_0NA_CF']]

NA_100 = pd.read_csv('/Users/sanaahmed/Desktop/ConferenceGraphs/DataForPLS/Cuvette/Training/100NA_FreshDMEMwFBS.csv', header=1700, skipfooter=133, engine='python',names=["Wavelength Sample 1", "Sample 1", "Wavelength Sample 2", "Sample 2", "Wavelength Sample 3", "Sample 3", "NAconc"])
NA_100 = NA_100.drop(['Wavelength Sample 2','Wavelength Sample 3', 'NAconc'], axis = 1)
NA_100['Average_Abs_100NA'] = NA_100[['Sample 1', 'Sample 2', 'Sample 3']].mean(axis=1)
NA_100_average = NA_100[['Wavelength Sample 1','Average_Abs_100NA']]

NA_50 = pd.read_csv('/Users/sanaahmed/Desktop/ConferenceGraphs/DataForPLS/Cuvette/Training/50NA_FreshDMEMwFBS.csv', header=1700, skipfooter=133, engine='python',names=["Wavelength Sample 1", "Sample 1", "Wavelength Sample 2", "Sample 2", "Wavelength Sample 3", "Sample 3", "NAconc"])
NA_50 = NA_50.drop(['Wavelength Sample 2','Wavelength Sample 3', 'NAconc'], axis = 1)
NA_50['Average_Abs_50NA'] = NA_50[['Sample 1', 'Sample 2', 'Sample 3']].mean(axis=1)
NA_50_average = NA_50[['Average_Abs_50NA']]

NA_25 = pd.read_csv('/Users/sanaahmed/Desktop/ConferenceGraphs/DataForPLS/Cuvette/Training/25NA_FreshDMEMwFBS.csv', header=1700, skipfooter=133, engine='python',names=["Wavelength Sample 1", "Sample 1", "Wavelength Sample 2", "Sample 2", "Wavelength Sample 3", "Sample 3", "NAconc"])
NA_25 = NA_25.drop(['Wavelength Sample 2','Wavelength Sample 3', 'NAconc'], axis = 1)
NA_25['Average_Abs_25NA'] = NA_25[['Sample 1', 'Sample 2', 'Sample 3']].mean(axis=1)
NA_25_average = NA_25[['Average_Abs_25NA']]

NA_12_5 = pd.read_csv('/Users/sanaahmed/Desktop/ConferenceGraphs/DataForPLS/Cuvette/Training/12,5NA_FreshDMEMwFBS.csv', header=1700, skipfooter=133, engine='python',names=["Wavelength Sample 1", "Sample 1", "Wavelength Sample 2", "Sample 2", "Wavelength Sample 3", "Sample 3", "NAconc"])
NA_12_5 = NA_12_5.drop(['Wavelength Sample 2','Wavelength Sample 3', 'NAconc'], axis = 1)
NA_12_5['Average_Abs_12.5NA'] = NA_12_5[['Sample 1', 'Sample 2', 'Sample 3']].mean(axis=1)
NA_12_5_average = NA_12_5[['Average_Abs_12.5NA']]

NA_6_25 = pd.read_csv('/Users/sanaahmed/Desktop/ConferenceGraphs/DataForPLS/Cuvette/Training/6,25NA_FreshDMEMwFBS.csv', header=1700, skipfooter=133, engine='python',names=["Wavelength Sample 1", "Sample 1", "Wavelength Sample 2", "Sample 2", "Wavelength Sample 3", "Sample 3", "NAconc"])
NA_6_25 = NA_6_25.drop(['Wavelength Sample 2','Wavelength Sample 3', 'NAconc'], axis = 1)
NA_6_25['Average_Abs_6.25NA'] = NA_6_25[['Sample 1', 'Sample 2', 'Sample 3']].mean(axis=1)
NA_6_25_average = NA_6_25[['Average_Abs_6.25NA']]

NA_3_125 = pd.read_csv('/Users/sanaahmed/Desktop/ConferenceGraphs/DataForPLS/Cuvette/Training/3,125NA_FreshDMEMwFBS.csv', header=1700, skipfooter=133, engine='python',names=["Wavelength Sample 1", "Sample 1", "Wavelength Sample 2", "Sample 2", "Wavelength Sample 3", "Sample 3", "NAconc"])
NA_3_125 = NA_3_125.drop(['Wavelength Sample 2','Wavelength Sample 3', 'NAconc'], axis = 1)
NA_3_125['Average_Abs_3.125NA'] = NA_3_125[['Sample 1', 'Sample 2', 'Sample 3']].mean(axis=1)
NA_3_125_average = NA_3_125[['Average_Abs_3.125NA']]

NA_0 = pd.read_csv('/Users/sanaahmed/Desktop/ConferenceGraphs/DataForPLS/Cuvette/Training/0NA_FreshDMEMwFBS.csv', header=1700, skipfooter=133, engine='python',names=["Wavelength Sample 1", "Sample 1", "Wavelength Sample 2", "Sample 2", "Wavelength Sample 3", "Sample 3", "NAconc"])
NA_0 = NA_0.drop(['Wavelength Sample 2','Wavelength Sample 3', 'NAconc'], axis = 1)
NA_0['Average_Abs_0NA'] = NA_0[['Sample 1', 'Sample 2', 'Sample 3']].mean(axis=1)
NA_0_average = NA_0[['Average_Abs_0NA']]

CuvettePrediction = pd.read_csv('/Users/sanaahmed/Desktop/ConferenceGraphs/DataForPLS/PredictedResultsCSV/predresults_cuvette.csv',names=["Predicted", "Actual"])
CuvettePredicted = CuvettePrediction['Predicted'].values
CuvetteActual = CuvettePrediction['Actual'].values
ComparingCouvette = pd.concat([NA_100_average, NA_50_average, NA_25_average, NA_12_5_average, NA_6_25_average, NA_3_125_average, NA_0_average], axis=1).reindex(NA_100_average_CF.index)

FlowCellPrediction = pd.read_csv('/Users/sanaahmed/Desktop/ConferenceGraphs/DataForPLS/PredictedResultsCSV/predresults_flowcell.csv',names=["Predicted", "Actual"])
FlowCellPredicted = FlowCellPrediction['Predicted'].values
FlowCellActual = FlowCellPrediction['Actual'].values
ComparingFlowCell = pd.concat([NA_100_average_CF, NA_50_average_CF, NA_25_average_CF, NA_12_5_average_CF, NA_6_25_average_CF, NA_3_125_average_CF, NA_0_average_CF], axis=1).reindex(NA_100_average_CF.index)


def func(x, a, b):
    return a * x + b
popt_Cuvette, pcov_Cuvette = curve_fit(func, CuvetteActual , CuvettePredicted)
residuals_Cuvette = CuvettePredicted- func(CuvetteActual, *popt_Cuvette)
ss_res_Cuvette = np.sum(residuals_Cuvette**2)
ss_tot_Cuvette = np.sum((CuvettePredicted-np.mean(CuvettePredicted))**2)
r_squared_Cuvette = 1 - (ss_res_Cuvette / ss_tot_Cuvette)
print(r_squared_Cuvette)

popt_FlowCell, pcov_FlowCell = curve_fit(func, FlowCellActual , FlowCellPredicted)
residuals_FlowCell = FlowCellPredicted- func(FlowCellActual, *popt_FlowCell)
ss_res_FlowCell = np.sum(residuals_FlowCell**2)
ss_tot_FlowCell = np.sum((FlowCellPredicted-np.mean(FlowCellPredicted))**2)
r_squared_FlowCell = 1 - (ss_res_FlowCell / ss_tot_FlowCell)
print(r_squared_FlowCell)

ComparingFlowCell['Average_Abs_100NA_CF'] = ComparingFlowCell['Average_Abs_100NA_CF'].div(2.8)
ComparingFlowCell['Average_Abs_50NA_CF'] = ComparingFlowCell['Average_Abs_50NA_CF'].div(2.8)
ComparingFlowCell['Average_Abs_25NA_CF'] = ComparingFlowCell['Average_Abs_25NA_CF'].div(2.8)
ComparingFlowCell['Average_Abs_12.5NA_CF'] = ComparingFlowCell['Average_Abs_12.5NA_CF'].div(2.8)
ComparingFlowCell['Average_Abs_6.25NA_CF'] = ComparingFlowCell['Average_Abs_6.25NA_CF'].div(2.8)
ComparingFlowCell['Average_Abs_3.125NA_CF'] = ComparingFlowCell['Average_Abs_3.125NA_CF'].div(2.8)
ComparingFlowCell['Average_Abs_0NA_CF'] = ComparingFlowCell['Average_Abs_0NA_CF'].div(2.8)

ComparingCouvette['Average_Abs_100NA'] = ComparingCouvette['Average_Abs_100NA'].div(2)
ComparingCouvette['Average_Abs_50NA'] = ComparingCouvette['Average_Abs_50NA'].div(2)
ComparingCouvette['Average_Abs_25NA'] = ComparingCouvette['Average_Abs_25NA'].div(2)
ComparingCouvette['Average_Abs_12.5NA'] = ComparingCouvette['Average_Abs_12.5NA'].div(2)
ComparingCouvette['Average_Abs_6.25NA'] = ComparingCouvette['Average_Abs_6.25NA'].div(2)
ComparingCouvette['Average_Abs_3.125NA'] = ComparingCouvette['Average_Abs_3.125NA'].div(2)
ComparingCouvette['Average_Abs_0NA'] = ComparingCouvette['Average_Abs_0NA'].div(2)

fig = plt.figure(constrained_layout=True, figsize=(28, 10), dpi=80)

gs1 = fig.add_gridspec(nrows=1, ncols=5)
axs2 = fig.add_subplot(gs1[0, :-2])
axs3 = fig.add_subplot(gs1[0, -2:])

axs2.plot(ComparingFlowCell['Wavelength Sample 1'], ComparingFlowCell['Average_Abs_100NA_CF'], label = '100 ug/ml NA Flow Cell', color = "crimson")
axs2.plot(ComparingCouvette['Wavelength Sample 1'], ComparingCouvette['Average_Abs_100NA'], "--", label = '100 ug/ml NA Cuvette', color = "crimson")
axs2.plot(ComparingFlowCell['Wavelength Sample 1'], ComparingFlowCell['Average_Abs_50NA_CF'], label = '50 ug/ml NA Flow Cell', color = "royalblue")
axs2.plot(ComparingCouvette['Wavelength Sample 1'], ComparingCouvette['Average_Abs_50NA'], "--", label = '50 ug/ml NA Cuvette', color = "royalblue")
axs2.plot(ComparingFlowCell['Wavelength Sample 1'], ComparingFlowCell['Average_Abs_25NA_CF'], label = '25 ug/ml NA Flow Cell', color = "orange")
axs2.plot(ComparingCouvette['Wavelength Sample 1'], ComparingCouvette['Average_Abs_25NA'], "--", label = '25 ug/ml NA Cuvette', color = "orange")
axs2.plot(ComparingFlowCell['Wavelength Sample 1'], ComparingFlowCell['Average_Abs_12.5NA_CF'], label = '12.5 ug/ml NA Flow Cell', color = "seagreen")
axs2.plot(ComparingCouvette['Wavelength Sample 1'], ComparingCouvette['Average_Abs_12.5NA'], "--", label = '12.5 ug/ml NA Cuvette', color = "seagreen")
axs2.plot(ComparingFlowCell['Wavelength Sample 1'], ComparingFlowCell['Average_Abs_6.25NA_CF'], label = '6.25 ug/ml NA Flow Cell', color = "mediumblue")
axs2.plot(ComparingCouvette['Wavelength Sample 1'], ComparingCouvette['Average_Abs_6.25NA'], "--", label = '6.25 ug/ml NA Cuvette', color = "mediumblue")
axs2.plot(ComparingFlowCell['Wavelength Sample 1'], ComparingFlowCell['Average_Abs_3.125NA_CF'], label = '3.125 ug/ml NA Flow Cell', color = "blueviolet")
axs2.plot(ComparingCouvette['Wavelength Sample 1'], ComparingCouvette['Average_Abs_3.125NA'], "--", label = '3.125 ug/ml NA Cuvette', color = "blueviolet")
axs2.plot(ComparingFlowCell['Wavelength Sample 1'], ComparingFlowCell['Average_Abs_0NA_CF'], label = '0 ug/ml NA Flow Cell', color = "darkolivegreen")
axs2.plot(ComparingCouvette['Wavelength Sample 1'], ComparingCouvette['Average_Abs_0NA'], "--", label = '0 ug/ml NA Cuvette', color = "darkolivegreen")

axs2.set_title('Comparing UV Vis Spectrums Obtained from Cuvette and Flow Cell', fontsize = 20)
axs2.legend(loc='best', fontsize = 20)
axs2.set_xlabel('Wavelength [nm]', fontsize = 20)
axs2.set_ylabel('Absorbance/mm', fontsize = 20)
# axs2.xlim((240, 350))

axs3.plot(CuvettePrediction['Actual'], CuvettePrediction['Predicted'],'.', label = 'Cuvette', color = "crimson")
axs3.plot(CuvettePredicted, func(CuvettePredicted, *popt_Cuvette), color = "crimson", alpha = 0.7)
axs3.text(10, 48, r'$r^2 = 0.9930$', fontsize=15, color = "crimson",   fontweight='bold', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
axs3.plot(FlowCellPrediction['Actual'], FlowCellPrediction['Predicted'],'x', label = 'Flow Cell', color = "blue")
axs3.plot(FlowCellPredicted, func(FlowCellPredicted, *popt_FlowCell), alpha = 0.7, color = "blue")
axs3.text(20, 8, r'$r^2 = 0.9996$', fontsize=15, color = "blue",   fontweight='bold', bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})

axs3.set_title('NA Concentration Prediction of Spiked Fresh DMEM with FBS', fontsize = 20)
axs3.legend(loc='best', fontsize = 20)
axs3.set_xlabel('Actual Concentration (ug/ml)', fontsize = 20)
axs3.set_ylabel('Predicted Concentration (ug/ml)', fontsize = 20)

savefig('/Users/sanaahmed/Desktop/Combined.png')
