import Model as m
import numpy as np
import GatherData as gd
import glob
import torch
import ExoMinerRawVariation as ExoMinerRaw
import FeatureEngineering as FE

def BinData(ID_Observation_Folded,mission,Nbins=6000):
    min_t = ID_Observation_Folded['time'].min().value
    max_t = ID_Observation_Folded['time'].max().value
    bin_w = (max_t + abs(min_t)) / Nbins
    binned = ID_Observation_Folded.bin(time_bin_size=bin_w, time_bin_start=min_t, n_bins=Nbins)
    try:

        flux = FE.interp_nans(binned['flux'].value)
        centroid_row = FE.interp_nans(binned['centroid_row'].value)
        centroid_col = FE.interp_nans(binned['centroid_col'].value)

    except:
        try:
            flux = FE.interp_nans(binned['flux'].value)
            centroid_row = FE.interp_nans(binned['xic'].value)
            centroid_col = FE.interp_nans(binned['yic'].value)
        except:
            flux = FE.interp_nans(binned['flux'].value)
            centroid_row = FE.interp_nans(binned['sap_x'].value)
            centroid_col = FE.interp_nans(binned['sap_y'].value)

    m = [1 if mission == 'Kepler' else 0]
    centroid_col = np.append(centroid_col,m)
    centroid_row = np.append(centroid_row, m)
    data = {'Flux': flux,
            'Centroid Row': centroid_row,
            'Centroid Col': centroid_col}

    return data

def UnpackRawParamsters(mission,ID):
    data= np.load('./RawData/' + mission + '/' + mission + 'Scalars' + '_load.npy', allow_pickle=True).item()
    return data[ID]

def PredictionData(mission,ID):
    DataObject= gd.GatherData(mission=mission)
    DataObject.IdData = DataObject.IdData[DataObject.IdData.WrapperId ==ID]
    lc_data = DataObject.GatherIdLcData(id=ID,kind='All')
    stiched_lc = FE.StichObservations(lc_data ,mission)
    Params = FE.GatherParams(DataObject, ID)
    ID_Observation_Folded = FE.FoldLightKurveData(stiched_lc, Period=Params['Period[Days]'],
                                               EpochStartTime=Params['FirstEpoch'])
    data = BinData(ID_Observation_Folded,mission, Nbins=6000)
    data_params = UnpackRawParamsters(mission,ID)
    data.update(data_params)
    return data




    #UnpackedRawScalars = YOU WILL HAVE TO WRITE AND MODEL THIS FUNCTION AFTER model.UnpackRawData

def LoadModel(modelpath1 ,modelpath2 ):
    model1 = ExoMinerRaw.ExoMinerRaw()
    model2 = ExoMinerRaw.ExoMinerRaw()
    model1_checkpoint = torch.load(modelpath2)
    model2_checkpoint = torch.load(modelpath1)
    model1.load_state_dict(model1_checkpoint)
    model2.load_state_dict(model2_checkpoint)
    model1.eval()
    model2.eval()
    return model1 ,model2

def Predict(mission,ID):
    data = PredictionData(mission,ID)
    fp1 = glob.glob(r'./All_BestFits/BestFit_Checkpoints/Oversample/Raw/*')[0]
    fp2 =glob.glob(r'./All_BestFits/BestFit_Checkpoints/Undersample/Raw/*')[0]
    model1 , model2 = LoadModel(modelpath1= fp1,
                                modelpath2=fp2)
    output1 = model1(
        flux=m.ReshapeEnsureNotNan(data['Flux'], shape=(1, 1, 6000)),  # flux
        centroid=m.ReshapeEnsureNotNan2D([data['Centroid Row'],data['Centroid Col']], shape=(1, 2, 6001)),
        # centroid data
        f2_2=m.ReshapeEnsureNotNan(data['TransitDepth'], shape=(1, 1)),  # transit depth
        f4_2=m.ReshapeEnsureNotNan(data['CentroidScalars'], shape=(1, 5)),  # centroid scalars
        f6_2=m.ReshapeEnsureNotNan(data['SecondaryScalars'], shape=(1, 4)),  # secondary Scalars
        f7=m.ReshapeEnsureNotNan(data['StellarParamsScalars'], shape=(1, 6)),  # Stellar parameters
        f8=m.ReshapeEnsureNotNan(data['DVDiagnosticScalars'], shape=(1, 6)))  # DV Diagnostics)

    output2 = model2(
        flux=m.ReshapeEnsureNotNan(data['Flux'], shape=(1, 1, 6000)),  # flux
        centroid=m.ReshapeEnsureNotNan2D([data['Centroid Row'], data['Centroid Col']], shape=(1, 2, 6001)),
        # centroid data
        f2_2=m.ReshapeEnsureNotNan(data['TransitDepth'], shape=(1, 1)),  # transit depth
        f4_2=m.ReshapeEnsureNotNan(data['CentroidScalars'], shape=(1, 5)),  # centroid scalars
        f6_2=m.ReshapeEnsureNotNan(data['SecondaryScalars'], shape=(1, 4)),  # secondary Scalars
        f7=m.ReshapeEnsureNotNan(data['StellarParamsScalars'], shape=(1, 6)),  # Stellar parameters
        f8=m.ReshapeEnsureNotNan(data['DVDiagnosticScalars'], shape=(1, 6)) ) # DV Diagnostics


    if(output1 >.5 and output2 >.5):
        return 'Conf'
    else:
        return 'Unk'


if __name__=='__main__':
    #prediciton = Predict(mission = 'Kepler', ID=10811496)
    prediciton = Predict(mission = 'Kepler', ID=10797460
)
#   prediciton = Predict(mission = 'TESS', ID="TIC 369960846")
    print(prediciton)
