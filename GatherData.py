import astropy
import numpy as np
import pandas as pd
import glob
import lightkurve as lk
import matplotlib.pyplot as plt

class GatherData():
    """
    mission, type str: Mission idnetifier options are 'Kepler','K2', or 'TESS'.
    path, type str: path to id data
    """
    def __init__(self,mission='Kepler',path = './data/'):
        self.mission = mission
        self.path_to_data = path+self.mission+'*'
        self.GatherIdData()


    def KeplerDispositionWrapper(self):
        """
        Keppler disposition catagories -> our Wrapper Disposition
        CANDIDATE -> CAND
        FALSE POSITIVE-> FP
        NOT DISPOSITIONED -> UNK
        CONFIRMED - > CONF
        """
        def TempFunc(dispo):
            if(dispo=='CANDIDATE'):
                return 'CAND'
            elif(dispo=='FALSE POSITIVE'):
                return 'FP'
            elif(dispo=='NOT DISPOSITIONED'):
                return 'UNK'
            elif(dispo=='CONFIRMED'):
                return 'CONF'
            else:
                print('ERROR mapping disposition')
                return None
        self.IdData['WrapperDispo'] =list(map(TempFunc,self.IdData['koi_disposition'].values))
    def TESSDispositionWrapper(self):
        """
        TESS disposition catagories -> our Wrapper Disposition:
        APC=ambiguous planetary candidate -> CAND
        CP=confirmed planet -> CONF
        FA=false alarm -> FP
        FP=false positive ->  FP (https://www.cbabelgium.com/peranso/UserGuideHTML/FalseAlarmProbabilityFAP.html)
        KP=known planet -> CONF
        PC=planetary candidate -> CAND
        """

        def TempFunc(dispo):
            if(dispo=='APC' or dispo=='PC'):
                return 'CAND'
            elif(dispo=='FP' or dispo=='FA'):
                return 'FP'
            elif(dispo=='CP' or dispo=='KP'):
                return 'CONF'
            elif(np.isnan(np.array(dispo))):
                return 'UNK'
            else:
                print('ERROR mapping disposition')
                return None
        self.IdData['WrapperDispo'] =list(map(TempFunc,self.IdData['tfopwg_disp'].values))

    def K2DispositionWrapper(self):
        """
        K2 disposition catagories -> our Wrapper Disposition
        CANDIDATE  -> CAND
        FALSE POSITIVE -> FP
        CONFIRMED -> CONF
        """

        def TempFunc(dispo):
            if (dispo == 'CANDIDATE'):
                return 'CAND'
            elif (dispo == 'FALSE POSITIVE'):
                return 'FP'
            elif (dispo == 'CONFIRMED'):
                return 'CONF'
            else:
                print('ERROR mapping disposition')
                return None
        self.IdData['WrapperDispo'] = list(map(TempFunc, self.IdData['disposition'].values))
    def GatherIdData(self):
        """
        This function gathers all data from the requisite mission .csv file and develops an id we call
        'WrapperId' and disposition we call 'WrapperDispo' that use to classify the exoplanet archive
        disposition. We do this to ensure no matter the mission our id and classification methods are
        the same no matter the data set. We choose to set the disposition of all data sets to the
        disposition used in he Kepler data. TESS and K2 have funcitons that map these dispositions
        to kepler disposition.
        """
        file = glob.glob(self.path_to_data)
        if(self.mission =='Kepler'):
            skip=132
        elif(self.mission == 'K2'):
            skip = 51
        elif(self.mission =='TESS'):
            skip = 74
        else:
            print('Please pick a proper mission')
        self.IdData = pd.read_csv(file[0],skiprows=skip) # open csv file to see the first 14 rows of headers.
        if(self.mission =='Kepler'):
            self.IdData['WrapperId'] = self.IdData['kepid']
            self.KeplerDispositionWrapper()
        elif(self.mission=='K2'):
            self.IdData['WrapperId'] = self.IdData['tic_id']
            self.K2DispositionWrapper()
        elif(self.mission=='TESS'):
            self.IdData['WrapperId'] = [f'TIC '+str(val) for val in self.IdData['tid'].values]
            self.TESSDispositionWrapper()
        else:
            print('Error developing WrapperId')
        self.IdData = self.IdData.drop_duplicates(subset='WrapperId',keep='first')
        self.CleanUpData()

    def CleanUpData(self):
        try:
            self.BadIds = np.load(self.mission + 'BlacklistedIDs.npy')
            self.IdData = self.IdData[~self.IdData.WrapperId.isin(self.BadIds)]
        except:
            pass



    def ExcludeK2AuthorError(self, obs):
        """
        Solves issue where lightcurve author is any of the shown below the download does not work
        """
        if(self.mission=='K2'):
            obs = obs[obs.author!= 'K2SC']
            obs= obs[obs.author!='K2VARCAT']
        else:
            pass
        return obs

    def GatherIdLcData(self,id,kind = 'All',quarters=None):
        """
        id, type st:, Use 'WrapperID' string that is located in the self.IdDate frame  .
        kind, type str: can be either 'First' of 'All'. 'First' downloads first obs, and is recomended for testing. 'All' downloads
        all observations and is recomended for implementation.
        """
        try:
            if(quarters!=None):
                all_obs = lk.search_lightcurve(target = str(id),mission = self.mission,quarter=quarters)
            else:
                all_obs = lk.search_lightcurve(target=str(id), mission=self.mission)
        except:
            print('Ensure id corrosponds to the correct mission')
            return None
        all_obs = self.ExcludeK2AuthorError(all_obs)
        if (kind=='All'):
            data = all_obs.download_all()
        elif(kind=='First'):
            data = all_obs.download()
        else:
            print("Please choose a valid 'kind'.")
            return None

        return data





def KeplerTest(ID):
    DataObject = GatherData(mission='Kepler')
    #ID = 10419211
    Dispo =DataObject.IdData['WrapperDispo'][DataObject.IdData['WrapperId'] == ID].values[0]
    print('Kepler Planet Disposition ', Dispo)
    Period = DataObject.IdData['koi_period'][DataObject.IdData['WrapperId'] == ID].values[0]
    Time_startbk = DataObject.IdData['koi_time0bk'][DataObject.IdData['WrapperId'] == ID].values[0]
    ID_Observation = DataObject.GatherIdLcData(id=ID, kind='All')
    ID_Observation = lk.LightCurveCollection.stitch(ID_Observation)
    ID_Observation_Folded = ID_Observation.fold(period=Period, epoch_time=Time_startbk)
    ID_Observation_Folded.plot()
    plt.show()
    return None

def TESSTEst(ID):
    DataObject = GatherData(mission='TESS')
    #ID = DataObject.IdData['WrapperId'][10]
    print(ID)
    print('TESS Planet Disposition ', DataObject.IdData['WrapperDispo'][0])
    Period = DataObject.IdData['pl_orbper'][DataObject.IdData['WrapperId'] == ID].values[0]
    Time_start = DataObject.IdData['pl_tranmid'][DataObject.IdData['WrapperId'] == ID].values[0] - 2457000.0 # in BJD so - 2457000.0 for BTJD
    ID_Observation = DataObject.GatherIdLcData(id=ID, kind='All')
    ID_Observation_Folded = ID_Observation.fold(period=Period, epoch_time=Time_start)
    ID_Observation_Folded.plot()
    plt.savefig('TESS_TEST_ID_'+str(ID)+'.png')
    return None

def K2TEst():
    DataObject = GatherData(mission='K2')
    ID = DataObject.IdData['WrapperId'][1] # 0 does not have a epoch start time
    print('K2 Planet Disposition ', DataObject.IdData['WrapperDispo'][1])
    Period = DataObject.IdData['pl_orbper'][DataObject.IdData['WrapperId'] == ID].values[1]
    Time_start = DataObject.IdData['pl_tranmid'][DataObject.IdData['WrapperId'] == ID].values[1] - 2454833.0 # in BJD so - 2457000.0 for BKJD
    ID_Observation = DataObject.GatherIdLcData(id=ID, kind='All')
    ID_Observation = lk.LightCurveCollection.stitch(ID_Observation)
    ID_Observation_Folded = ID_Observation.fold(period=Period, epoch_time=Time_start)
    ID_Observation_Folded.plot()
    plt.savefig('K2_TEST_ID_'+str(ID)+'.png')
    return None




KeplerTest(10419211)
# if __name__=='__main__':
#     KeplerTest(10419211)
#     #TESSTEst()
#     # K2TEst()