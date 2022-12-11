import numpy as np
#import ExoMinerArch as ExoMiner

import torch.nn as nn
import FeatureEngineering as FE
import GatherData as gd
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import glob

#import MyDataSet as D
#import RawDataSet as R
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.classification import BinaryPrecision, BinaryRecall,BinaryAccuracy,BinaryStatScores,Precision , BinaryAUROC,BinaryF1Score
import ExoMinerRawVariation as ExoMinerRaw
def GenRandomTensor(size):
    return torch.rand(size)
def RandTestModel(seed=None):
    if (seed == None):
        pass
    else:
        torch.manual_seed(seed)
    f1 = GenRandomTensor((1, 1, 301))  # feature 1 input data
    f2 = GenRandomTensor((1, 1, 31))  # feature 2 input data
    f2_2 = GenRandomTensor((1, 1))  # feature 2 transit depth
    f3 = GenRandomTensor((1, 1, 301))  # feature 3 input data
    f4 = GenRandomTensor((1, 1, 31))  # feature 4 input data
    f4_2 = GenRandomTensor((1, 5))  # feature 4 centroid scalars
    f5_odd = GenRandomTensor((1, 1, 31))  # feature 5 odd input data
    f5_even = GenRandomTensor((1, 1, 31))  # feature 5 even input data
    f6 = GenRandomTensor((1, 1, 31))  # feature 6 input data
    f6_2 = GenRandomTensor((1, 4))  # feature 6 secondary scalars
    f7 = GenRandomTensor((1, 6))  # feature 7 stellar parameters
    f8 = GenRandomTensor((1, 6))  # feature 7 DV Diagnostic tests
    model = ExoMiner.ExoMiner()
    out = model.forward(f1, f2, f2_2, f3, f4, f4_2, f5_odd, f5_even, f6, f6_2, f7, f8)
    return out
def GenerateFeatures(ID, mish, Plot):
    Data = FE.Features(mish=mish, max_number_of_obs=3, Chosen_ID=ID, Plot=Plot)
    return Data
def Test():
    out = RandTestModel(1)
    print(out)
    return None
def GetRandomID(DataObject, label='CAND'):
    ValidCriteria = DataObject.IdData['WrapperDispo'][DataObject.IdData['WrapperDispo'] == label]
    idx = np.random.choice(ValidCriteria.index.values)
    return idx

def ReshapeEnsureNotNan(Feature, shape,nan=0):
    Feature[Feature==0] = nan
    Feature[Feature < 0]= nan
    return torch.tensor(np.nan_to_num(Feature,nan = nan)).reshape(shape).float()
def ReshapeEnsureNotNan2D(Feature,shape,nan=0):
    f1 = ReshapeEnsureNotNan(Feature[0], (shape[0],1,shape[2]),nan=nan)
    f2 = ReshapeEnsureNotNan(Feature[1], (shape[0], 1,shape[2]), nan=nan)
    return torch.cat((f1, f2), 1)
def UnPackData(mission,fp='FeatureData/'):
    files = glob.glob('./'+fp+mission+'/'+mission+r'Features_*.npy')
    idxs = ['load' not in file for file in files]
    f = []
    for i in range (0,len(idxs)):
        if(idxs[i]):
            f.append(files[i])
        else:
            pass
    files = f
    files.sort()
    data_return = {}
    for file in files:
        shutil.copyfile(file , file[:-4]+'_load.npy')
        data = np.load(file[:-4]+'_load.npy', allow_pickle=True).item()
        for id in data.keys():
            data_return[id] = data[id]
    return data_return

def DispoFunc(dispo):
    if(dispo=='CAND' or dispo=='UNK'):
        return .5
    elif(dispo =='CONF'):
        return 1
    else:
        return 0

def ReduceSetForBatchSize(data,batch_size,set_type):
    #r = len(data) % batch_size
    #data = data.sample(n=int(len(data)-r),random_state = 1)
    #print('Removing {} ids from {} to use batch size of {}'.format(r,set_type,batch_size))
    #data = data.sample(n=int(len(data) - 0), random_state=1)
    return data


def OverSample(CPs ,FPs):
    # oversample
    len_cp =CPs.__len__()
    len_fp = FPs.__len__()
    if(len_cp <=len_fp):
        df_small=CPs
        df_large=FPs
        len_small = df_small.__len__()
        len_large = df_large.__len__()
        ret =1
    else:
        df_small  = FPs
        df_large  = CPs
        len_small = df_small.__len__()
        len_large = df_large.__len__()
        ret = 2
    len_delta = int(len_large - len_small)
    df_small_temp = df_small.sample(n=len_delta , random_state=1 , replace=True)
    df_small_new = pd.concat([df_small,df_small_temp]).sample(frac=1, random_state=1, replace=False)
    assert df_small_new.index.nunique() ==df_small.index.nunique()
    assert df_small_new.__len__() == df_large.__len__()
    if(ret==1):
        # small = CP
        return df_small_new ,df_large
    else:
        # large = CP
        return df_large,df_small_new

def UnderSample(CPs,FPs):
    num_conf = np.floor(CPs.__len__())
    num_false = np.floor(FPs.__len__())
    num_conf = int(num_conf)
    num_false = int(num_false)
    n = min(num_conf, num_false)
    CPs = CPs.sample(n=n, random_state=1)
    FPs = FPs.sample(n=n, random_state=1)
    return CPs,FPs

def ClassImbalance(CPs,FPs,type_sample='Oversample'):

    if(type_sample!='None'):
        if(type_sample=='Oversample'):
            CPs,FPs = OverSample(FPs, CPs)
        elif(type_sample=='Undersample'):
            CPs, FPs = UnderSample(FPs, CPs)
    else:
        pass
    return  CPs,FPs

def GetTrainingAndValidationSets(CPs,FPs,training_size):
    CPs_for_training = CPs.sample(frac=training_size, random_state=1, replace=False)
    FPs_for_training = FPs.sample(frac=training_size, random_state=1, replace=False)
    #

    indexs_in_CP_train = CPs_for_training.index
    indexs_in_FP_train = FPs_for_training.index
    #
    CP_bool_array = ~CPs.index.isin(indexs_in_CP_train)
    FP_bool_array = ~FPs.index.isin(indexs_in_FP_train)
    #
    CPs_for_validation = CPs[CP_bool_array]
    FPs_for_validation = FPs[FP_bool_array]
    return CPs_for_training ,FPs_for_training ,CPs_for_validation ,FPs_for_validation

def GrabDatasets(data,DataObject,training_size,batch_size,Combo = True,type_sample='Oversample',Raw=False):
    df = pd.DataFrame(data).T
    Ids = DataObject.IdData['WrapperId']
    Dispo = DataObject.IdData['WrapperDispo']
    Dispo = pd.DataFrame(Dispo.values, columns=['Disposition'])
    Ids = pd.DataFrame(Ids.values, columns=['ID'])
    df2 = pd.concat([Dispo, Ids], axis=1)
    df2 = df2.set_index('ID')
    df = pd.concat([df, df2], axis=1)
    targets = pd.DataFrame([DispoFunc(x) for x in df.Disposition.values], columns=['Target'], index=df.index)
    df = pd.concat([df, targets], axis=1)
    known_set = df[df['Target'] != .5]
    unk_set = df[df['Target']==.5]
    CPs = known_set[known_set['Target']==1]
    FPs = known_set[known_set['Target']==0]
    CPs_for_training ,FPs_for_training ,CPs_for_validation ,FPs_for_validation =GetTrainingAndValidationSets(CPs, FPs, training_size)
    CPs_for_training, FPs_for_training = ClassImbalance(CPs_for_training, FPs_for_training,type_sample=type_sample)
    train_data = pd.concat([CPs_for_training,FPs_for_training]).sample(frac=1,random_state=1)
    #train_data = ReduceSetForBatchSize(train_data,batch_size,set_type='Training data')
    valid_data = pd.concat([CPs_for_validation, FPs_for_validation]).sample(frac=1, random_state=1)
    #valid_data = ReduceSetForBatchSize(valid_data, batch_size,set_type='Validation data')
    #unk_set = ReduceSetForBatchSize(unk_set, batch_size, set_type='Candidate data')
    if(Raw==False):
        TrainDataset = D.MyDataset(train_data)
        ValidDataset = D.MyDataset(valid_data)
        CandDataset = D.MyDataset(unk_set)
    elif(Raw==True):
        TrainDataset = R.RawDataset(train_data)
        ValidDataset = R.RawDataset(valid_data)
        CandDataset = R.RawDataset(unk_set)
    if(type_sample!='None'):
        weight=torch.tensor([1])
    else:

        weight =  torch.tensor(
            (len(known_set['Target']) - known_set['Target'].sum())
            /len(known_set['Target'])
        )
    if(Combo):
        return train_data ,valid_data ,unk_set ,weight
    else:
        return TrainDataset ,ValidDataset ,CandDataset ,weight

def train_one_epoch(epoch_index, training_loader,optimizer,model,criterion,mission):
    running_loss = 0.
    last_loss = 0.
    batch_size = training_loader.batch_size

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    mission_flag = 'Raw' in mission  # if true then running new model
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels,_ = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch

        if(mission_flag):
            p = np.random.permutation(len(_))
            inputs = [inputs[0][p], inputs[1][p], inputs[2][p], inputs[3][p], inputs[4][p], inputs[5][p],
                            inputs[6][p],inputs[7][p]]
            labels = labels[p]
            outputs = model(
                flux     = ReshapeEnsureNotNan(inputs[0], shape=(batch_size, 1, 6000)),              #flux
                centroid = ReshapeEnsureNotNan2D([inputs[1],inputs[2]], shape=(batch_size, 2, 6001)) , #centroid data
                f2_2 = ReshapeEnsureNotNan(inputs[3], shape=(batch_size, 1)),                        #transit depth
                f4_2 = ReshapeEnsureNotNan(inputs[4], shape=(batch_size, 5)),                        #centroid scalars
                f6_2 = ReshapeEnsureNotNan(inputs[5], shape=(batch_size, 4)),                        #secondary Scalars
                f7   = ReshapeEnsureNotNan(inputs[6], shape=(batch_size, 6)),                        #Stellar parameters
                f8   = ReshapeEnsureNotNan(inputs[7], shape=(batch_size, 6))                        #DV Diagnosti                                                            #model Vector
            )
        else:
            p = np.random.permutation(len(_))
            inputs = [inputs[0][p], inputs[1][p], inputs[2][p], inputs[3][p], inputs[4][p], inputs[5][p],
                      inputs[6][p], inputs[7][p],inputs[8][p],inputs[9][p],inputs[10][p],inputs[11][p]]
            labels = labels[p]
            #run ExoMiner
            outputs=model(f1=ReshapeEnsureNotNan(inputs[0], shape=(batch_size, 1, 301)),
                  f2=ReshapeEnsureNotNan(inputs[1], shape=(batch_size, 1, 31)),
                  f2_2=ReshapeEnsureNotNan(inputs[2], shape=(batch_size, 1)),
                  f3=ReshapeEnsureNotNan(inputs[3], shape=(batch_size, 1, 301)),
                  f4=ReshapeEnsureNotNan(inputs[4], shape=(batch_size, 1, 31)),
                  f4_2=ReshapeEnsureNotNan(inputs[5], shape=(batch_size, 5)),
                  f5_odd=ReshapeEnsureNotNan(inputs[6], shape=(batch_size, 1, 31)),
                  f5_even=ReshapeEnsureNotNan(inputs[7], shape=(batch_size, 1, 31)),
                  f6=ReshapeEnsureNotNan(inputs[8], shape=(batch_size, 1, 31)),
                  f6_2=ReshapeEnsureNotNan(inputs[9], shape=(batch_size, 4)),
                  f7=ReshapeEnsureNotNan(inputs[10], shape=(batch_size, 6)),
                  f8=ReshapeEnsureNotNan(inputs[11], shape=(batch_size, 6)))

        # Compute the loss and its gradients
        labels = [torch.tensor([l]) for l in labels]
        loss = criterion(outputs, torch.tensor([labels]).reshape(outputs.shape).float())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss/len(training_loader.sampler)

def TrainAndValidate(NumEpochs,training_loader,validation_loader,optimizer,model,criterion,mission):
    epoch_number = 0
    batch_size = training_loader.batch_size
    TrainingLoss, ValidationLoss = [],[]
    writer = SummaryWriter()
    mission_flag = 'Raw' in mission # if true then running new model
    if(mission_flag):
        best_vloss =.61
    else:
        best_vloss =.47
    for epoch in range(NumEpochs):
        #print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        training_loss = train_one_epoch(epoch_number,training_loader,optimizer,model,criterion,mission)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        vdata_arr = []
        vlabels_arr = []
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels,_ = vdata
                if(mission_flag):
                    #run our model

                    voutputs = model(
                        flux=ReshapeEnsureNotNan(vinputs[0], shape=(batch_size, 1, 6000)),  # flux
                        centroid=ReshapeEnsureNotNan2D([vinputs[1], vinputs[2]], shape=(batch_size, 2, 6001)),
                        # centroid data
                        f2_2=ReshapeEnsureNotNan(vinputs[3], shape=(batch_size, 1)),  # transit depth
                        f4_2=ReshapeEnsureNotNan(vinputs[4], shape=(batch_size, 5)),  # centroid scalars
                        f6_2=ReshapeEnsureNotNan(vinputs[5], shape=(batch_size, 4)),  # secondary Scalars
                        f7=ReshapeEnsureNotNan(vinputs[6], shape=(batch_size, 6)),  # Stellar parameters
                        f8=ReshapeEnsureNotNan(vinputs[7], shape=(batch_size, 6))  # DV Diagnostics
                          # model Vector
                    )
                else:
                    #run exominer
                    voutputs = model(f1=ReshapeEnsureNotNan(vinputs[0], shape=(batch_size, 1, 301)),
                          f2=ReshapeEnsureNotNan(vinputs[1], shape=(batch_size, 1, 31)),
                          f2_2=ReshapeEnsureNotNan(vinputs[2], shape=(batch_size, 1)),
                          f3=ReshapeEnsureNotNan(vinputs[3], shape=(batch_size, 1, 301)),
                          f4=ReshapeEnsureNotNan(vinputs[4], shape=(batch_size, 1, 31)),
                          f4_2=ReshapeEnsureNotNan(vinputs[5], shape=(batch_size, 5)),
                          f5_odd=ReshapeEnsureNotNan(vinputs[6], shape=(batch_size, 1, 31)),
                          f5_even=ReshapeEnsureNotNan(vinputs[7], shape=(batch_size, 1, 31)),
                          f6=ReshapeEnsureNotNan(vinputs[8], shape=(batch_size, 1, 31)),
                          f6_2=ReshapeEnsureNotNan(vinputs[9], shape=(batch_size, 4)),
                          f7=ReshapeEnsureNotNan(vinputs[10], shape=(batch_size, 6)),
                          f8=ReshapeEnsureNotNan(vinputs[11], shape=(batch_size, 6)))

                vlabels = [torch.tensor([l]) for l in vlabels]
                vloss = criterion(voutputs, torch.tensor([vlabels]).reshape(voutputs.shape).float())
                running_vloss += vloss
                vdata_arr.append([voutputs.detach().numpy()])
                vlabels_arr.append([vlabels])
            vdata_arr= torch.as_tensor(vdata_arr).reshape(torch.as_tensor(vdata_arr).shape[0:3])
            vlabels_arr = torch.as_tensor(vlabels_arr).reshape(vdata_arr.shape)
            # plot epochs used for testing
            plot_epochs = [10,20,50,75,100]
            validation_loss = running_vloss / len(validation_loader.sampler)
            print('Epoch {}: LOSS train {} valid {}'.format(epoch_number + 1, training_loss, validation_loss))
            TrainingLoss.append(training_loss)
            ValidationLoss.append(validation_loss.detach().item())
            writer.add_scalars(f'Loss/LossPlots_{batch_size}_{optimizer.param_groups[0]["lr"]}_{NumEpochs}', {
                'train': training_loss,
                'validation': validation_loss }
                               ,epoch_number)


                # p_val_dict = {}
                # a_val_dict = {}
                # r_val_dict = {}
                # fpr_val_dict ={}
                # for t in np.arange(0,1,.01):
                #     percision = BinaryPrecision(threshold=t)
                #     recall =BinaryRecall(threshold=t)
                #     accuracy =BinaryAccuracy(threshold=t)
                #     stats=  BinaryStatScores(threshold=t)
                #     TP, FP, TN, FN, _ = stats(vdata_arr, vlabels_arr)
                #     fpr_val_dict[t] = FP / (FP + TN)
                #     p_val_dict[t]  = percision(vdata_arr, vlabels_arr)
                #     r_val_dict[t]  = recall(vdata_arr, vlabels_arr)
                #     a_val_dict[t]  = accuracy(vdata_arr, vlabels_arr)

                # PlotAccuracyThreshold(mission, a_val_dict, epoch+1, optimizer.param_groups[0]['lr'], batch_size)
                # PlotFprRecal(mission, fpr_val_dict, r_val_dict,epoch+1,optimizer.param_groups[0]['lr'],batch_size)
                # PlotPercisionRecal(mission,p_val_dict,r_val_dict,epoch+1,optimizer.param_groups[0]['lr'],batch_size)
                # SaveStats(mission, a_val_dict, p_val_dict, r_val_dict, fpr_val_dict,epoch+1,optimizer.param_groups[0]['lr'],batch_size)
                # LossCurve(mission, TrainingLoss, ValidationLoss, epoch+1, optimizer.param_groups[0]['lr'], batch_size)


            # Track best performance, and save the model's state
            if validation_loss < best_vloss:
                best_vloss = validation_loss
                model_path = 'ModelCheckpoints/'+mission+'/_bf_model_{}_batch_size_{}_lr_{}'.format(epoch_number,batch_size,optimizer.param_groups[0]['lr'])
                torch.save(model.state_dict(), model_path)
                plot_data_set = validation_loader.dataset
                Plot_loader = DataLoader(plot_data_set, batch_size=1)
                GenPercisionPlots(model, Plot_loader, mission, optimizer, epoch_number + 1, fp='Predictions/')
                LossCurve(mission, TrainingLoss, ValidationLoss, epoch_number + 1, optimizer.param_groups[0]['lr'], batch_size)
            epoch_number += 1
    LossCurve(mission , TrainingLoss, ValidationLoss,NumEpochs,optimizer.param_groups[0]['lr'],batch_size)
    writer.flush()


    return model

def PlotPercisionRecal(mission,Percision,Recall,EpochNum,lr,batch_size,fp = 'Figures/' ):
    p_temp = []
    r_temp = []
    for ci in Percision.keys():
        p_temp.append(Percision[ci].item())
        r_temp.append(Recall[ci].item())
    plt.plot(r_temp,p_temp)
    v = Recall[.5]
    v2 = Percision[.5]
    plt.scatter(Recall[.5],Percision[.5],label=f'R(t=.5)={v:.2f},P(t=.5)={v2:.2f}',c='r')
    plt.xlabel('Recall')
    plt.ylabel('Percision')
    plt.title('Precision vs Recall Plot')
    plt.legend()
    plt.savefig(fp +mission + f'/Epoch_{EpochNum}_lr_{lr}_batchsize_{batch_size}_RecallVsPercision.png')
    plt.close()
    return None

def PlotFprRecal(mission,Fpr,Recall,EpochNum,lr,batch_size,fp = 'Figures/'):
    fpr_temp = []
    r_temp = []
    for ci in Fpr.keys():
        fpr_temp.append(Fpr[ci].item() )
        r_temp.append(Recall[ci].item())
    plt.plot(fpr_temp,r_temp,label='Recall')

    x = [0,1]
    y= [0,1]
    plt.plot(x,y, c='g', ls='--', label='No Skill')
    v = Fpr[.5]
    v2 = Recall[.5]
    plt.scatter( Fpr[.5],Recall[.5], label=f'FPR(t=.5)={v:.2f},R(t=.5)={v2:.2f}', c='r')
    plt.ylabel('Recall')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.savefig(fp +mission + f'/Epoch_{EpochNum}_lr_{lr}_batchsize_{batch_size}_RecallVsFPR.png')
    plt.close()
    return None

def PlotAccuracyThreshold(mission,Acc,EpochNum,lr,batch_size,fp = 'Figures/'):
    acc_temp = []
    for ci in Acc.keys():
        acc_temp.append(Acc[ci].item() )
    plt.plot(Acc.keys(),acc_temp,label = 'Accuracy')
    v = Acc[.5]
    plt.scatter(.5,Acc[.5],label=f't=.5, Accuracy={v:.2f}',c='r')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title(mission + f' Accuracy vs Threshold lr={lr},batch_size={batch_size}')
    plt.legend()
    plt.savefig(fp +mission + f'/Epoch_{EpochNum}_lr_{lr}_batchsize_{batch_size}_AccVsThreshold.png')
    plt.close()
    return None

def PlotAccuracyEpoch(mission,Acc,Epochs,t,lr,batch_size,fp = 'Figures/'):
    plt.plot(Epochs,Acc,label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy at {}'.format(t))
    plt.legend()
    plt.savefig(fp +mission + f'/All_Epochs_Accuracy_Threshold_{t}_lr_{lr}_batchsize_{batch_size}.png')
    plt.close()
    return None

def LossCurve(mission,TrainingLoss,ValidationLoss,EpochNum,lr,batch_size,fp = 'Figures/'):
    plt.plot(TrainingLoss, label='Training Loss')
    plt.plot(ValidationLoss, label='Validation Loss')
    plt.legend()
    plt.savefig(fp +mission+ f'/Epoch_{EpochNum}_lr_{lr}_batchsize_{batch_size}_ExoMinerTrainAndValidLossPlot.png')
    plt.close()
    return None

def SaveStats(mission ,Accuracy,Percision,Recall,FPR,F1,AUROC, EpochNum,lr,batch_size,fp='Stats/'):
    file_str = fp + mission + f'/Epoch_{EpochNum}_lr_{lr}_batchsize_{batch_size}_ExoMinerResults.txt'
    File = open(file_str, 'w')
    for t in Accuracy.keys():
        ExoMinerdata = {'Threshold': t,'Accuracy': Accuracy[t].item(), 'Percision': Percision[t].item(), 'Recall': Recall[t].item(),
                        'False Positive Rate':FPR[t].item() , 'F1 Score': F1[t].item()}

        File.write(str(ExoMinerdata))
        File.write('\n')
    File.write(f'AUROC Val:{AUROC}')
    File.close()
    return None

def SavePatK(mission,PatK ,EpochNum,lr,batch_size,fp='Stats/PatK/'):
    file_str = fp + mission + f'/Epoch_{EpochNum}_lr_{lr}_batchsize_{batch_size}_PatKResults.txt'
    File = open(file_str, 'w')
    for key in PatK .keys():
        PatKdata = {'k= '+str(key) :  'val='+str(PatK[key])}
        File.write(str(PatKdata))
        File.write('\n')
    File.close()
    return None



def GenPercisionPlots(model,Plot_loader,mission,optimizer, EpochNum ,fp='Predictions/'):
    batch_size = Plot_loader.batch_size
    outputs_arr = []
    targets = []
    mission_flag = 'Raw' in mission
    with torch.no_grad():
        for i, cdata in enumerate(Plot_loader):
            vinputs, clabels,ids = cdata
            if (mission_flag):
                # run our model

                coutputs = model(
                    flux=ReshapeEnsureNotNan(vinputs[0], shape=(batch_size, 1, 6000)),  # flux
                    centroid=ReshapeEnsureNotNan2D([vinputs[1], vinputs[2]], shape=(batch_size, 2, 6001)),
                    # centroid data
                    f2_2=ReshapeEnsureNotNan(vinputs[3], shape=(batch_size, 1)),  # transit depth
                    f4_2=ReshapeEnsureNotNan(vinputs[4], shape=(batch_size, 5)),  # centroid scalars
                    f6_2=ReshapeEnsureNotNan(vinputs[5], shape=(batch_size, 4)),  # secondary Scalars
                    f7=ReshapeEnsureNotNan(vinputs[6], shape=(batch_size, 6)),  # Stellar parameters
                    f8=ReshapeEnsureNotNan(vinputs[7], shape=(batch_size, 6))  # DV Diagnostics
                    # model Vector
                )
            else:
                # run exominer
                coutputs = model(f1=ReshapeEnsureNotNan(vinputs[0], shape=(batch_size, 1, 301)),
                                 f2=ReshapeEnsureNotNan(vinputs[1], shape=(batch_size, 1, 31)),
                                 f2_2=ReshapeEnsureNotNan(vinputs[2], shape=(batch_size, 1)),
                                 f3=ReshapeEnsureNotNan(vinputs[3], shape=(batch_size, 1, 301)),
                                 f4=ReshapeEnsureNotNan(vinputs[4], shape=(batch_size, 1, 31)),
                                 f4_2=ReshapeEnsureNotNan(vinputs[5], shape=(batch_size, 5)),
                                 f5_odd=ReshapeEnsureNotNan(vinputs[6], shape=(batch_size, 1, 31)),
                                 f5_even=ReshapeEnsureNotNan(vinputs[7], shape=(batch_size, 1, 31)),
                                 f6=ReshapeEnsureNotNan(vinputs[8], shape=(batch_size, 1, 31)),
                                 f6_2=ReshapeEnsureNotNan(vinputs[9], shape=(batch_size, 4)),
                                 f7=ReshapeEnsureNotNan(vinputs[10], shape=(batch_size, 6)),
                                 f8=ReshapeEnsureNotNan(vinputs[11], shape=(batch_size, 6)))
            out = coutputs.detach().numpy()
            target = clabels.detach().numpy()
            out = np.reshape(out, batch_size)
            target = np.reshape(target , batch_size)
            outputs_arr.append(out[0])
            targets.append(target[0])
    p_val_dict = {}
    a_val_dict = {}
    r_val_dict = {}
    fpr_val_dict = {}
    f1_score_dict ={}
    Patk = {50:0,100:0,200:0,1000:0,2200:0}

    for t in np.arange(0, 1, .01):
        percision = BinaryPrecision(threshold=t)
        recall = BinaryRecall(threshold=t)
        accuracy = BinaryAccuracy(threshold=t)
        stats = BinaryStatScores(threshold=t)
        f1 = BinaryF1Score(threshold=t)
        TP, FP, TN, FN, _ = stats(torch.tensor(outputs_arr), torch.tensor(targets))




        fpr_val_dict[t] = FP / (FP + TN)
        f1_score_dict[t] = f1(torch.tensor(outputs_arr), torch.tensor(targets))
        p_val_dict[t] = percision(torch.tensor(outputs_arr), torch.tensor(targets))
        r_val_dict[t] = recall(torch.tensor(outputs_arr), torch.tensor(targets))
        a_val_dict[t] = accuracy(torch.tensor(outputs_arr), torch.tensor(targets))

        if (t == .5):
            for k in Patk.keys():
                PatK = Precision(task ='binary' , threshold=.5, top_k=k)
                Patk[k] = PatK(torch.tensor(outputs_arr), torch.tensor(targets))
            SavePatK(mission, Patk, EpochNum, optimizer.param_groups[0]['lr'], batch_size)
    auroc = BinaryAUROC(thresholds=None)
    auroc_val = auroc(torch.tensor(outputs_arr), torch.tensor(targets))

    SaveStats(mission, a_val_dict, p_val_dict, r_val_dict, fpr_val_dict,f1_score_dict,auroc_val, EpochNum, optimizer.param_groups[0]['lr'], batch_size, fp='Stats/')
    PlotPrecisionThreshold(mission, p_val_dict, EpochNum , optimizer.param_groups[0]['lr'], batch_size)
    PlotF1Score(mission, f1_score_dict, EpochNum, optimizer.param_groups[0]['lr'], batch_size)
    PlotRecallThreshold(mission, r_val_dict, EpochNum, optimizer.param_groups[0]['lr'], batch_size)
    PlotAccuracyThreshold(mission, a_val_dict, EpochNum , optimizer.param_groups[0]['lr'], batch_size)
    PlotFprRecal(mission, fpr_val_dict, r_val_dict, EpochNum, optimizer.param_groups[0]['lr'], batch_size)
    PlotPercisionRecal(mission=mission, Percision=p_val_dict, Recall=r_val_dict, EpochNum=EpochNum, lr=optimizer.param_groups[0]['lr'], batch_size=batch_size)
    return None

def PlotF1Score(mission,f1_dict,EpochNum,lr,batch_size,fp = 'Figures/'):
    f1_temp = []
    for ci in f1_dict.keys():
        f1_temp.append(f1_dict[ci].item())
    plt.plot(f1_dict.keys(), f1_temp, label='F1 Score')
    v = f1_dict[.5]
    plt.scatter(.5,f1_dict[.5],label=f't=.5, F1={v:.2f}',c='r')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title(mission + f' Precision vs Threshold lr={lr},batch_size={batch_size}')
    plt.legend()
    plt.savefig(fp + mission + f'/Epoch_{EpochNum}_lr_{lr}_batchsize_{batch_size}_F1ScoreVsThreshold.png')
    plt.close()

def PlotPrecisionThreshold(mission,Precision,EpochNum,lr,batch_size,fp = 'Figures/'):
    Precision_temp = []
    for ci in Precision.keys():
        Precision_temp.append(Precision[ci].item() )

    plt.plot(Precision.keys(),Precision_temp,label = 'Precision')
    v =Precision[.5]
    plt.scatter(.5,Precision[.5],label=f't=.5, Precision={v:.2f}',c='r')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title(mission + f' Precision vs Threshold lr={lr},batch_size={batch_size}')
    plt.legend()
    plt.savefig(fp +mission + f'/Epoch_{EpochNum}_lr_{lr}_batchsize_{batch_size}_PrecisionVsThreshold.png')
    plt.close()
    return None

def PlotRecallThreshold(mission,Recall,EpochNum,lr,batch_size,fp = 'Figures/'):
    recall_temp = []
    for ci in Recall.keys():
        recall_temp.append(Recall[ci].item() )
    plt.plot(Recall.keys(),recall_temp,label = 'Recall')
    v = Recall[.5]
    plt.scatter(.5,Recall[.5],label=f't=.5, Recall={v:.2f}',c='r')
    plt.xlabel('Threshold')
    plt.ylabel('Recall')
    plt.title(mission + f' Recall vs Threshold lr={lr},batch_size={batch_size}')
    plt.legend()
    plt.savefig(fp +mission + f'/Epoch_{EpochNum}_lr_{lr}_batchsize_{batch_size}_RecallVsThreshold.png')
    plt.close()
    return None



def UnpackRawData(mission, fp='RawData/'):
    files = glob.glob('./' + fp + mission + r'/dataset_*.npy')
    idxs = ['load' not in file for file in files]
    f = []
    for i in range(0, len(idxs)):
        if (idxs[i]):
            f.append(files[i])
        else:
            pass
    files = f
    files.sort()
    data_return = {}
    for file in files:
        shutil.copyfile(file, file[:-4] + '_load.npy')
        shutil.copyfile('./RawData/'+mission+'/'+mission+'Scalars.npy', './RawData/'+mission+'/'+mission+'Scalars' + '_load.npy')
        data = np.load(file[:-4]+'_load.npy', allow_pickle=True).item()
        data2= np.load('./RawData/'+mission+'/'+mission+'Scalars' + '_load.npy',allow_pickle=True).item()
        for id in data.keys():
            data_return[id] = data[id]
            temp = data2[id]
            d = {}
            for key in list(temp.keys()):
                d[key] = temp[key]
            m = [1 if mission=='Kepler'  else 0] # mission feature to determine Kepler or TESS
            centroid_col = data_return[id]['Centroid Col']
            centroid_row = data_return[id]['Centroid Row']
            centroid_col = np.append(centroid_col,m)
            centroid_row = np.append(centroid_row,m)
            data_return[id]['Centroid Col'] =centroid_col
            data_return[id]['Centroid Row'] =centroid_row
            data_return[id].update(d)
    return data_return




def RunModelOnCandidates(mission,CandLoader,model,threshold =.5,fp='Predictions/'):
    batch_size = CandLoader.batch_size
    df = pd.DataFrame()
    mission_flag = 'Raw' in mission
    with torch.no_grad():
        for i, cdata in enumerate(CandLoader):
            vinputs, clabels,ids = cdata
            if (mission_flag):
                # run our model

                coutputs = model(
                    flux=ReshapeEnsureNotNan(vinputs[0], shape=(batch_size, 1, 6000)),  # flux
                    centroid=ReshapeEnsureNotNan2D([vinputs[1], vinputs[2]], shape=(batch_size, 2, 6001)),
                    # centroid data
                    f2_2=ReshapeEnsureNotNan(vinputs[3], shape=(batch_size, 1)),  # transit depth
                    f4_2=ReshapeEnsureNotNan(vinputs[4], shape=(batch_size, 5)),  # centroid scalars
                    f6_2=ReshapeEnsureNotNan(vinputs[5], shape=(batch_size, 4)),  # secondary Scalars
                    f7=ReshapeEnsureNotNan(vinputs[6], shape=(batch_size, 6)),  # Stellar parameters
                    f8=ReshapeEnsureNotNan(vinputs[7], shape=(batch_size, 6))  # DV Diagnostics
                    # model Vector
                )
            else:
                # run exominer
                coutputs = model(f1=ReshapeEnsureNotNan(vinputs[0], shape=(batch_size, 1, 301)),
                                 f2=ReshapeEnsureNotNan(vinputs[1], shape=(batch_size, 1, 31)),
                                 f2_2=ReshapeEnsureNotNan(vinputs[2], shape=(batch_size, 1)),
                                 f3=ReshapeEnsureNotNan(vinputs[3], shape=(batch_size, 1, 301)),
                                 f4=ReshapeEnsureNotNan(vinputs[4], shape=(batch_size, 1, 31)),
                                 f4_2=ReshapeEnsureNotNan(vinputs[5], shape=(batch_size, 5)),
                                 f5_odd=ReshapeEnsureNotNan(vinputs[6], shape=(batch_size, 1, 31)),
                                 f5_even=ReshapeEnsureNotNan(vinputs[7], shape=(batch_size, 1, 31)),
                                 f6=ReshapeEnsureNotNan(vinputs[8], shape=(batch_size, 1, 31)),
                                 f6_2=ReshapeEnsureNotNan(vinputs[9], shape=(batch_size, 4)),
                                 f7=ReshapeEnsureNotNan(vinputs[10], shape=(batch_size, 6)),
                                 f8=ReshapeEnsureNotNan(vinputs[11], shape=(batch_size, 6)))
            out = coutputs.detach().numpy()
            out = np.reshape(out, batch_size)
            ids = np.array(ids)
            dispo = ['Conf' if o>=threshold else 'unk' for o in out]
            mission_arr=[mission]*len(ids)
            df_temp = pd.DataFrame([mission_arr,out,dispo],index=['Mission','Score','Disposition'],columns=ids)
            df = pd.concat([df,df_temp.T])
    df.to_csv(fp+mission + '/'+'Predictions_threshold_{}_.csv'.format(threshold))
    return None

def LoadBestModel(batch_size,lr,mission,fp = 'ModelCheckpoints/'):
    files = glob.glob('./' + fp  + mission + r'/model_*_batch_size_{}_lr_{}*'.format(batch_size,lr))
    files.sort()
    checkpoint = torch.load(files[-1])
    return checkpoint


def LoadBestFitExoMiner(mission='Combo',fp = 'ModelCheckpoints/'):
    files = glob.glob('./' + fp  + mission + r'/*bf*')
    files.sort()
    checkpoint = torch.load(files[-1])
    return checkpoint





def UseExoMinerWeights(ExoMinerRawModel,ExoMiner):
    ExoMinerRawModel.f1_conv1.weight.data = ExoMiner['f1_conv1.weight']
    ExoMinerRawModel.f1_conv1.weight.requires_grad_(True)
    ExoMinerRawModel.f1_conv1.bias.data =ExoMiner['f1_conv1.bias']

    ###########
    ExoMinerRawModel.f1_conv2.weight.data =ExoMiner['f1_conv2.weight']
    ExoMinerRawModel.f1_conv2.weight.requires_grad_(True)
    ExoMinerRawModel.f1_conv2.bias.data =ExoMiner['f1_conv2.bias']

    ###########
    ExoMinerRawModel.f1_conv3.weight.data =ExoMiner['f1_conv3.weight']
    ExoMinerRawModel.f1_conv3.weight.requires_grad_(True)
    ExoMinerRawModel.f1_conv3.bias.data =ExoMiner['f1_conv3.bias']

    ###########
    ExoMinerRawModel.f1_lin1.weight.data =ExoMiner['f1_lin1.weight']
    ExoMinerRawModel.f1_lin1.weight.requires_grad_(True)
    ExoMinerRawModel.f1_lin1.bias.data =ExoMiner['f1_lin1.bias']






    #####
    ExoMinerRawModel.f2_conv1.weight.data = ExoMiner['f2_conv1.weight']
    ExoMinerRawModel.f2_conv1.weight.requires_grad_(True)
    ExoMinerRawModel.f2_conv1.bias.data =ExoMiner['f2_conv1.bias']

    ###########
    ExoMinerRawModel.f2_conv2.weight.data =ExoMiner['f2_conv2.weight']
    ExoMinerRawModel.f2_conv2.weight.requires_grad_(True)
    ExoMinerRawModel.f2_conv2.bias.data =ExoMiner['f2_conv2.bias']

    ###########


    ###########
    ExoMinerRawModel.f2_lin1.weight.data =ExoMiner['f2_lin1.weight']
    ExoMinerRawModel.f2_lin1.weight.requires_grad_(True)
    ExoMinerRawModel.f2_lin1.bias.data =ExoMiner['f2_lin1.bias']


    #####
    ExoMinerRawModel.f3_conv1.weight.data = ExoMiner['f3_conv1.weight']
    ExoMinerRawModel.f3_conv1.weight.requires_grad_(True)
    ExoMinerRawModel.f3_conv1.bias.data = ExoMiner['f3_conv1.bias']

    ###########
    ExoMinerRawModel.f3_conv2.weight.data = ExoMiner['f3_conv2.weight']
    ExoMinerRawModel.f3_conv2.weight.requires_grad_(True)
    ExoMinerRawModel.f3_conv2.bias.data = ExoMiner['f3_conv2.bias']

    ###########
    ExoMinerRawModel.f3_conv3.weight.data = ExoMiner['f3_conv3.weight']
    ExoMinerRawModel.f3_conv3.weight.requires_grad_(True)
    ExoMinerRawModel.f3_conv3.bias.data = ExoMiner['f3_conv3.bias']

    ###########
    ExoMinerRawModel.f3_lin1.weight.data = ExoMiner['f3_lin1.weight']
    ExoMinerRawModel.f3_lin1.weight.requires_grad_(True)
    ExoMinerRawModel.f3_lin1.bias.data = ExoMiner['f3_lin1.bias']


    #####
    ExoMinerRawModel.f4_conv1.weight.data = ExoMiner['f4_conv1.weight']
    ExoMinerRawModel.f4_conv1.weight.requires_grad_(True)
    ExoMinerRawModel.f4_conv1.bias.data = ExoMiner['f4_conv1.bias']

    ExoMinerRawModel.f4_conv2.weight.data = ExoMiner['f4_conv2.weight']
    ExoMinerRawModel.f4_conv2.weight.requires_grad_(True)
    ExoMinerRawModel.f4_conv2.bias.data = ExoMiner['f4_conv2.bias']





    ###########
    ExoMinerRawModel.f4_lin1.weight.data = ExoMiner['f4_lin1.weight']
    ExoMinerRawModel.f4_lin1.weight.requires_grad_(True)
    ExoMinerRawModel.f4_lin1.bias.data = ExoMiner['f4_lin1.bias']


    #####
    ExoMinerRawModel.f5_conv1.weight.data = ExoMiner['f5_conv1.weight']
    ExoMinerRawModel.f5_conv1.weight.requires_grad_(True)
    ExoMinerRawModel.f5_conv1.bias.data = ExoMiner['f5_conv1.bias']

    ###########
    ExoMinerRawModel.f5_conv2.weight.data = ExoMiner['f5_conv2.weight']
    ExoMinerRawModel.f5_conv2.weight.requires_grad_(True)
    ExoMinerRawModel.f5_conv2.bias.data = ExoMiner['f5_conv2.bias']



    ###########
    ExoMinerRawModel.f5_lin1.weight.data = ExoMiner['f5_lin1.weight']
    ExoMinerRawModel.f5_lin1.weight.requires_grad_(True)
    ExoMinerRawModel.f5_lin1.bias.data = ExoMiner['f5_lin1.bias']

    #####
    ExoMinerRawModel.f6_conv1.weight.data = ExoMiner['f6_conv1.weight']
    ExoMinerRawModel.f6_conv1.weight.requires_grad_(True)
    ExoMinerRawModel.f6_conv1.bias.data = ExoMiner['f6_conv1.bias']

    ###########
    ExoMinerRawModel.f6_conv2.weight.data = ExoMiner['f6_conv2.weight']
    ExoMinerRawModel.f6_conv2.weight.requires_grad_(True)
    ExoMinerRawModel.f6_conv2.bias.data = ExoMiner['f6_conv2.bias']


    ###########
    ExoMinerRawModel.f6_lin1.weight.data = ExoMiner['f6_lin1.weight']
    ExoMinerRawModel.f6_lin1.weight.requires_grad_(True)
    ExoMinerRawModel.f6_lin1.bias.data = ExoMiner['f6_lin1.bias']



    ####
    ExoMinerRawModel.FinalLin1.weight.data = ExoMiner['FinalLin1.weight']
    ExoMinerRawModel.FinalLin2.weight.data = ExoMiner['FinalLin2.weight']
    ExoMinerRawModel.FinalLin3.weight.data = ExoMiner['FinalLin3.weight']
    ExoMinerRawModel.FinalLin4.weight.data = ExoMiner['FinalLin4.weight']
    ExoMinerRawModel.FinalLin1.weight.requires_grad_(True)
    ExoMinerRawModel.FinalLin2.weight.requires_grad_(True)
    ExoMinerRawModel.FinalLin3.weight.requires_grad_(True)
    ExoMinerRawModel.FinalLin4.weight.requires_grad_(True)
    ExoMinerRawModel.FinalLin1.bias.data = ExoMiner['FinalLin1.bias']
    ExoMinerRawModel.FinalLin2.bias.data = ExoMiner['FinalLin2.bias']
    ExoMinerRawModel.FinalLin3.bias.data = ExoMiner['FinalLin3.bias']
    ExoMinerRawModel.FinalLin4.bias.data = ExoMiner['FinalLin4.bias']


    return ExoMinerRawModel







def RunExoMiner(mission,lr,batch_size,training_size,epochs,type_sample='Oversample',load=True):  # mission
    torch.manual_seed(1)
    np.random.seed(1)
    print('Running ' + mission)
    if(mission=='TESS' or mission=='Kepler'):
        DataObject = gd.GatherData(mission=mission)
        data = UnPackData(mission)
        DataObject.IdData = DataObject.IdData[DataObject.IdData.WrapperId.isin(data.keys())]
        TrainDataset, ValidDataset, CandDataset, weight = GrabDatasets(data, DataObject, training_size, batch_size,Combo = False,type_sample=type_sample)
        model = ExoMiner.ExoMiner()
    elif(mission=='Combo'):
        DataObjectTESS = gd.GatherData(mission='TESS')
        dataTESS = UnPackData('TESS')
        DataObjectKepler = gd.GatherData(mission='Kepler')
        dataKepler = UnPackData('Kepler')
        DataObjectTESS.IdData = DataObjectTESS.IdData[DataObjectTESS.IdData.WrapperId.isin(dataTESS.keys())]
        DataObjectKepler.IdData = DataObjectKepler.IdData[DataObjectKepler.IdData.WrapperId.isin(dataKepler.keys())]
        TrainDatasetTESS, ValidDatasetTESS, CandDatasetTESS, weight = GrabDatasets(dataTESS, DataObjectTESS, training_size, batch_size,type_sample=type_sample)
        TrainDatasetKepler, ValidDatasetKepler, CandDatasetKepler, weight = GrabDatasets(dataKepler, DataObjectKepler, training_size, batch_size,type_sample=type_sample)
        train_data =pd.concat([TrainDatasetTESS,TrainDatasetKepler])
        valid_data =pd.concat([ValidDatasetTESS,ValidDatasetKepler])
        unk_data  =pd.concat([CandDatasetTESS,CandDatasetKepler])
        TrainDataset = D.MyDataset(train_data)
        ValidDataset = D.MyDataset(valid_data)
        CandDataset = D.MyDataset(unk_data)
        model = ExoMiner.ExoMiner()
    elif(mission=='ComboRaw'):
        DataObjectTESS = gd.GatherData(mission='TESS')
        dataTESS = UnpackRawData('TESS')
        DataObjectKepler = gd.GatherData(mission='Kepler')
        dataKepler = UnpackRawData('Kepler')
        DataObjectTESS.IdData = DataObjectTESS.IdData[DataObjectTESS.IdData.WrapperId.isin(dataTESS.keys())]
        DataObjectKepler.IdData = DataObjectKepler.IdData[DataObjectKepler.IdData.WrapperId.isin(dataKepler.keys())]
        TraindfTESS, ValiddfTESS, CanddfTESS, weight = GrabDatasets(dataTESS, DataObjectTESS,
                                                                                   training_size, batch_size,
                                                                                   type_sample=type_sample,Raw =True)
        TraindfKepler, ValiddfKepler, CanddfKepler, weight = GrabDatasets(dataKepler, DataObjectKepler,
                                                                                         training_size, batch_size,
                                                                                         type_sample=type_sample,Raw =True)
        train_data = pd.concat([TraindfTESS, TraindfKepler])
        valid_data = pd.concat([ValiddfTESS, ValiddfKepler])
        unk_data = pd.concat([CanddfTESS, CanddfKepler])
        TrainDataset = R.RawDataset(train_data)
        ValidDataset = R.RawDataset(valid_data)
        CandDataset = R.RawDataset(unk_data)
        model = ExoMinerRaw.ExoMinerRaw()

    elif (mission == 'TESSRaw'):
        DataObject = gd.GatherData(mission='TESS')
        data = UnpackRawData('TESS')
        DataObject.IdData = DataObject.IdData[DataObject.IdData.WrapperId.isin(data.keys())]
        TrainDataset, ValidDataset, CandDataset, weight = GrabDatasets(data, DataObject, training_size, batch_size,
                                                                       Combo=False, type_sample=type_sample,Raw =True)
        model = ExoMinerRaw.ExoMinerRaw()
    elif (mission == 'KeplerRaw'):
        print('Running '+mission)
        DataObject = gd.GatherData(mission='Kepler')
        data = UnpackRawData('Kepler')
        DataObject.IdData = DataObject.IdData[DataObject.IdData.WrapperId.isin(data.keys())]
        TrainDataset, ValidDataset, CandDataset, weight = GrabDatasets(data, DataObject, training_size, batch_size,
                                                                       Combo=False, type_sample=type_sample,Raw =True)
        model = ExoMinerRaw.ExoMinerRaw()
    if('Raw' in mission and load==True):
        checkpoint =LoadBestFitExoMiner()
        model = UseExoMinerWeights(ExoMinerRawModel=model, ExoMiner=checkpoint)

    TrainLoader = DataLoader(TrainDataset,batch_size=batch_size,drop_last=True)
    ValidLoader = DataLoader(ValidDataset,batch_size=batch_size,drop_last=True)
    #CandLoader = DataLoader(CandDataset,batch_size=batch_size,drop_last=True)
    #PlotLoader = DataLoader(ValidDataset, batch_size=1)

    criterion = nn.BCELoss(weight = weight, size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = TrainAndValidate(NumEpochs=epochs,
                     training_loader=TrainLoader,
                     validation_loader=ValidLoader,
                     optimizer=optimizer,
                     model=model,
                     criterion=criterion,
                     mission=mission)
    return None

def ModelChoice(model,mission,lr,training_size,batch_size,epochs,type_sample):
    if (model == "ExoMiner"):

        RunExoMiner(mission=mission,
                    lr=lr,
                    training_size=training_size,
                    batch_size=batch_size,
                    epochs=epochs,
                    type_sample=type_sample)

    elif(model == 'ExoMinerRaw'):
        RunExoMiner(mission=mission,
                    lr=lr,
                    training_size=training_size,
                    batch_size=batch_size,
                    epochs=epochs,
                    type_sample=type_sample)
    else:
        print('Choose either "ExoMiner" or "ExoMinerRaw" for model argument')
    return None



if __name__=='__main__':
    ModelChoice(model='ExoMinerRaw',
                  mission='ComboRaw',
                  lr=6.37e-5,
                  training_size=.8,
                  batch_size=10,
                  epochs=500,
                  type_sample='Oversample')
