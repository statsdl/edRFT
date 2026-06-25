import os
import math
import torch
import pickle
import ForecastLib
import numpy as np
import pandas as pd
import padasip as pa
from itertools import product
from sklearn import preprocessing
from hyperopt import fmin, tpe, hp
from main import TRVFL


import warnings
warnings.filterwarnings("ignore")

###############################################################################

def format_data(data, order, idx=0):
    n_sample = data.shape[0]-order
    x = np.zeros((n_sample, data.shape[1]*order))
    y = np.zeros((n_sample, 1))
    for i in range(n_sample):
        x[i,:] = data[i:i+order,:].ravel()
        y[i] = data[i+order,idx]
    return x, y

###############################################################################

def select_indexes(data, indexes):
    return data[indexes,:]

###############################################################################
    
def compute_error(actuals,predictions,history=None):
    actuals=actuals.ravel()
    predictions=predictions.ravel()
    
    metric=ForecastLib.TsMetric()
    error={}
    error['RMSE']=metric.RMSE(actuals, predictions)
    error['MAPE']=metric.MAPE(actuals,predictions)
    if history is not None:
        history=history.ravel()
        error['MASE']=metric.MASE(actuals,predictions,history)
    
    return error

###############################################################################

def get_data(name):
    file_name = name+'.csv'
    dat = pd.read_csv(file_name)
    dat = dat.fillna(method='ffill')
    return dat,dat.columns

###############################################################################

class Struct(object): pass

###############################################################################
def config_load(iss,IP_indexes):

    configs = Struct()
    configs.iss = iss # set insput scale 0.1 for all recurrent layers
    configs.IPconf = Struct()
    configs.IPconf.DeepIP = 0 # activate pre-train
    configs.enhConf = Struct()
    configs.enhConf.connectivity = 1 # connectivity of recurrent matrix
    configs.readout = Struct()
    configs.readout.trainMethod = 'Ridge' # train with singular value decomposition (more accurate)
    
    return configs 

###############################################################################################################################

def dRVFL_predict(hyper,data,train_idx,test_idx,layer,s,last_states=None):

    np.random.seed(s) # random seed

    Nh = hyper[0][0] # number of hidden neurons
    Nl = layer # Layer
    
    reg, iss, nl, heads, dr = [], [], [], [], []
    for h in hyper:
        reg.append(h[1])        
        iss.append(h[2])
        nl.append(h[3])
        heads.append(h[4])
        dr.append(h[5])
        
  
    trainX = select_indexes(data.inputs, train_idx)
    train_targets = select_indexes(data.targets, train_idx)

    if Nl == 1:
        last_states = data.inputs

    edrft = TRVFL(Nl-1, last_states.shape[1], trainX.shape[1], int(Nh), lamb=reg[-1], input_scale=iss[-1],
                  device='cpu', nt_layers=nl[-1], num_heads=int(heads[-1]), dropout_rate=dr[-1])

    edrft.train()
    edrft.eval()

    if Nl == 1:
        with torch.no_grad():
            states = edrft.transform(X_raw=torch.Tensor(data.inputs).float()).detach().clone()
    else:
        with torch.no_grad():
            states = edrft.transform(X_raw=torch.Tensor(data.inputs).float(), X=torch.Tensor(last_states).float()).detach().clone()

    edrft.init_weight(torch.Tensor(select_indexes(states, train_idx)).float(),
                      torch.Tensor(train_targets).float(), torch.Tensor(trainX).float())

    if Nl == 1:
        with torch.no_grad():
            preds = edrft(X_raw=torch.Tensor(data.inputs).float(), X=torch.Tensor(states).float()).cpu().detach().numpy()
    else:
        with torch.no_grad():
            preds = edrft(X_raw=torch.Tensor(data.inputs).float(), X=torch.Tensor(states).float()).cpu().detach().numpy()

    test_outputs_norm = select_indexes(preds, test_idx)

    return test_outputs_norm, states[:, :]

###############################################################################################################################

def edrft_predict(hyper,data,train_idx,test_idx,s):
    
    np.random.seed(s)
    Nr = hyper[0][0]
    Nl = len(hyper)
    
    
    reg, iss, nl, heads, dr = [], [], [], [], []
    for h in hyper:
        reg.append(h[1])        
        iss.append(h[2])
        nl.append(h[3])
        heads.append(h[4])
        dr.append(h[5])

    last_states = None
    outputs = np.zeros((len(test_idx), Nl))

    trainX = select_indexes(data.inputs, train_idx)
    train_targets = select_indexes(data.targets, train_idx)

    for l in range(Nl):
        if l == 0:
            last_states = data.inputs

        edrft = TRVFL(l, last_states.shape[1], trainX.shape[1], int(Nr), lamb=reg[l], input_scale=iss[l],
                      device='cpu', nt_layers=nl[l], num_heads=int(heads[l]), dropout_rate=dr[l])

        edrft.train()
        edrft.eval()

        if l == 0:
            with torch.no_grad():
                states = edrft.transform(X_raw=torch.Tensor(data.inputs).float()).detach().clone()
                preds = edrft(X_raw=torch.Tensor(data.inputs).float(), X=torch.Tensor(states).float()).cpu().detach().numpy()
        else:
            with torch.no_grad():
                states = edrft.transform(X_raw=torch.Tensor(data.inputs).float(), X=torch.Tensor(last_states).float()).detach().clone()
                preds = edrft(X_raw=torch.Tensor(data.inputs).float(), X=torch.Tensor(states).float()).cpu().detach().numpy()

        edrft.init_weight(torch.Tensor(select_indexes(states, train_idx)).float(),
                          torch.Tensor(train_targets).float(), torch.Tensor(trainX).float())

        if l == 0:
            with torch.no_grad():
                preds = edrft(X_raw=torch.Tensor(data.inputs).float(), X=torch.Tensor(states).float()).cpu().detach().numpy()
        else:
            with torch.no_grad():
                preds = edrft(X_raw=torch.Tensor(data.inputs).float(), X=torch.Tensor(states).float()).cpu().detach().numpy()

        last_states = states

        test_outputs_norm = select_indexes(preds, test_idx)
        outputs[:, l:l+1] = test_outputs_norm

    return np.median(outputs, axis=1).reshape(-1, 1), outputs

###############################################################################################################################

def cross_validation(hypers, data, raw_data, train_idx, val_idx, Nl, regs, input_scale, scaler=None, s=0, boat=50):
    best_hypers = []
    np.random.seed(s)
    layer_s = None

    for i in range(Nl):
        layer = i + 1
        layer_h, layer_s = layer_cross_validation(hypers, data, raw_data, train_idx, val_idx, layer,
                                                  scaler=scaler, s=s, last_states=layer_s, best_hypers=best_hypers.copy(), boat=boat)

        Nhs = [layer_h[0]]
        if layer == 1:
            hypers = list(product(Nhs, regs, input_scale))

        best_hypers.append(layer_h)

    return best_hypers

###############################################################################################################################

def layer_cross_validation(hypers, data, raw_data, train_idx, val_idx, layer, scaler=None, s=0, last_states=None, best_hypers=None, boat=50):
    np.random.seed(s)   # Random seed

    space = {
        'layer': layer,
        'data': data,
        'raw_data': raw_data,
        'last_states': last_states,
        'scaler': scaler,
        's': s,
        'val_idx': val_idx,
        'train_idx': train_idx,
        'best_hypers': best_hypers,
        'input_scale': hp.uniform('input_scale', 0, 1),
        'regs': hp.uniform('regs', 0, 0.1),
        'dr': hp.uniform('dr', 0, 0.5),
        'nl': hp.randint('nl', 1, 5),
        'heads': hp.quniform('heads', 1, 8, 2),
    }

    if layer == 1:
        space['Nhs'] = hp.quniform('Nhs', 1, 1024, 32)
    else:
        best_hidden = best_hypers[0][0] if best_hypers else 32
        space['Nhs'] = best_hidden

    # Defining the hyperopt optimization function
    args = fmin(
        fn=layer_obj,
        space=space,
        max_evals=boat,
        rstate=np.random.default_rng(seed=0),
        algo=tpe.suggest
    )

    if layer == 1:
        best_hyper = [args['Nhs'], args['regs'], args['input_scale'], args['nl'], args['heads'], args['dr']]
    else:
        best_hyper = [best_hidden, args['regs'], args['input_scale'], args['nl'], args['heads'], args['dr']]

    if layer > 1:
        hyper_ = best_hypers.copy() if best_hypers else []
        hyper_.append(best_hyper)
    else:
        hyper_ = [best_hyper]

    _, best_state = dRVFL_predict(hyper_, data, train_idx, val_idx, layer, s, last_states=last_states)

    return best_hyper, best_state

###############################################################################################################################

def layer_obj(args):
    layer = args['layer']
    best_hypers = args['best_hypers']

    hyper = [args['Nhs'], args['regs'], args['input_scale'], args['nl'], args['heads'], args['dr']]
    data = args['data']
    train_idx, val_idx = args['train_idx'], args['val_idx']
    scaler = args['scaler']
    s = args['s']
    raw_data, last_states = args['raw_data'], args['last_states']

    if layer > 1:
        hyper_ = [i for i in best_hypers]
        hyper_.append(hyper)
    else:
        hyper_ = [hyper]

    test_outputs_norm, _ = dRVFL_predict(hyper_, data, train_idx, val_idx, layer, s, last_states=last_states)
    test_outputs_norm = (test_outputs_norm+1)/2
    test_outputs=scaler.inverse_transform(test_outputs_norm)
    actuals=raw_data[-len(val_idx):]
    test_err=compute_error(actuals,test_outputs,None)
    
    return test_err['RMSE']


###############################################################################################################################

import datetime
import warnings

steps = 1

Nls = [10]
Nhs=np.arange(50,300,50) #number of hidden neurons
regs=[0] #value of the regularization parameters
input_scale=[0.1] #value of the input scale parameters
deepRVFL_hypers=list(product(Nhs,regs,input_scale))
seeds = 1
boat = 100  #number of epochs/trials in hyperopt
for year in ['2022','2021','2020','2019','2018','2017'][::-1]:
    for station in ['46083h','46080h','46076h','46001h','46077h']:
        results = []
        print(year+station)
        features=[ 'WDIR', 'WSPD', 'GST','APD','WVHT']
        data=pd.read_csv('wave/'+station+year+'.txt.gz',delim_whitespace=True)
        # print(data)
        data=data[features]


        var_name=data.columns
        data=data.where(data!='99.0',np.nan)
        data=data.where(data!='99.00',np.nan)
        data=data.fillna(method='ffill')
        while data.isnull().values.any():
            data=data.fillna(method='ffill')

    
        data_=data['WVHT'].values[1:].astype(float).reshape(-1,1)
        ml_data=pd.DataFrame(data[features].values[1:,:].astype(float),columns=features)


        ml_data['WVHT']=data_
        scaler=preprocessing.MinMaxScaler()    
        tscaler=preprocessing.MinMaxScaler()           
        val_l,test_l=int(0.1*data_.shape[0]),int(0.2*data_.shape[0])
        train_l=data_.shape[0]-val_l-test_l
        
        #cross validation 
        scale_errs=[]  
        test_pres_ea=[]
        allpres = []
        for s in np.arange(seeds-1,seeds):
            test_scale_pres=[]
            np.random.seed(s)
            
            
            starttime=datetime.datetime.now()
            subs=ml_data.values#np.concatenate((subs,fs),axis=1)
            data=Struct()
            scaler.fit(subs[:-test_l-val_l,:])
            tscaler.fit(subs[:-test_l-val_l,-1:])
            norm_data=2*(scaler.transform(subs))-1
            

            # data inputs and targets
            data.inputs = norm_data[:-1, :]
            data.targets = norm_data[1:, -1:]

            
            # data lenght and index
            train_l = data.inputs.shape[0] - val_l - test_l
            train_idx = range(train_l)
            val_idx = range(train_l, train_l + val_l)
            test_idx = range(train_l+val_l, data.inputs.shape[0])
            
            # #best hyperparameters
            # ed_best_hypers=cross_validation(deepRVFL_hypers[:],data,data_[:-test_l],
            #                                   train_idx,val_idx,Nls[0],regs,
            #                                   input_scale,scaler=tscaler,s=s,boat=boat)
            
            
            ###############################################################
            
            #Testing begins
            print('Test')
            
            # test_outputs_norm_mea,alllayers=edrft_predict(ed_best_hypers,data,train_idx,test_idx,s)
            # test_outputs_norm_mea = (test_outputs_norm_mea + 1)/2
            
            # alllayers=tscaler.inverse_transform((alllayers + 1)/2)
            # allpres.append(alllayers)
            # test_outputs_ea=tscaler.inverse_transform(test_outputs_norm_mea)
            # test_pres_ea.append(test_outputs_ea)
            
            # Specify the file path
            file_path = f'edrft_wave_results/{year}/{station}/edrftBOA{boat}{year}{station}'
            
            # Load the data using Pandas
            try:
                dfea = pd.read_csv(file_path, index_col=0)  # Assuming the DataFrame has an index column
                test_outputs_ea = np.array(dfea)  # Convert DataFrame to NumPy array
            
            except Exception as e:
                print(f"An error occurred: {e}")

        
            actuals=data_[-test_l:]
            history=data_[:-test_l]

            test_err=compute_error(actuals,test_outputs_ea,history)
            print(test_err)
            # print(len(ed_best_hypers)) 
            
            
            results.append([year, station, test_err['RMSE'], test_err['MAPE'], test_err['MASE']])

        
        # all_p=np.concatenate(allpres,axis=1)
        # dfall=pd.DataFrame(all_p)
        # dfall.to_csv('wave/ABA/allp_edrft_BOA'+str(Nls[0])+str(50)+year+station+'.csv')

        output_loc = f"edrft_wave_results//{year}//{station}//"

        if not os.path.exists(output_loc):
            os.makedirs(output_loc)
            
        results_df = pd.DataFrame(results, columns=['year', 'station', 'test_rmse', 'test_mape', 'test_mase'])
        results_df.to_csv(f'edrft_wave_results//{year}//{station}//results.csv')

        # test_p=np.concatenate(test_pres_ea,axis=1)
        # dfea=pd.DataFrame(test_p)
        # dfea.to_csv(f'edrft_wave_results//{year}//{station}//edrftBOA{boat}{year}{station}')

