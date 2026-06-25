import os
import numpy as np
import pandas as pd
from itertools import product
import ForecastLib  #importing forecastlib.py file

from sklearn import preprocessing
from hyperopt import fmin, tpe, hp
from DeepRVFL_.DeepRVFL import DeepRVFL  #import DeepRVFL class instance from DeepRVFL_ folder and DeepRVFL.py file name

import warnings
warnings.filterwarnings("ignore")


###############################################################################################################################

def format_data(dat,order,target):

    n_sample=dat.shape[0]-order-step+1
    x=np.zeros((n_sample,dat.shape[1]*order))
    y=np.zeros((n_sample,1))
    for i in range(n_sample):
        x[i,:]=dat[i:i+order,:].ravel()
        y[i]  =target[i+order+step-1]

    return x.T,y.T
###############################################################################################################################

def select_indexes(data, indexes):
    return data[:,indexes]
    
###############################################################################################################################

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

###############################################################################################################################

def get_data(name):
    file_name = name+'.csv'
    dat = pd.read_csv(file_name)
    dat = dat.fillna(method='ffill')
    return dat,dat.columns

###############################################################################################################################

class Struct(object): pass

###############################################################################################################################

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
    Nu=data.inputs.shape[0] # number of input features

    Nh = hyper[0][0] # number of hidden neurons
    Nl = layer # Layer
    
    #defining the list for regularization and input scale parameters
    reg, iss = [], []
    
    for h in hyper:
        reg.append(h[1])        
        iss.append(h[2])
        
    #loading the configurations    
    configs=config_load(iss,train_idx)
    #calling the deep RVFL instance
    deepRVFL = DeepRVFL(Nu, Nh, Nl, configs)
    #train targets
    train_targets = select_indexes(data.targets, train_idx)

    
    # hidden states calculation adaptive for first layer and more than one
    if Nl==1:
        states = deepRVFL.computeLayerState(0,data.inputs)
    else:
        states = deepRVFL.computeLayerState(Nl-1,data.inputs,last_states[:,:])
        

    # Concatenating states and inputs for training and test data
    train_states = select_indexes(np.concatenate([states,data.inputs],axis=0), train_idx)
    test_states = select_indexes(np.concatenate([states,data.inputs],axis=0), test_idx)

    # weight(beta) calculation
    deepRVFL.trainReadout(train_states[:,:], train_targets, reg[-1])
    
    #test output calculation
    test_outputs_norm = deepRVFL.computeOutput(test_states[:,:]).T

    return test_outputs_norm,states[:,:]

###############################################################################################################################

def edRVFL_predict(hyper,data,train_idx,test_idx,s):
    
    np.random.seed(s) # random seed
    Nu=data.inputs.shape[0] # number of input features

    Nr = hyper[0][0] # number of hidden neurons
    Nl = len(hyper) # number of recurrent layers
    
    #defining the list for regularization and input scale parameters
    reg, iss = [], []
    
    for h in hyper:
        reg.append(h[1])        
        iss.append(h[2])
        
    #loading the configurations    
    configs=config_load(iss,train_idx)
    #calling the deep RVFL instance
    deepRVFL = DeepRVFL(Nu, Nr, Nl, configs)
    
    #Intialising the last_states as None
    last_states=None
    #Defining an empty numpy array of shape test_idx and number of layers
    outputs=np.zeros((len(test_idx),Nl))
    #train targets
    train_targets = select_indexes(data.targets, train_idx)
    
    
    for l in range(Nl):
        # hidden states calculation adaptive for zero^th layer and more than one
        if l==0:
            states = deepRVFL.computeLayerState(l,data.inputs,inistate=None)
        else:
            states=deepRVFL.computeLayerState(l,data.inputs,last_states)
            
        #Updating the last_states to be newly evaluated/calculated states
        last_states=states
        
        # Concatenating states and inputs for training and test data
        train_states = select_indexes(np.concatenate([states,data.inputs],axis=0), train_idx)
        test_states = select_indexes(np.concatenate([states,data.inputs],axis=0), test_idx)
        
        # weight(beta) calculation
        deepRVFL.trainReadout(train_states, train_targets, reg[l])
        #test output calculation
        test_outputs_norm = deepRVFL.computeOutput(test_states).T
        #Saving the test outputs corresponding to each layer
        outputs[:,l:l+1]=test_outputs_norm

    return np.median(outputs,axis=1).reshape(-1,1), outputs#outputs.mean(axis=1).reshape(-1,1)

###############################################################################################################################

def cross_validation(hypers,data,raw_data,train_idx,val_idx,Nl,regs,input_scale,scaler=None,s=0,boat=50):
    
    #defining an empty list to store the best hyperparameters corresponding to each layer
    best_hypers = [] 
    np.random.seed(s)  # random seed
    layer_s = None # intialising the state layer to None
    for i in range(Nl):

        layer = i+1
        layer_h,layer_s = layer_cross_validation(hypers,data,raw_data,train_idx,val_idx,layer,
                           scaler=scaler,s=s,last_states=layer_s,best_hypers=best_hypers.copy(),boat=boat)

        Nhs=[layer_h[0]]  #number of hidden units corresponding to each optimized layer
        if layer==1:
            hypers=list(product(Nhs,regs,input_scale))   
            
        best_hypers.append(layer_h)

    return best_hypers

###############################################################################################################################

def layer_cross_validation(hypers,data,raw_data,train_idx,val_idx,layer,
                           scaler=None,s=0,last_states=None,best_hypers=None,boat=50):

    np.random.seed(s)   #random seed

    space={'layer' : hp.choice('layer', [layer]),
           'data' : hp.choice('data', [data]),
           'raw_data' : hp.choice('raw_data', [raw_data]),
           'last_states' : hp.choice('last_states', [last_states]),
           'scaler' : hp.choice('scaler', [scaler]),
           's' : hp.choice('s', [s]),
           'val_idx' : hp.choice('val_idx', [val_idx]),
           'train_idx' : hp.choice('train_idx', [train_idx]),
           'best_hypers' : hp.choice('best_hypers', [best_hypers]),
            'input_scale' : hp.uniform('input_scale', 0,1),
            'regs' : hp.uniform('regs', 0, 1)}
    
    
    if layer==1:
        space['Nhs']=hp.randint('Nhs', 10, 200)
    else:
        best_hidden=[best_hypers[0][0]]
        space['Nhs']=hp.choice('Nhs', [best_hypers[0][0]])
        
    # defining the hyperopt optimization function
    args=fmin(fn=layer_obj,
                space=space,
                max_evals=boat,
                rstate=np.random.default_rng(seed=0),
                algo=tpe.suggest)
    
    if layer==1:
        best_hyper=[args['Nhs'],args['regs'],args['input_scale']]
    else:
        best_hyper=[best_hidden[0],args['regs'],args['input_scale']]
        
    if layer>1:
            hyper_=best_hypers.copy()
            hyper_.append(best_hyper)
    else:
        hyper_=[best_hyper]
        
    _,best_state=dRVFL_predict(hyper_,data,train_idx,val_idx,layer,
                                         s,last_states=last_states)
   
    return best_hyper,best_state

###############################################################################################################################

def layer_obj(args):
    layer=args['layer']
    best_hypers=args['best_hypers']

    hyper=[args['Nhs'],args['regs'],args['input_scale']]
    data=args['data']
    train_idx,val_idx=args['train_idx'],args['val_idx']
    scaler=args['scaler']
    s=args['s']
    raw_data,last_states=args['raw_data'],args['last_states']
    
    if layer>1:
            hyper_=[i for i in best_hypers]
            hyper_.append(hyper)
    else:
        hyper_=[hyper]


    test_outputs_norm,_=dRVFL_predict(hyper_,data,train_idx,val_idx,layer,
                                     s,last_states=last_states)
    test_outputs_norm = (test_outputs_norm+1)/2
    test_outputs=scaler.inverse_transform(test_outputs_norm)
    actuals=raw_data[-len(val_idx):]
    test_err=compute_error(actuals,test_outputs,None)
    
    return test_err['RMSE']

###############################################################################################################################



Nhs=np.arange(50,300,50)

Nls=[1]#np.arange(2,12,4)
regs=[0]
input_scale=[0.1]#,0.1,0.001]#[0.1,0.01,0.001]
ratios=np.arange(0.05,1,0.05)
deepRVFL_hypers=list(product(Nhs,regs,input_scale,ratios))
order=24
step = 4
seeds = 1
boat=100
    

for year in ['2017','2018','2019']:
    for station in ['46083h','46080h','46076h','46001h']:
        features=[ 'WDIR', 'WSPD', 'GST','APD','WVHT']
        
        data=pd.read_csv('wave/'+station+year+'.txt.gz',delim_whitespace=True)
        data=data[features]

        var_name=data.columns
        data=data.where(data!='99.0',np.nan)
        data=data.where(data!='99.00',np.nan)
        data=data.fillna(method='ffill')
        while data.isnull().values.any():
            data=data.fillna(method='ffill')
        print(data.isnull().values.any())
        print(year+station)

        data_=data['WVHT'].values[1:].astype(float).reshape(-1,1)


        val_l,test_l=int(0.2*data_.shape[0]),int(0.2*data_.shape[0])
        ml_data=pd.DataFrame(data[features].values[1:,:].astype(float),columns=features)
        ml_data['WVHT']=data_

        scaler=preprocessing.MinMaxScaler() 
        target_scaler=preprocessing.MinMaxScaler() 
        
        #cross validation 
        scale_errs=[]  
        test_pres_ea=[]
        allpres = []
        results = []
        
        for s in np.arange(seeds-1,seeds):
            target_scaler.fit(data_[:-test_l-val_l].reshape(-1,1))
            target_norm=2*(target_scaler.transform(data_.reshape(-1,1)))-1
            scaler.fit(ml_data.values[:-test_l-val_l])
            norm_data=2*(scaler.transform(ml_data.values))-1

            
            data=Struct()
            data.inputs,data.targets=format_data(norm_data,order,target_norm)
            train_l=data.inputs.shape[1]-val_l-test_l
            train_idx=range(train_l)
            val_idx=range(train_l,train_l+val_l)
            test_idx=range(train_l+val_l,data.inputs.shape[1])

            
            best_hypers=cross_validation(deepRVFL_hypers[:],data,data_[:-test_l],
                                                  train_idx,val_idx,Nls[0],regs,
                                                  input_scale,scaler=target_scaler,s=s,boat=boat)
            

            test_outputs_norm_mea,alllayers=edRVFL_predict(best_hypers,data,train_idx,test_idx,s)
            test_outputs_norm_mea = (test_outputs_norm_mea + 1)/2
            
            alllayers=target_scaler.inverse_transform((alllayers + 1)/2)
            allpres.append(alllayers)
            test_outputs_ea=target_scaler.inverse_transform(test_outputs_norm_mea)
            test_pres_ea.append(test_outputs_ea)
            
            #Appending the test outputs in the list corresponding to each seed
            test_pres_ea.append(test_outputs_ea)
            
            
            actuals=data_[-test_l:]
            history=data_[:-test_l]
    
            test_err=compute_error(actuals,test_outputs_ea,history)
            print(test_err)
            print(len(best_hypers))
            
            
            results.append([year, station, test_err['RMSE'], test_err['MAPE'], test_err['MASE']])

        
        all_p=np.concatenate(allpres,axis=1)
        dfall=pd.DataFrame(all_p)
        dfall.to_csv('wave/ABA/allp_rvfl_four_BOA'+str(Nls[0])+str(100)+year+station+'.csv')

        output_loc = f"rvfl_wave_four_results//{year}//{station}//"

        if not os.path.exists(output_loc):
            os.makedirs(output_loc)
            
        results_df = pd.DataFrame(results, columns=['year', 'station', 'test_rmse', 'test_mape', 'test_mase'])
        results_df.to_csv(f'rvfl_wave_four_results//{year}//{station}//results.csv')

        test_p=np.concatenate(test_pres_ea,axis=1)
        dfea=pd.DataFrame(test_p)
        dfea.to_csv(f'rvfl_wave_four_results//{year}//{station}//rvflBOA{boat}{year}{station}')