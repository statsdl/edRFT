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

def format_data(dat,order,idx=0):
    n_sample=dat.shape[0]-order
    x=np.zeros((n_sample,dat.shape[1]*order))
    y=np.zeros((n_sample,1))
    for i in range(n_sample):
        x[i,:]=dat[i:i+order,:].ravel()
        y[i]  =dat[i+order,idx]
    return x.T,y.T

###############################################################################################################################

decomposer=ForecastLib.TsPrep()

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

def dRVFL_predict(hyper,norm_data,train_idx,test_idx,layer,s,last_states=None,dec_=None,alldecs=None):
   
    np.random.seed(s)
    w, k=hyper[0][3], hyper[0][4]
    
    data = Struct()
    if dec_ is None:
        dec_=alldecs[str(w)+str(k)]
    
   

    data.inputs=np.zeros((k*1+1,dec_.shape[0]))
    data.targets=np.zeros((1,dec_.shape[0]))
    for i in range(dec_.shape[0]):
        data.inputs[:,i]=dec_[i,-1:,:].ravel()
        data.targets[0, i] = norm_data[w+i][0]  

    Nu = data.inputs.shape[0]
    Nh = hyper[0][0] 
    Nl = layer 
    
    reg, iss = [], []
    for h in hyper:
        reg.append(h[1])        
        iss.append(h[2])
        
    configs=config_load(iss,train_idx)
    deepRVFL = DeepRVFL(Nu, Nh, Nl, configs)
    train_targets = select_indexes(data.targets, train_idx)

    if Nl==1:
        states = deepRVFL.computeLayerState(0,data.inputs)
    else:
        states=deepRVFL.computeLayerState(Nl-1,data.inputs,last_states[:,:])
        
    
    train_states = select_indexes(np.concatenate([states,data.inputs],axis=0), train_idx)#(Nh,n_sample)
    test_states = select_indexes(np.concatenate([states,data.inputs],axis=0), test_idx)

    deepRVFL.trainReadout(train_states[:,:], train_targets, reg[-1])
    test_outputs_norm = deepRVFL.computeOutput(test_states[:,:]).T

    return test_outputs_norm, states[:,:], dec_

###############################################################################################################################
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
    dec_s=None
    alldecs={}
    
    
    print('collect all dec series')
    for w in [2,3,4]:
        for k in [2,3,4]:
            dec_=decomposer.ts_walkforward(data.ravel(),w,k,'ewt',pad_l=0,pad_raw=True)
            alldecs[str(w)+str(k)]=dec_
            
    for i in range(Nl):

        layer = i+1
        layer_h,layer_s = layer_cross_validation(hypers,data,raw_data,train_idx,val_idx,layer,
                           scaler=scaler,s=s,last_states=layer_s,best_dec=dec_s,best_hypers=best_hypers.copy(),boat=boat,
                           alldecs=alldecs)

        Nhs=[layer_h[0]]  #number of hidden units corresponding to each optimized layer
        if layer==1:
            hypers=list(product(Nhs,regs,input_scale))   
            
        best_hypers.append(layer_h)

    return best_hypers


###############################################################################################################################

def layer_cross_validation(hypers,data,raw_data,train_idx,val_idx,layer,
                           scaler=None,s=0,last_states=None,best_dec=None,best_hypers=None,boat=50,alldecs=None):

    np.random.seed(s)   #random seed

    space={'layer' : hp.choice('layer', [layer]),
           'data' : hp.choice('data', [data]),
           'raw_data' : hp.choice('raw_data', [raw_data]),
           'last_states' : hp.choice('last_states', [last_states]),
           'best_dec':hp.choice('best_dec', [best_dec]),
           'scaler' : hp.choice('scaler', [scaler]),
           's' : hp.choice('s', [s]),
           'val_idx' : hp.choice('val_idx', [val_idx]),
           'train_idx' : hp.choice('train_idx', [train_idx]),
           'best_hypers' : hp.choice('best_hypers', [best_hypers]),
           'w':hp.choice('w', [2,3,4]),
           'k':hp.choice('k', [2,3,4]),
           'alldecs':hp.choice('alldecs', [alldecs]),
            'input_scale' : hp.uniform('input_scale', 0,1),
            'regs' : hp.uniform('regs', 0, 1)}
    
    
    if layer==1:
        space['Nhs']=hp.randint('Nhs', 10, 200)
        space['w']=hp.choice('w', [2,3,4])
        space['k']=hp.choice('k', [2,3,4])
    else:
        best_hidden=[best_hypers[0][0]]
        space['Nhs']=hp.choice('Nhs', [best_hypers[0][0]])
        
        best_w=[best_hypers[0][3]]
        space['w']=hp.choice('w', best_w)
        
        best_k=[best_hypers[0][4]]
        space['k']=hp.choice('k', best_k)
        
    # defining the hyperopt optimization function
    args=fmin(fn=layer_obj,
                space=space,
                max_evals=boat,
                rstate=np.random.default_rng(seed=0),
                algo=tpe.suggest)
    
    if layer==1:
        wi=args['w']
        ki=args['k']
        issi=args['input_scale']
        best_hyper=[args['Nhs'],args['regs'],issi,[2,3,4][wi],[2,3,4][ki]]
    else:
        issi=args['input_scale']
        best_hyper=[best_hidden[0],args['regs'],issi,best_w[0],best_k[0]]
        
    if layer>1:
            hyper_=best_hypers.copy()
            hyper_.append(best_hyper)
    else:
        hyper_=[best_hyper]
        
    _,best_state,best_dec=dRVFL_predict(hyper_,data,train_idx,val_idx,layer,
                                         s,last_states=last_states,dec_=best_dec,alldecs=alldecs)
   
    return best_hyper,best_state,best_dec

###############################################################################################################################

def layer_obj(args):
    layer=args['layer']
    best_hypers=args['best_hypers']
    best_dec=args['best_dec']
    alldecs=args['alldecs']
    hyper=[args['Nhs'],args['regs'],args['input_scale'],args['w'],args['k']]
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


    test_outputs_norm,_,dec_=dRVFL_predict(hyper_,data,train_idx,val_idx,layer,
                                     s,last_states=last_states,dec_=best_dec,alldecs=alldecs)
    test_outputs=scaler.inverse_transform(test_outputs_norm)
    actuals=raw_data[-len(val_idx):]
    test_err=compute_error(actuals,test_outputs,None)
    
    return test_err['RMSE']

###############################################################################################################################

import datetime
import warnings

steps = 1

Nls = [10]
Nhs = np.arange(50,300,50) #number of hidden neurons
regs = [0] #value of the regularization parameters
input_scale=[0.1] #value of the input scale parameters
deepRVFL_hypers=list(product(Nhs,regs,input_scale))
seeds = 1
boat = 100  #number of epochs/trials in hyperopt
for year in ['2019','2018','2017'][::-1]:
    for station in ['46083h','46080h','46076h','46001h','46077h']:
        results = []
        print(year+station)
        features=[ 'WDIR', 'WSPD', 'GST','APD','WVHT']
        data=pd.read_csv('wave/'+station+year+'.txt.gz',delim_whitespace=True)
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
        for s in np.arange(seeds):
            test_scale_pres=[]
            np.random.seed(s)
            
            
            starttime=datetime.datetime.now()
            subs=ml_data.values#np.concatenate((subs,fs),axis=1)

            data=Struct()
            scaler.fit(subs[:-test_l-val_l,:])
            tscaler.fit(subs[:-test_l-val_l,-1:])
            norm_data=np.transpose(scaler.transform(subs))


            data.inputs=norm_data[:,:-4]
            data.targets=norm_data[-1:,4:]
            
            
            train_l=data.inputs.shape[1]-val_l-test_l
            train_idx=range(train_l-1)
            val_idx=range(train_l-1,train_l+val_l-1)
            test_idx=range(train_l+val_l-1,data.inputs.shape[1]-1)
            

            
            #best hyperparameters
            ed_best_hypers=cross_validation(deepRVFL_hypers[:],subs,data_[:-test_l],
                                              train_idx,val_idx,Nls[0],regs,
                                              input_scale,scaler=tscaler,s=s,boat=boat)
            

            test_outputs_norm_mea,alllayers=edRVFL_predict(ed_best_hypers,data,train_idx,test_idx,s)
            alllayers=tscaler.inverse_transform(alllayers)
            allpres.append(alllayers)
            test_outputs_ea=tscaler.inverse_transform(test_outputs_norm_mea)
            
            #Appending the test outputs in the list corresponding to each seed
            test_pres_ea.append(test_outputs_ea)
            
            
            actuals=data_[-test_l:]
            history=data_[:-test_l]
    
            test_err=compute_error(actuals,test_outputs_ea,history)
            print(test_err)
            print(len(ed_best_hypers))
            
            
            results.append([year, station, test_err['RMSE'], test_err['MAPE'], test_err['MASE']])

        
        all_p=np.concatenate(allpres,axis=1)
        dfall=pd.DataFrame(all_p)
        dfall.to_csv('wave/ABA/allp_edrvfl_four_BOA'+str(Nls[0])+str(50)+year+station+'.csv')

        output_loc = f"edrvfl_wave_four_results//{year}//{station}//"

        if not os.path.exists(output_loc):
            os.makedirs(output_loc)
            
        results_df = pd.DataFrame(results, columns=['year', 'station', 'test_rmse', 'test_mape', 'test_mase'])
        results_df.to_csv(f'edrvfl_wave_four_results//{year}//{station}//results.csv')

        test_p=np.concatenate(test_pres_ea,axis=1)
        dfea=pd.DataFrame(test_p)
        dfea.to_csv(f'edrvfl_wave_four_results//{year}//{station}//edrvflBOA{boat}{year}{station}')