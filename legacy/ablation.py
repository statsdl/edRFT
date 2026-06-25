from sklearn import preprocessing
from ForecastLib import TsMetric
import numpy as np
import pandas as pd
import numpy.ma as ma
import collections
import matplotlib.pyplot as plt
from scipy.stats import rankdata,wilcoxon,friedmanchisquare
import scikit_posthocs as sp 


def get_data(name):

    file_name = name+'.csv'
    print(file_name)
    dat = pd.read_csv(file_name)
    # dat = dat.fillna(method='ffill')
    return dat


def compute_pre(x):
    nsample=x.shape[0]
    seed=10
    each_seed=np.zeros((nsample,10))
    for i in range(seed):
        cp=x[:,i*10:i*10+nl]
        each_seed[:,i]=np.mean(cp,axis=1)
    return each_seed


rmse_all=[]
mape_all=[]
mase_all=[]


# mss=['Jan','Apr','Jul','Oct']
year='2020'
locs=['SA','NSW','VIC', 'TAS']
months=['01_','04_','07_','10_']
NLS=[2,4,6,8,10]

BOAPAedRVFL_means=[]
BOAedRVFL_means=[]
BOAAedRVFL_means=[]

BOAPAedRVFL_rmsemeans=[]
BOAedRVFL_rmsemeans=[]
BOAAedRVFL_rmsemeans=[]


for nl in NLS:
    for loc in locs:
        axi=0

        plt.rcParams["font.family"] = "Times New Roman"
    
        for month_ in months:

            month='PRICE_AND_DEMAND_2020'+month_+loc+str(1)
            dataset='D:\\AEMO\\'+loc+'\\'+ year +'\\'+month
            pddata=get_data(dataset)
            print(pddata)
            data_=pddata['TOTALDEMAND'].values.reshape(-1,1)

            np_data=data_

            validation_l,test_l=int(0.1*np_data.shape[0]),int(0.2*np_data.shape[0])
            train_l=len(np_data)-test_l-validation_l
            target=np_data[-test_l:].ravel()
            history=np_data[:-test_l].ravel()
            prediction={}
            rmse={}
            mape={}
            mase={}
            error={}
            
            metric=TsMetric()
            import os
            print(os.getcwd())
            
            boapaedrvfl_loc=r'AEMO\ABA\allp_edRVFL_attentive_patching_BOA1050'+month + '.csv'
            print(boapaedrvfl_loc)
            boapaedrvflpres=get_data(boapaedrvfl_loc).values[:,1:]
            boapaedrvflpres=compute_pre(boapaedrvflpres)
            prediction['PAedRVFLBOA']=boapaedrvflpres
            boapaedrvfl_rmse=np.zeros(boapaedrvflpres.shape[1])
            boapaedrvfl_mape=np.zeros(boapaedrvflpres.shape[1])
            boapaedrvfl_mase=np.zeros(boapaedrvflpres.shape[1])
            for k in range(boapaedrvflpres.shape[1]):
                boapaedrvfl_rmse[k]=metric.RMSE(target, boapaedrvflpres[:,k])
                boapaedrvfl_mape[k]=metric.MAPE(target, boapaedrvflpres[:,k])
                boapaedrvfl_mase[k]=metric.MASE(target, boapaedrvflpres[:,k],history)
            rmse['PAedRVFLBOA']=boapaedrvfl_rmse.mean()
            mase['PAedRVFLBOA']=boapaedrvfl_mase.mean()
            mape['PAedRVFLBOA']=boapaedrvfl_mape.mean()
            print("grid DONE")

            boaaedrvfl_loc='AEMO/ABA/allp_edRVFL_attentive_BOA1050'+month+'.csv'
            boaaedrvflpres=get_data(boaaedrvfl_loc).values[:,1:]#297*100
            boaaedrvflpres=compute_pre(boaaedrvflpres)
            prediction['AedRVFLBOA']=boaaedrvflpres#.mean(axis=1)
            boaaedrvfl_rmse=np.zeros(boaaedrvflpres.shape[1])
            boaaedrvfl_mape=np.zeros(boaaedrvflpres.shape[1])
            boaaedrvfl_mase=np.zeros(boaaedrvflpres.shape[1])
            for k in range(boaaedrvflpres.shape[1]):
                boaaedrvfl_rmse[k]=metric.RMSE(target, boaaedrvflpres[:,k])
                boaaedrvfl_mape[k]=metric.MAPE(target, boaaedrvflpres[:,k])
                boaaedrvfl_mase[k]=metric.MASE(target, boaaedrvflpres[:,k],history)
            rmse['AedRVFLBOA']=boaaedrvfl_rmse.mean()
            mase['AedRVFLBOA']=boaaedrvfl_mase.mean()
            mape['AedRVFLBOA']=boaaedrvfl_mape.mean()
            
            
            boaedrvfl_loc='AEMO/ABA/allp_edRVFLBOA1050'+month+'.csv'
            boaedrvflpres=get_data(boaedrvfl_loc).values[:,1:]#297*100
            boaedrvflpres=compute_pre(boaedrvflpres)
            prediction['edRVFLBOA']=boaedrvflpres#.mean(axis=1)
            boaedrvfl_rmse=np.zeros(boaedrvflpres.shape[1])
            boaedrvfl_mape=np.zeros(boaedrvflpres.shape[1])
            boaedrvfl_mase=np.zeros(boaedrvflpres.shape[1])
            for k in range(boaedrvflpres.shape[1]):
                boaedrvfl_rmse[k]=metric.RMSE(target, boaedrvflpres[:,k])
                boaedrvfl_mape[k]=metric.MAPE(target, boaedrvflpres[:,k])
                boaedrvfl_mase[k]=metric.MASE(target, boaedrvflpres[:,k],history)
            rmse['edRVFLBOA']=boaedrvfl_rmse.mean()
            mase['edRVFLBOA']=boaedrvfl_mase.mean()
            mape['edRVFLBOA']=boaedrvfl_mape.mean()


            error['RMSE']=rmse
            error['MAPE']=mape
            error['MASE']=mase
            error_df=pd.DataFrame.from_dict(error,orient='index')

            rmse_all.append(error_df.loc['RMSE'].values.reshape(1,-1))
            mape_all.append(error_df.loc['MAPE'].values.reshape(1,-1))
            mase_all.append(error_df.loc['MASE'].values.reshape(1,-1))


    rmse_all_np=np.concatenate(rmse_all,axis=0)
    mase_all_np=np.concatenate(mase_all,axis=0)
    mape_all_np=np.concatenate(mape_all,axis=0)
    scaler=preprocessing.MinMaxScaler()
    rmse_all_np=scaler.fit_transform(rmse_all_np.T).T
    mase_all_np=scaler.fit_transform(mase_all_np.T).T
    rmsealldf=pd.DataFrame(data=rmse_all_np,columns=error_df.columns)
    masealldf=pd.DataFrame(data=mase_all_np,columns=error_df.columns)
    mapealldf=pd.DataFrame(data=mape_all_np,columns=error_df.columns)
    
    ranks=np.zeros(rmse_all_np.shape)
    av=[]
    for err in [rmse_all_np,mase_all_np,mape_all_np]:
        for i in range(err.shape[0]):

            ranks[i,:]=rankdata(err[i,:])

        af2=friedmanchisquare(*err)
        print(af2)
        avranks=np.mean(ranks,axis=0).reshape(1,-1)
        av.append(avranks)


    avrank=np.concatenate(av,axis=0)
    avrank_df=pd.DataFrame(data=avrank,columns=error_df.columns)
    print(avrank_df)

    
    BOAPAedRVFL_means.append(masealldf['PAedRVFLBOA'].values.mean())
    BOAAedRVFL_means.append(masealldf['AedRVFLBOA'].values.mean())
    BOAedRVFL_means.append(masealldf['edRVFLBOA'].values.mean())           
    # BOAEWTedRVFL_means.append(masealldf['EWTedRVFLBOA'].values.mean())
    
    BOAPAedRVFL_rmsemeans.append(rmsealldf['PAedRVFLBOA'].values.mean())
    BOAAedRVFL_rmsemeans.append(rmsealldf['AedRVFLBOA'].values.mean())
    BOAedRVFL_rmsemeans.append(rmsealldf['edRVFLBOA'].values.mean())
    # BOAEWTedRVFL_rmsemeans.append(rmsealldf['EWTedRVFLBOA'].values.mean())
    
x = np.arange(len(NLS))  # the label locations
width = 0.2  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, BOAPAedRVFL_means, width, label='PAedRVFL')
rects2 = ax.bar(x ,  BOAAedRVFL_means, width, label='AedRVFL')
rects3 = ax.bar(x + width/2, BOAedRVFL_means, width, label='edRVFL')

# Add some text for labels, title and custom x-axis tick labels, etc.
labels=NLS
ax.set_ylabel('Errors')
# ax.set_title('Scores by group and gender')
ax.set_xticks(x, labels)
ax.legend()
fig.tight_layout()
plt.show()

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, BOAPAedRVFL_rmsemeans, width, label='PAedRVFL')
rects2 = ax.bar(x ,  BOAAedRVFL_rmsemeans, width, label='AedRVFL')
rects3 = ax.bar(x + width, BOAedRVFL_rmsemeans, width, label='edRVFL')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Normalized RMSE')
ax.set_xlabel('Layers')
# ax.set_title('Scores by group and gender')
ax.set_xticks( x)
ax.set_xticklabels(labels)
# ax.legend()
ax.legend(loc='center left', framealpha=0,fontsize=12,bbox_to_anchor=(1, 0.5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# ax.legend(framealpha=0,fontsize=7)
ax.set_facecolor('white')
fig.tight_layout()
plt.savefig('ablationstudyRMSE.png',dpi=1000,format='png', bbox_inches = 'tight' ,  pad_inches = 0)
plt.savefig('ablationstudyRMSE.eps',dpi=1000,format='eps', bbox_inches = 'tight' ,  pad_inches = 0)
plt.show()

