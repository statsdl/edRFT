import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis





def calculate_statistics(data):
    max_value = np.max(data)
    min_value = np.min(data)
    median_value = np.median(data)
    mean_value = np.mean(data)
    std_dev = np.std(data)
    skewness = skew(data)
    kurt = kurtosis(data)

    return max_value, min_value, median_value, mean_value, std_dev, skewness, kurt




countrys=['QLD']  # countrys listed in the AEMO
year='2020'  # year of the AEMO dataset
boat=50  #number of epochs/trials in hyperopt
for year in ['2019', '2018', '2017'][::-1]:
    for station in ['46001h','46076h','46077h','46080h', '46083h']:
        # print(year+station)
        results = []

        features=[ 'WDIR', 'WSPD', 'GST','APD','WVHT']
        df_data=pd.read_csv('wave/'+station+year+'.txt.gz',delim_whitespace=True)

        data=df_data['WVHT'].values[1:].astype(float).reshape(-1,1)


        max_val, min_val, median_val, mean_val, std_dev_val, skewness_val, kurtosis_val = calculate_statistics(data)

        # print(f"Max: {max_val}")
        # print(f"Min: {min_val}")
        # print(f"Median: {median_val}")
        # print(f"Mean: {mean_val}")
        # print(f"Standard Deviation: {std_dev_val}")
        # print(f"Skewness: {skewness_val}")
        print(f"{np.round(kurtosis_val,2).item()}")


