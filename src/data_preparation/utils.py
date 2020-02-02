from datetime import datetime
import glob
import pandas as pd
import os
import multiprocessing as mp
import numpy as np
import json
import os.path
import re
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def percentile(series, quantile = [i*.05 for i in range(20)], fig_size=(12,8), show_plot=True):
    print(series.describe(quantile))
    plt.figure(figsize=fig_size)
    if show_plot:
        sns.distplot(series)
    
    
# get notebook id
def get_notebook_kernel_id():
   
    """
    Return the full path of the jupyter notebook.
    """
    
    import ipykernel
    import requests
    from requests.compat import urljoin
    import warnings
    
        
    from notebook.notebookapp import list_running_servers
    from IPython.utils.shimmodule import ShimWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ShimWarning)
    from notebook.notebookapp import list_running_servers
    
        
        
    kernel_id = re.search('kernel-(.*).json', ipykernel.connect.get_connection_file()).group(1)
    return kernel_id


def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables

    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal

    Returns:
       psi_values: ndarray of psi values for each variable

    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable

        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into

        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input


        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])



        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)

def pretty_print(col_list_or_df, is_printing_comma = True):
    if type(col_list_or_df) == type(pd.DataFrame()):
        col_list = col_list_or_df.columns
    else:
        col_list = col_list_or_df
    if is_printing_comma:
        for col in col_list:
            print('"{}",'.format(col))
    else:
        for col in col_list:
            print('{}'.format(col))
    return None


#read parquet/csv folder recursively
def get_list_files_in_folder(path, file_type = "parquet", file_name_start_with = None):
    if file_name_start_with is not None:
        file_paths =  glob.glob(os.path.join(path, (file_name_start_with + "*." + file_type)))
    else:
        file_paths =  glob.glob(os.path.join(path, ("*." + file_type)))
    if len(file_paths) > 0: # one level - do nothing
        pass
    else:
        sub_folder_paths =  glob.glob(os.path.join(path, "*"))
        for p in sub_folder_paths:
            file_paths += get_list_files_in_folder(p)
    return file_paths
from pyspark.sql.functions import lit
    
def read_folder_into_pd(path, file_type = "parquet",  file_name_start_with = None,exclude_lugi_temp_folder=False):
    file_path_list = get_list_files_in_folder(path, file_type, file_name_start_with)
    if exclude_lugi_temp_folder:
        file_path_list = [path for path in file_path_list if "luigi-tmp" not in path]
        
    if file_type == "parquet":
        df = pd.concat((pd.read_parquet(f, 'fastparquet') for f in file_path_list))
    elif file_type == "csv":
        df = pd.concat((pd.read_csv(f) for f in file_path_list))
        
    df.reset_index(drop=True, inplace=True)
    return df


##########################################
### READ FILE
def union_all(dfs):
    if len(dfs) > 1:
        return dfs[0].unionAll(union_all(dfs[1:]))
    else:
        return dfs[0]

    
########################################
    
def gen_hourdate(data_path ,datehour):
    return (data_path \
            +'/{datehour:%Y/%m/%d/%H}'.format(datehour=datehour)) \
            + "/*.parquet" 


def read_path_and_gen_hourdate(spark, data_path, start_day, end_day, month, year = 2019):
    data = []
    for i in range(start_day,end_day+1,1):
        for j in range(24):
            datehour = datetime(year=2019,month=month,day=i,hour=j)
            datehour_string = str(year) + "-" + str(month).zfill(2) + "-" + str(i).zfill(2) + " " + str(j).zfill(2)
            hadoop_paths = gen_hourdate(data_path, datehour)
            df = spark.read.parquet(hadoop_paths)
            df = df.withColumn("dateTime", lit(datehour_string))
            data.append(df)
    return union_all(data)


#####################################

def gen_date(data_path, date):
    return (data_path \
            +'/{date:%Y/%m/%d}'.format(date=date)) \
            + "/*.parquet" 


def read_path_and_gen_date(spark, data_path, start_day, end_day, month, year=2019):
    data = []
    for i in range(start_day,end_day+1,1):
        date = datetime(year=year,month=month,day=i)
        date_string = str(year) + "-" + str(month).zfill(2) + "-" + str(i).zfill(2)
        hadoop_paths = gen_date(data_path, date)
        df = spark.read.parquet(hadoop_paths)
        df = df.withColumn("dateTime", lit(date_string))
        data.append(df)
    return union_all(data)



def parallelize_dataframe(df, func, n_cores = 10, n_partitions = None):
    if n_partitions is None:
        n_partitions = n_cores
        
    df_split = np.array_split(df, n_partitions)
    with mp.Pool(n_cores) as pool:
        res = pd.concat(pool.map(func, df_split))
    return res

def feature_selection(feature_matrix, missing_threshold=90, correlation_threshold=0.95, show_cols_to_remove = False):
    """Feature selection for a dataframe."""
    
#     feature_matrix = pd.get_dummies(feature_matrix)
#     n_features_start = feature_matrix.shape[1]
    print('Original shape: ', feature_matrix.shape)

#     _, idx = np.unique(feature_matrix, axis = 1, return_index = True)
#     feature_matrix = feature_matrix.iloc[:, idx]
#     n_non_unique_columns = n_features_start - feature_matrix.shape[1]
#     print('{}  non-unique valued columns.'.format(n_non_unique_columns))
    n_non_unique_columns = 0
    
    # Find missing and percentage
    missing = pd.DataFrame(feature_matrix.isnull().sum())
    missing['percent'] = 100 * (missing[0] / feature_matrix.shape[0])
    missing.sort_values('percent', ascending = False, inplace = True)

    # Missing above threshold
    missing_cols = list(missing[missing['percent'] > missing_threshold].index)
    n_missing_cols = len(missing_cols)

    # Remove missing columns
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in missing_cols]]
    print('{} missing columns with threshold: {}.'.format(n_missing_cols,
                                                                        missing_threshold))
    
    # Zero variance
    unique_counts = pd.DataFrame(feature_matrix.nunique()).sort_values(0, ascending = True)
    zero_variance_cols = list(unique_counts[unique_counts[0] == 1].index)
    n_zero_variance_cols = len(zero_variance_cols)

    # Remove zero variance columns
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in zero_variance_cols]]
    print('{} zero variance columns.'.format(n_zero_variance_cols))
    
    # Correlations
    corr_matrix = feature_matrix.corr()

    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))

    # Select the features with correlations above the threshold
    # Need to use the absolute value
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

    n_collinear = len(to_drop)
    
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in to_drop]]
    print('{} collinear columns removed with threshold: {}.'.format(n_collinear,
                                                                          correlation_threshold))
    
    total_removed = n_non_unique_columns + n_missing_cols + n_zero_variance_cols + n_collinear
    
    print('Total columns removed: ', total_removed)

    if show_cols_to_remove:
        print("+ missing cols:", missing_cols)
        print("+ zero variance cols:", zero_variance_cols)
        print("+ missing cols:", missing_cols)
        print("+ collinear cols:", to_drop)
    
    print('Shape after feature selection: {}.'.format(feature_matrix.shape))
    return feature_matrix

def corr_check_with_label(df, label = "label", ignore_na = False):
    corr = {}
    for c in df.columns:
        if c == label:
            continue
        if ignore_na:
            tmp = df[[c, label]].corr()
            if tmp.shape == (2,2):
                corr[c] = tmp.iloc[0,-1]
            else:
                continue
        else:
            try:
                corr[c] = np.corrcoef(df[c].values, df[label].values)[0][-1]
            except:
                pass
           
    print("num cols calculated corr = ", len(corr))
    print("min corr:", min (x for x in corr.values() if ~np.isnan(x) ))
    print("max corr:", max (x for x in corr.values() if ~np.isnan(x) ))

    corr = dict( sorted(corr.items(), key=lambda x: np.abs(x[1]), reverse=True) )  
    return corr


def corr_get_top_pair(data):
    corr_matrix = data.corr().abs()

    #the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
    sol = pd.DataFrame(corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                     .stack()
                     .sort_values(ascending=False))
    sol.columns = ["corr"]
    return sol



def corr_color_all_pair(data, threshold=0.4, color_value="red") : 
    """
        corr_color_all_pair(data[feature_cols], threshold=0.025, color_value = "orange")
        
    """
    def color_func(corr_value) : 
        if (abs(corr_value) >= threshold) & (abs(corr_value)!= 1) : 
            color_ = color_value
        else :
            color_ = "white"
        return 'background-color: %s' % color_
    tmp = data.corr()
    tmp = tmp.style.applymap(color_func)
    return tmp





def convert_timestamp_to_date(milliseconds):
    '''
        be careful, make sure it's timestamp or unix_timestamp
    '''
    if type(milliseconds) == str or  milliseconds is None:
        try:
            milliseconds = int(milliseconds)
        except:
            return datetime(1900,1,1)
        
    date = datetime.fromtimestamp(milliseconds/1000.0)
    date = date.strftime('%Y-%m-%d %H:%M:%S')
 
    return date



def remove_accents(input_str):
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s