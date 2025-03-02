# %%
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.csv as pv_csv
import pyarrow.parquet as pq
from pathlib import Path
import os
# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=True)

# Vars
# File Path
PATH_CSV_FILE = Path(r'C:\Users\EDY\Desktop\PROJECT\PI\cloudsail\database\aggregated_database\aggregated_data_60s(test_51turbines_ori).csv')
PATH_PARQUET_FILE = Path(r'C:\Users\EDY\Desktop\PROJECT\PI\cloudsail\database\aggregated_database\aggregated_data_60s(test_51turbines_ori).parquet')
# Column mapping, should be depand on original data columns
DICT_COLUMN_MAPPING = {
    'rectime': 'time',  # imoprtant var, change it necessary
    'turbid': 'turbine_id', 
    'igenpower_avg': 'power_avg', # turbine power, not grid power
    'igenspeed_avg': 'generatorspeed_avg', 
    'ipitchangle1_avg': 'pitchangle1_avg', 
    'iwindspeed_avg': 'windspeed_avg', 
    'ivanediiection_avg': 'winddirection_relative_avg', 
    'iwinddirection_avg': 'winddirection_avg', 
    'iairdensity_avg': 'airdensity_avg', 
    'inacellepositionltd_avg': 'yawposition_avg', 
}
# Time column name
COLUMN_TIME = 'rectime'    # should be repalce by original data time clomun name if wanna change time type to pyarrow time type
# analysis reley on basic columns
LIST_BASIC_COLUMN = [
    'time','turbine_id', # 时间，机组编号
    'windspeed_avg', 
    # 'windspeed_max', 'windspeed_min', 'windspeed_std', # 风速的平均值，最大值，最小值和标准差
    'winddirection_avg',
    # 'winddirection_max', 'winddirection_min', 'winddirection_std', # 风向的平均值，最大值，最小值和标准差
    'winddirection_relative_avg',
    # 'winddirection_relative_max', 'winddirection_relative_min', 'winddirection_relative_std', # 相对风向（测量风向-机舱位置）的平均值，最大值，最小值和标准差
    'yawposition_avg', # 机舱位置的平均值
    'pitchangle1_avg', 
    # 'pitchangle2_avg', 'pitchangle3_avg', # 3个桨距角的平均值
    'generatorspeed_avg', 
    # 'generatortorque_avg', # 发电机转速鸡发电机转矩的平均值
    'power_avg', 
    # 'power_max', 'power_min', 'power_std', # 发电功率的平均值，最大值，最小值和标准差
    # 'airtemperature_avg', 'airpressure_avg', 'airhumidity_avg', 
    'airdensity_avg', # 大气温度，大气压力，大气湿度，大气密度的平均值
    'operatingmode_cntmax', 
    # 'operatingmode_cnt', 
    'yawmode_cntmax', 
    # 'yawmode_cnt' 
    'brakemode_cntmax',
    # 'brakemode_cnt'# 运行模式、偏航模式和刹车模式的最大出现次数及其次数
]
# Rated power
RATED_POWER = 2100

## key columns might be not necessary
## analysis reley on key columns
# self.key_cols_list = [
#     'time','turbine_id', # 时间，机组编号
#     'windspeed_avg', 'windspeed_max', 'windspeed_min', 'windspeed_std', # 风速的平均值，最大值，最小值和标准差
#     'winddirection_avg', 'winddirection_max', 'winddirection_min', 'winddirection_std', # 风向的平均值，最大值，最小值和标准差
#     'winddirection_relative_avg', 'winddirection_relative_max', 'winddirection_relative_min', 'winddirection_relative_std', # 相对风向（测量风向-机舱位置）的平均值，最大值，最小值和标准差
#     'yawposition_avg', # 机舱位置的平均值
#     'pitchangle1_avg', 'pitchangle2_avg', 'pitchangle3_avg', # 3个桨距角的平均值
#     'generatorspeed_avg', 'generatortorque_avg', # 发电机转速发电机转矩的平均值
#     'power_avg', 'power_max', 'power_min', 'power_std', # 发电功率的平均值，最大值，最小值和标准差
#     'airtemperature_avg', 'airpressure_avg', 'airhumidity_avg','airdensity_avg', # 大气温度，大气压力，大气湿度，大气密度的平均值
#     'operatingmode_cntmax', 'operatingmode_cntmaxcnt', 'yawmode_cntmax', 'yawmode_cntmaxcnt', 'limitmode_cntmax', 'limitmode_cntmaxcnt' # 运行模式，偏航模式和限电模式的最大出现次数及其次数
# ]
# self.key_floatcols_list = [
#     'windspeed_avg', 'windspeed_max', 'windspeed_min', 'windspeed_std', # 风速的平均值，最大值，最小值和标准差
#     'winddirection_avg', 'winddirection_max', 'winddirection_min', 'winddirection_std', # 风向的平均值，最大值，最小值和标准差
#     'winddirection_relative_avg', 'winddirection_relative_max', 'winddirection_relative_min', 'winddirection_relative_std', # 相对风向（测量风向-机舱位置）的平均值，最大值，最小值和标准差
#     'yawposition_avg', # 机舱位置的平均值
#     'pitchangle1_avg', 'pitchangle2_avg', 'pitchangle3_avg', # 3个桨距角的平均值
#     'generatorspeed_avg', 'generatortorque_avg', # 发电机转速及发电机转矩的平均值
#     'power_avg', 'power_max', 'power_min', 'power_std', # 发电功率的平均值，最大值，最小值和标准差
#     'airtemperature_avg', 'airpressure_avg', 'airhumidity_avg','airdensity_avg', # 大气温度，大气压力，大气湿度，大气密度的平均值
# ]
# self.key_intcols_list = [
#     'operatingmode_cntmax', 'operatingmode_cntmaxcnt', 'yawmode_cntmax', 'yawmode_cntmaxcnt', 'limitmode_cntmax', 'limitmode_cntmaxcnt', 'brakemode_cntmax', 'brakemode_cnt' # 运行模式，偏航模式和限电模式的最大出现次数及其次数
# ]

def csv2parquet(df:pd.DataFrame=None, 
                column_time:int=COLUMN_TIME, 
                path_csv_file:Path=PATH_CSV_FILE, 
                path_save_file:Path=PATH_PARQUET_FILE
                ) -> pd.DataFrame: 
    """
    Read .csv data and covert to .parques data

    Args:
        df (pd.DataFrame, optional): If DataFrame data exist, convert it to .parquet. Defaults to None.
        path_csv_file (path, optional): .csv file to convert. Defaults to PATH_CSV_FILE.
        path_save_file (path, optional): Save file path. Defaults to PATH_PARQUET_FILE.

    Returns:
        pd.DataFrame: PyArrow DataFrame
    """

    if df is None:
        df = pd.read_csv(path_csv_file)
    print(df.dtypes)

    if not pd.api.types.is_datetime64_any_dtype(df[column_time]):
        df[column_time] = pd.to_datetime(df[column_time])

    df = pa.Table.from_pandas(df)
    df = df.to_pandas(types_mapper=pd.ArrowDtype) 

    df.to_parquet(path_save_file)
    print(f"Data has been converted and saved to {path_save_file}")

    return df

def largecsv2multiparquet(df:pd.DataFrame=None, 
                     column_time:str='rectime', 
                     path_largecsv_file:Path=Path(r"D:\temp_data\my2000_onedata_202501231426.csv"), 
                     chunk_size:int=1000000, 
                     path_save_dir:Path=Path(r"D:\temp_data")):
    chunks = pd.read_csv(path_largecsv_file, chunksize=chunk_size)
    file_count = 0
    for chunk in chunks:

        if not pd.api.types.is_datetime64_any_dtype(chunk[column_time]):
            chunk[column_time] = pd.to_datetime(chunk[column_time])

        table = pa.Table.from_pandas(chunk)
        df = table.to_pandas(types_mapper=pd.ArrowDtype)

        output_file = os.path.join(path_save_dir, f"my2000_onedata_202501231426_{file_count}.parquet")
        df.to_parquet(output_file)
        file_count += 1
        print(f"Saving file {output_file}")



def rename_columns_in_parquet_and_save(df:pd.DataFrame=None, 
                            dict_column_mapping:dict=DICT_COLUMN_MAPPING, 
                            path_save_file:Path=PATH_PARQUET_FILE
                            ) -> pd.DataFrame:
    """
    Rename columns by inputs columns name mapper dict or default

    Args:
        df (pd.DataFrame, optional): Rename data. Defaults to None.
        dict_column_mapping (dict, optional): Rename mapping dict. Defaults to DICT_COLUMN_MAPPING.
        path_save_file (path, optional): Save file path. Defaults to PATH_PARQUET_FILE.

    Returns:
        pd.DataFrame: Renamed data
    """
    
    if df is None:
        df = pd.read_parquet(PATH_PARQUET_FILE)    

    df.rename(columns=dict_column_mapping, inplace=True)            

    df.to_parquet(path_save_file)
    print(f"Columns have been renamed and the file has been saved to {path_save_file}")

    return df

def sort_values_and_save(df:pd.DataFrame=None, 
                         list_sort_column:list=['turbine_id', 'time'], 
                         ascending:bool=True, 
                         path_save_file:Path=PATH_PARQUET_FILE
                         ) -> pd.DataFrame:
    """
    Sort values and save data

    Args:
        df (pd.DataFrame, optional): Sort data. Defaults to None.
        list_sort_column (list, optional): Sort columns. Defaults to ['turbine_id', 'time'].
        ascending (bool, optional): Ascending type. Defaults to True.
        path_save_file (path, optional): Save file path. Defaults to PATH_PARQUET_FILE.

    Returns:
        pd.DataFrame: Sorted data
    """
    if df is None:
        df = pd.read_parquet(PATH_PARQUET_FILE)   

    df = df.sort_values(list_sort_column, ascending=ascending)

    df = df.reset_index(drop=False)

    # this line might be not necessary, make sure all types convert to pyarrow types
    df = pa.Table.from_pandas(df)
    df = df.to_pandas(types_mapper=pd.ArrowDtype)

    df.to_parquet(path_save_file)
    print(f"Data sorted and saved to {path_save_file}")

    return df
    
def label_timeseria_type(df:pd.DataFrame=None, 
        time_interval:int=60
        ) -> pd.DataFrame:
    """
    Label time seria type

    Args:
        df (pd.DataFrame, optional): Data to label time seria type. Defaults to None.
        time_interval (int, optional): Time seria interval. Defaults to 60s.

    Returns:
        pd.DataFrame: Labeled time seria type data
    """

    if df is None:
        df = pd.read_parquet(PATH_PARQUET_FILE)

    timeseria_problem_data_dict = {}

    if not df[['turbine_id', 'time']].equals(df[['turbine_id', 'time']].sort_values(['turbine_id', 'time'])):
        df = df.sort_values(['turbine_id', 'time'])

    # 'turbine_id' and 'time' combine null type
    df['label_idtime_null'] = df['turbine_id'].isnull() | df['time'].isnull()   # 0: no NaN; 1: contain NaN
    timeseria_problem_data_dict['idtime_nan_data'] = df[df['label_idtime_null']]

    # 'turbine_id' and 'time' combine duplicate type
    is_idtime_duplicated_seria = df[['turbine_id', 'time']].duplicated(keep=False)
    is_idtime_duplicated_butfirst_seria = df[['turbine_id', 'time']].duplicated(keep=False) & (~df[['turbine_id', 'time']].duplicated(keep='first'))
    is_idtime_duplicated_notfirst_seria = df[['turbine_id', 'time']].duplicated(keep='first')
    # 'turbine_id' and 'time' combine duplicate condictions list
    label_idtime_duplicated_conditions = [
        ~is_idtime_duplicated_seria, 
        is_idtime_duplicated_butfirst_seria, 
        is_idtime_duplicated_notfirst_seria, 
    ]
    # 'turbine_id' and 'time' combine duplicate label list
    label_idtime_duplicated_choices = [0,   # 0: no duplicate
                                       1,   # duplicate but first
                                       2    # 2: duplicate and not first
                                       ]
    # 'turbine_id' and 'time' combine mapping duplicate condiction and duplicate label
    df['label_idtime_duplicated'] = np.select(
        label_idtime_duplicated_conditions,
        label_idtime_duplicated_choices,
        default=0,
    )
    timeseria_problem_data_dict['idtime_duplicated_data'] = df[df['label_idtime_duplicated'] != 0]
    
    # 'turbine_id' and 'time' combine continuety type
    # sign time interval >= time_interval data
    df['time_interval'] = df.groupby('turbine_id')['time'].transform(lambda x: (x - x.shift()).fillna(pd.Timedelta(seconds=0)))
    # label the continuety type
    df['label_idtime_continuous_type'] = ((df['time_interval'] != pd.Timedelta(seconds=time_interval)) & (df['time_interval'] != pd.Timedelta(seconds=0)))   # 0: continuety; 1: not continuety
    # create not continue data
    df_notcontinuity_data = df.loc[(df['label_idtime_continuous_type']), ['turbine_id', 'time', 'time_interval']]
    df_notcontinuity_data['start_time'] = df_notcontinuity_data['time'] - df_notcontinuity_data['time_interval']
    # record not continue data
    timeseria_problem_data_dict['idtime_notcontinuity_data'] = df_notcontinuity_data

    print(timeseria_problem_data_dict)

    df.drop(columns=['time_interval'], inplace=True)
    
    df['label_idtime_null'] = df['label_idtime_null'].astype(int)   # Convert True/False to 1/0
    df['label_idtime_continuous_type'] = df['label_idtime_continuous_type'].astype(int)  # Convert True/False to 1/0
    # this line might be not necessary, make sure all types convert to pyarrow types
    df = pa.Table.from_pandas(df)
    df = df.to_pandas(types_mapper=pd.ArrowDtype)

    return df

def label_outliers_type(df:pd.DataFrame=None, 
        column_float_type:list=['windspeed_avg', 'winddirection_avg'], 
        continuety_num:int=3, 
        ) -> pd.DataFrame:
    """
    Label outliers type

    Args:
        df (pd.DataFrame, optional): Data to label outliers type. Defaults to None.
        column_float_type (list, optional): Float type column list. Defaults to LIST_BASIC_COLUMN.
        continuety_num (int, optional): Value continuety occur times. Defaults to 3.

    Returns:
        pd.DataFrame: Data with labeled outliers type
    """

    if df is None:
        df = pd.read_parquet(PATH_PARQUET_FILE)

    outlier_problem_dict = {}
    
    # Label rows based on the type of NaN values
    num_col_list = df.drop(columns=['time','turbine_id']).columns.tolist()

    is_null_seria = df[num_col_list].isnull().any(axis=1)
    is_littlenull_seria = df[num_col_list].isnull().any(axis=1) & (~df[num_col_list].isnull().all(axis=1))
    is_allnull_seria = df[num_col_list].isnull().all(axis=1)
    label_nan_conditions = [
        ~is_null_seria,         # 0: No NaN values
        is_littlenull_seria,    # 1: Some NaN values but not all
        is_allnull_seria,       # 2: All values are NaN
    ]
    label_nan_choices = [0, 1, 2]
    df['label_NaN_type'] = np.select(
        label_nan_conditions,
        label_nan_choices,
        default=0,
    )
    outlier_problem_dict['NaN_data'] = df[df['label_NaN_type'] != 0]
    
    # todo: Pandarallel is not satisfy with def, change it
    """
    Label rows with repeated values in continuous time series.
    24.8.6 Repeated values need to be confirmed with data collection to determine data change frequency
    """
    col_float_set = set(df.columns) & set(column_float_type)
    condiction_idtime_notnull = (df['label_idtime_null'] == 0)
    condiction_idtime_notduplicated = (df['label_idtime_duplicated'] != 2)
    df_timeseria_filtered = df[condiction_idtime_notnull & condiction_idtime_notduplicated].sort_values(['turbine_id', 'time'])
    for i in col_float_set:
        df['continue_duplicate_type_' + i] = (df[i] == df[i].shift(-1)) & (df[i] == df[i].shift(-2))    
        df.loc[df['continue_duplicate_type_' + i] == True, 'continue_duplicate_type_' + i].shift(1) == True
        df.loc[df['continue_duplicate_type_' + i] == True, 'continue_duplicate_type_' + i].shift(2) == True
        df.loc[df['continue_duplicate_type_' + i] == None, 'continue_duplicate_type_' + i] == False
    # df_continue_duplicate_problem = df_timeseria_filtered.groupby('turbine_id')[list(col_float_set)].rolling(window=continuety_num, min_periods=continuety_num).parallel_apply(
    #         lambda y: len(set(y)) == 1).shift(
    #             -(continuety_num - 1)).transform(
    #                 lambda x: ((x.shift((continuety_num - 1)) == 1) | (x == 1)))
    # df_continue_duplicate_problem.columns = ["continue_duplicate_type_" + col for col in df_continue_duplicate_problem.columns]
    # df_continue_duplicate_problem.index = df_continue_duplicate_problem.index.get_level_values(1)
    # outlier_problem_dict['continue_duplicate_problem'] = df_continue_duplicate_problem

    # todo: get every measure value solid
    # Label rows with out-of-range wind speed.
    overrange_windspeed = (df['windspeed_avg'] >= 30) | (df['windspeed_avg'] < 0)
    
    # print(outlier_problem_dict)

    # df = pd.concat([df, df_continue_duplicate_problem], axis=1)

    # this line might be not necessary, make sure all types convert to pyarrow types
    df = pa.Table.from_pandas(df)
    df = df.to_pandas(types_mapper=pd.ArrowDtype)
    
    return df

# todo: fix from this part
def label_situations(df: pd.DataFrame,
    rated_power:int=RATED_POWER,
    pitch_limit:int=4,
) -> pd.DataFrame:
    """
    打特殊工况标签。
    24.8.12 调频限电目前没有限电标志位，
    24.8.12 惯量调频。

    Args:
        df (pd.DataFrame): 打完异常数据标签后的数据。
        rated_wind_speed (int, optional): 额定风速。 Defaults to 10.
        pitch_limit (int, optional): 变桨限值。 Defaults to 4.

    Returns:
        tuple[pd.DataFrame, dict]: 打完工况后的标签。
    """
    
    situation_dict = {}
    # 24.8.12 艳萍：与集团确认最佳桨距角?
    limited_pitch_level1 = (df['power_avg'] <= rated_power) & (df['pitchangle1_avg'] >= pitch_limit)
    limited_pitch_level2 = (df['power_avg'] <= rated_power * 0.5) & (df['pitchangle1_avg'] >= pitch_limit * 0.5)
    limited_pitch_level3 = (df['power_avg'] <= rated_power * 0.25) & (df['pitchangle1_avg'] >= pitch_limit * 0.25)
    limited_pitch_condictions = [
        (~limited_pitch_level1) & (~limited_pitch_level2) & (~limited_pitch_level3),
        limited_pitch_level1,
        limited_pitch_level2,
        limited_pitch_level3,
    ]
    limited_pitch_choices = [0, # No limite
                             1, # Limite 
                             2, # Half limite
                             3  # 1/4 limite
                             ]
    df['label_condiction_limited'] = np.select(
        limited_pitch_condictions,
        limited_pitch_choices,
        default=0,
    )
    
    situation_dict['limited'] = df[df['label_condiction_limited'] != 0]
    
    if situation_dict['limited'].empty:
        print("没有限电工况。")
        
    print("工况字典: \n", situation_dict)
    
    return df

def label_operation(df: pd.DataFrame) -> pd.Series:
    """
    对风向数据进行分类标签化处理。

    Args:
        df (pd.DataFrame): 包含风向数据的 DataFrame。

    Returns:
        pd.Series: 含有风向分类标签的新列，标签为区间范围。
    """

    # os_dict = {}

    # 创建风向划分的区间（每 22.5 度一个区间），从 0 到 360 度。
    wdb = np.arange(
        0,
        382.5,
        22.5
    )
    ## 根据区间边界创建标签。例如，(0, 22.5)、(22.5, 45) 等。
    # wdb_label = [
    #     f'({wdb[i]}, {wdb[i+1]})'
    #     for i in range(len(wdb)-1)
    # ]
    # Using int inplace real bin like (0, 22.5)、(22.5, 45)
    wdb_label = [i for i in range(16)]
    # 使用 pd.cut 函数将 'winddirection_avg' 列的数据划分到上述区间，并赋予相应的标签。
    df['label_wdb'] = pd.cut(
        df['winddirection_avg'],
        bins=wdb,
        labels=wdb_label,
        right=True,
        include_lowest=False
    )
    
    npb = np.arrage(
        0, 
        382.5, 
        22.5
    )
    # npb_label = [
    #     f'({npb[i]}, {npb[i+1]})'
    #     for i in range(len(npb)-1)
    # ]
    npb_label = [i for i in range(16)]
    df['label_npb'] = pd.cut(
        df['yawposition_avg'], 
        bins=npb, 
        labels=npb_label, 
        right=True, 
        include_lowest=False
    )

    # 返回新生成的风向分类标签列
    return df


def integrity_validation(
    df: pd.DataFrame,
    validation_windspeedbinvalue: float = 0.5,
    validation_windspeedmax: float = 15.25,
    validation_windspeedmin: float = 2.25,
    validation_wholecumtime: float = 180.0 * 60.0 * 60.0,
    validation_bincumtime: float = 0.5 * 60.0 * 60.0,
) -> pd.DataFrame:
    """
    验证数据完整性，包括风速分箱和累计时间检查。

    Args:
        df (pd.DataFrame): 需要验证的数据集，包含 'windspeed_avg', 'time_diff', 和 'turbine_id' 列。
        validation_windspeedbinvalue (float, optional): 风速分箱的步长。 Defaults to 0.5.
        validation_windspeedmax (float, optional): 风速分箱的最大值。 Defaults to 15.25.
        validation_windspeedmin (float, optional): 风速分箱的最小值。 Defaults to 2.25.
        validation_wholecumtime (float, optional): 累计时间的最小阈值（秒）。 Defaults to 180*60*60 (5 hours).
        validation_bincumtime (float, optional): 每个风速分箱的最小累计时间（秒）。 Defaults to 0.5*60*60 (30 minutes).

    Returns:
        tuple[pd.DataFrame, dict]: 包含满足条件的 DataFrame 和完整性问题字典。
    """

    # 初始化用于存储完整性问题的字典。
    integrity_problem_dict = {}
    
    # 创建风速分箱区间，并生成对应的标签。
    validation_windspeedbin = np.arange(
        validation_windspeedmin, 
        validation_windspeedmax, 
        validation_windspeedbinvalue
    )
    validation_bins_label = [
        f'({validation_windspeedbin[i]}, {validation_windspeedbin[i+1]})'
        for i in range(len(validation_windspeedbin)-1)
    ]
    # 将 'windspeed_avg' 列的数据划分到指定的风速分箱中，并生成标签。
    df['label_windspeedbin'] = pd.cut(
        df['windspeed_avg'],
        bins=validation_windspeedbin,
        labels=validation_bins_label,
        right=True,
        include_lowest=False,
    )
    
    df['time_interval'] = df.groupby('turbine_id')['time'].transform(lambda x: (x - x.shift()).fillna(pd.Timedelta(seconds=0)))
    # 计算每个机组的累计时间，并与设定的最小累计时间进行比较。
    wholecumtime = df.groupby('turbine_id')['time_interval'].sum()
    validation_wholecumtime = pd.Timedelta(seconds=validation_wholecumtime)
    condiction_wholecumtime = wholecumtime >= validation_wholecumtime
    wholecumtime_satisfy_idlist = wholecumtime[condiction_wholecumtime].index.tolist()
    
    # 筛选出累计时间满足条件的风机数据。
    df_wholecumtime_satisfy = df[df['turbine_id'].isin(wholecumtime_satisfy_idlist)]
    
    # 记录不满足累计时间条件的风机数据。
    integrity_problem_dict['whole'] = df[~df['turbine_id'].isin(wholecumtime_satisfy_idlist)]
    
    # if integrity_problem_dict['whole'].empty:
    #     print("Whole integrity validation pass.")
    
    # 计算满足条件的风机数据的风速分箱累计时间，并与设定的最小累计时间进行比较。
    bincumtime = df_wholecumtime_satisfy.groupby(['turbine_id', 'label_windspeedbin'], observed=True)['time_interval'].sum()
    validation_bincumtime = pd.Timedelta(seconds=validation_bincumtime)
    condiction_bincumtime = bincumtime >= validation_bincumtime
    # bincumtime_notsatisfy_indexlist = bincumtime[~condiction_bincumtime].index.tolist()
    bincumtime_satisfy_idlist = bincumtime[condiction_bincumtime].index.get_level_values(0).tolist()
    
    ## 筛选出风速分箱累计时间满足条件的风机数据。
    df_wholebincumtime_satisfy = df_wholecumtime_satisfy[df_wholecumtime_satisfy['turbine_id'].isin(bincumtime_satisfy_idlist)]
    
    # 记录不满足风速分箱累计时间条件的风机数据。
    integrity_problem_dict['bin'] = df[~df['turbine_id'].isin(bincumtime_satisfy_idlist)]
    
    # 新增一列'label_integirity_validation'，标记出现的'turbine_id'为0，未出现的为1
    df['label_integirity_validation'] = np.where(df['turbine_id'].isin(wholecumtime_satisfy_idlist), 0, 1)

    df.drop(columns=['label_windspeedbin', 'time_interval'], inplace=True)
    # this line might be not necessary, make sure all types convert to pyarrow types
    df = pa.Table.from_pandas(df)
    df = df.to_pandas(types_mapper=pd.ArrowDtype)

    # 输出完整性问题字典，并返回满足条件的数据及问题字典。
    print(integrity_problem_dict)

    return df

def windspeed_normalized_byairdensity(
    df: pd.DataFrame,
    R0: float = 287.05,
    Rw: float = 461.5,
    B_10min: float = 101325,
    phi: float = 0.5,
) -> tuple[pd.Series, pd.Series]:
    """
    根据空气密度对风速进行归一化处理。

    Args:
        df (pd.DataFrame): 包含风速 ('windspeed_avg') 和空气温度 ('airtemperature_avg') 数据的 DataFrame。
        R0 (float, optional): 气体常数（干空气），默认为 287.05 J/(kg·K)。
        Rw (float, optional): 气体常数（水蒸气），默认为 461.5 J/(kg·K)。
        B_10min (float, optional): 标准大气压，默认为 101325 Pa。
        phi (float, optional): 相对湿度，默认为 0.5。

    Returns:
        tuple[pd.Series, pd.Series]: 包含计算得到的空气密度和归一化风速的 Series。
    """
    
    # def airdensity_compute(T_10min_C: float) -> float:

    #     # 空气温度转换为开尔文温度。
    #     # T_10min_K = T_10min_C + 273.15
    #     # Pw = 0.0000205 * np.exp(0.0631846 * T_10min_K)
    #     # 计算空气密度（使用公式：ρ = (P0 / (R0 * T)) - φ * (Pw / (Rw * T)))
    #     rho_10min = (1 / (T_10min_C + 273.15)) * (B_10min / R0 - phi * 0.0000205 * np.exp(0.0631846 * (T_10min_C + 273.15)) * (1 / R0 - 1 / Rw))
    
    #     return rho_10min
    
    # # 标准空气密度（海平面标准大气）
    # rho0 = 1.225
    
    # # 使用 transform 函数应用 airdensity_compute 函数计算每个记录的空气密度
    # df['air_density_bycompute'] = df['airtemperature_avg'].transform(airdensity_compute)
    
    # 计算归一化风速，使用公式：Vn = Va * (ρ/ρ0)^(1/3)
    df['normalized_windspeed_byairdensity'] = df['windspeed_avg'] * ((df['airdensity_avg']/rho0) ** (1/3))
    
    return df

def power_curve_compute(
    df: pd.DataFrame,
    normalized_windspeedbinvalue: float = 0.5,
    normalized_windspeedmax: float = 15.25,
    normalized_windspeedmin: float = 2.25,
) -> pd.DataFrame:
    """
    计算功率曲线值，包括风速分箱和功率统计量。

    Args:
        df (pd.DataFrame): 包含归一化风速 ('normalized_windspeed_byairdensity') 和功率 ('power_avg') 数据的 DataFrame。
        normalized_windspeedbinvalue (float, optional): 归一化风速分箱的步长。 Defaults to 0.5.
        normalized_windspeedmax (float, optional): 归一化风速分箱的最大值。 Defaults to 15.25.
        normalized_windspeedmin (float, optional): 归一化风速分箱的最小值。 Defaults to 2.25.

    Returns:
        pd.DataFrame: 包含每个风速分箱的风速均值、功率均值、功率均方根偏差和计数的 DataFrame。
    """
    # 创建归一化风速的分箱区间，并生成对应的标签。
    normalized_windspeedbin = np.arange(
        normalized_windspeedmin,
        normalized_windspeedmax,
        normalized_windspeedbinvalue
    )
    normalized_bins_label = [
        f'({normalized_windspeedbin[i]}, {normalized_windspeedbin[i+1]})'
        for i in range(len(normalized_windspeedbin)-1)
    ]

    # 将 'normalized_windspeed_byairdensity' 列的数据划分到指定的风速分箱中，并生成标签。
    df['label_normalized_windspeedbin_byairdensity'] = pd.cut(
        df['normalized_windspeed_byairdensity'],
        bins=normalized_windspeedbin,
        labels=normalized_bins_label,
        right=True,
        include_lowest=False,
    )
    
    # 按风机ID和风速分箱计算风速均值、功率均值、功率均方根偏差和计数。
    normalized_windspeedbin_byairdensity_windspeed_mean = df.groupby(['turbine_id', 'label_normalized_windspeedbin_byairdensity'], observed=False)['normalized_windspeed_byairdensity'].mean()
    normalized_windspeedbin_byairdensity_power_mean = df.groupby(['turbine_id', 'label_normalized_windspeedbin_byairdensity'], observed=True)['power_avg'].mean()
    normalized_windspeedbin_byairdensity_power_rmsd = df.groupby(['turbine_id', 'label_normalized_windspeedbin_byairdensity'], observed=True)['power_avg'].apply(lambda x: np.sqrt(((x - x.mean()) ** 2).mean()))
    normalized_windspeedbin_byairdensity_power_cnt = df.groupby(['turbine_id', 'label_normalized_windspeedbin_byairdensity'], observed=True)['power_avg'].count()
    # 合并计算结果，并重命名列名。
    df_normalized_windspeedbin_byairdensity_agg = pd.concat([normalized_windspeedbin_byairdensity_windspeed_mean, normalized_windspeedbin_byairdensity_power_mean, normalized_windspeedbin_byairdensity_power_rmsd, normalized_windspeedbin_byairdensity_power_cnt], axis=1)
    df_normalized_windspeedbin_byairdensity_agg.columns = ['wind_mean', 'power_mean', 'power_rmsd', 'cnt']

    return df_normalized_windspeedbin_byairdensity_agg


def calculate_AEP(df: pd.DataFrame, df_windspeedbin_agg: pd.DataFrame) -> pd.Series:
    """
    计算等效发电性能（AEP）。

    Args:
        df (pd.DataFrame): 包含风速分箱标签 ('label_normalized_windspeedbin_byairdensity') 的 DataFrame。
        df_windspeedbin_agg (pd.DataFrame): 包含每个风速分箱的统计信息的 DataFrame，需包含 'power_mean' 列。

    Returns:
        pd.Series: 每个风机的等效发电性能（AEP）。
    """

    # 计算每个风机在每个风速分箱中的样本计数。
    df_agg1_cnt = df.groupby(['turbine_id', 'label_normalized_windspeedbin_byairdensity'])['label_normalized_windspeedbin_byairdensity'].count()
    # 计算每个风机的总样本计数。
    df_agg_cnt = df.groupby(['turbine_id'])['label_normalized_windspeedbin_byairdensity'].count()
    
    # 计算每个风速分箱的样本概率。
    df_probability = df_agg1_cnt / df_agg_cnt
    # 将风速分箱的统计信息与样本概率合并。
    df_windspeedbin_agg1 = df_windspeedbin_agg.join(df_probability)
    # 计算功率与概率的乘积，并保存到新列 'pp'。
    df_windspeedbin_agg1['pp'] = df_windspeedbin_agg1['power_mean'] * df_windspeedbin_agg1['label_normalized_windspeedbin_byairdensity']
    # 按风机 ID 汇总 'pp' 列，得到每个风机的等效发电性能（AEP）。
    df_AEP = df_windspeedbin_agg1.groupby('turbine_id')['pp'].sum()
    
    return df_AEP

def calculate_Cp(
    df_windspeedbin_agg: pd.DataFrame, 
    rho0: float = 1.225, 
    rotor_D: float = 82, 
) -> pd.DataFrame: 
    """
    计算功率系数Cp。

    Args:
        df_windspeedbin_agg (pd.DataFrame): 包含风速均值 ('wind_mean') 和功率均值 ('power_mean') 的 DataFrame。
        rho0 (float, optional): 标准空气密度，默认为 1.225 kg/m³。
        rotor_D (float, optional): 风轮直径，默认为 82 米。

    Returns:
        pd.DataFrame: 包含原始统计信息和计算出的功率系数（'Cp'）的 DataFrame。
    """

    # 计算功率系数 Cp。
    # Cp = P / (0.5 * ρ * A * V^3)
    # 其中 P 为功率，ρ 为空气密度，A 为风轮的扫掠面积，V 为风速。
    df_windspeedbin_agg1 = pd.DataFrame()
    df_windspeedbin_agg1 = (df_windspeedbin_agg['power_mean'] * 1000) / (1/2 * rho0 * (np.pi* (rotor_D/2) ** 2) * (df_windspeedbin_agg['wind_mean'] ** 3))
    # 设置计算结果的列名为 'Cp'。
    df_windspeedbin_agg1.name = 'Cp'
    
    # 将计算出的功率系数 Cp 添加到原始 DataFrame 中。
    df_windspeedbin_agg_addcp = df_windspeedbin_agg.join(df_windspeedbin_agg1)
    
    return df_windspeedbin_agg_addcp

def calculate_wind_probability(df):
    # 计算用于分析的机组风频分布，暂用使用所有数据统计。
    # 输入数据目前必须是包含有归一化风速区间的数据。
    df_wind_probability = df['label_normalized_windspeedbin_byairdensity'].value_counts(normalize=True).sort_index()
    return df_wind_probability

def calculate_weight_power(
        df_power_curve, 
        df_wind_probability, 
        ): 
    # 计算加权功率。
    df_values = df_power_curve.droplevel(level=0).loc[:, 'power_mean'].sort_index()
    df_weight_power = df_wind_probability * df_values

    return df_weight_power
# %%

def test(path=PATH_PARQUET_FILE):
    df = pd.read_parquet(path)
    return df