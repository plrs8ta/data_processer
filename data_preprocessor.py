# %%
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import os


# Vars
# File Path
PATH_CSV_FILE = Path(r'C:\Users\EDY\Desktop\PROJECT\PI\cloudsail\database\aggregated_database\aggregated_data_60s(test_51turbines_ori).csv')
PATH_PARQUET_FILE = Path(r'C:\Users\EDY\Desktop\PROJECT\PI\cloudsail\database\aggregated_database\aggregated_data_60s(test_51turbines_ori).parquet')
# Column mapping, should be depand on original data columns
DICT_COLUMN_MAPPING = {
    '本地时间戳': 'time',  # imoprtant var, change it necessary
    '远程时间戳': 'time_remote',
    # 'turbid': 'turbine_id', 
    '功率': 'power_avg', # turbine power, not grid power
    '转速': 'generatorspeed_avg', 
    '桨叶1角度': 'pitchangle1_avg', 
    '风速1': 'windspeed_avg', 
    '风速2': 'windspeed_avg_2', 
    'ivanediiection_avg': 'winddirection_relative_avg', 
    'iwinddirection_avg': 'winddirection_avg', 
    '空气密度': 'airdensity_avg', 
    'inacellepositionltd_avg': 'yawposition_avg', 
}
# DICT_COLUMN_MAPPING = {
#     'rectime': 'time',  # imoprtant var, change it necessary
#     'turbid': 'turbine_id', 
#     'igenpower_avg': 'power_avg', # turbine power, not grid power
#     'igenspeed_avg': 'generatorspeed_avg', 
#     'ipitchangle1_avg': 'pitchangle1_avg', 
#     'iwindspeed_avg': 'windspeed_avg', 
#     'ivanediiection_avg': 'winddirection_relative_avg', 
#     'iwinddirection_avg': 'winddirection_avg', 
#     'iairdensity_avg': 'airdensity_avg', 
#     'inacellepositionltd_avg': 'yawposition_avg', 
# }
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

def csv2parquet(df: pl.DataFrame = None, 
                path_csv_file: Path = PATH_CSV_FILE, 
                path_save_file: Path = PATH_PARQUET_FILE
                ) -> pl.DataFrame: 
    """
    Read .csv data and covert to .parques data

    Args:
        df (pl.DataFrame, optional): If DataFrame data exist, convert it to .parquet. Defaults to None.
        path_csv_file (path, optional): .csv file to convert. Defaults to PATH_CSV_FILE.
        path_save_file (path, optional): Save file path. Defaults to PATH_PARQUET_FILE.

    Returns:
        
    """

    if df is None:
        df = pl.read_csv(path_csv_file, 
                         ignore_errors=True, 
                         try_parse_dates=True
                         )
    print(df.dtypes)

    df.write_parquet(path_save_file)
    print(f"Data has been converted and saved to {path_save_file}")

    return df


def rename_columns_in_parquet_and_save(df: pl.DataFrame = None, 
                                       dict_column_mapping: dict = DICT_COLUMN_MAPPING, 
                                       path_save_file: Path = PATH_PARQUET_FILE
                                       ) -> pd.DataFrame:
    """
    Rename columns by inputs columns name mapper dict or default

    Args:
        df (pl.DataFrame, optional): Rename data. Defaults to None.
        dict_column_mapping (dict, optional): Rename mapping dict. Defaults to DICT_COLUMN_MAPPING.
        path_save_file (path, optional): Save file path. Defaults to PATH_PARQUET_FILE.

    Returns:
        pd.DataFrame: Renamed data
    """
    
    if df is None:
        df = pd.read_parquet(PATH_PARQUET_FILE)    

    df = df.rename(dict_column_mapping)

    df.write_parquet(path_save_file)
    print(f"Columns have been renamed and the file has been saved to {path_save_file}")

    return df

