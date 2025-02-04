# %%
import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
import os
# %%

# Vars
# File Path
PATH_CSV_FILE = Path(r'D:\temp_data\SCADAdata_23\aggregated_data_60s(test_51turbines_ori).csv')
PATH_PARQUET_FILE = Path(r'D:\temp_data\SCADAdata_23\aggregated_data_60s(test_51turbines_ori)_test.parquet')
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

# Main data processor fucntions
def csv2parquet(df: pl.DataFrame = None,
                column_time: str = COLUMN_TIME,
                path_csv_file: Path = PATH_CSV_FILE,
                path_save_file: Path = PATH_PARQUET_FILE) -> pl.DataFrame:
    """
    Read .csv data and convert to .parquet data

    Args:
        df (pl.DataFrame, optional): If DataFrame data exists, convert it to .parquet. Defaults to None.
        path_csv_file (Path, optional): .csv file to convert. Defaults to PATH_CSV_FILE.
        path_save_file (Path, optional): Save file path. Defaults to PATH_PARQUET_FILE.

    Returns:
        pl.DataFrame: Polars DataFrame
    """
    if df is None:
        df = pl.read_csv(
            path_csv_file, 
            ignore_errors=True, 
            try_parse_dates=True)
    print(df.dtypes)

    # # Convert time column to datetime
    # df = df.with_columns(pl.col(column_time).str.strptime(pl.Datetime))

    df.write_parquet(path_save_file)
    print(f"Data has been converted and saved to {path_save_file}")

    return df

# def largecsv2multiparquet(df: pl.DataFrame = None,
#                          column_time: str = 'rectime',
#                          path_largecsv_file: Path = Path(r"D:\temp_data\my2000_onedata_202501231426.csv"),
#                          chunk_size: int = 1000000,
#                          path_save_dir: Path = Path(r"D:\temp_data")):
#     """
#     Convert large CSV to multiple parquet files using Polars with streaming
#     """
#     file_count = 0
#     for chunk in pl.read_csv(path_largecsv_file, rechunk=True).iter_chunks(chunk_size):
#         # Convert time column to datetime
#         chunk = chunk.with_columns(pl.col(column_time).str.strptime(pl.Datetime))
        
#         output_file = os.path.join(path_save_dir, f"my2000_onedata_202501231426_{file_count}.parquet")
#         chunk.write_parquet(output_file)
#         file_count += 1
#         print(f"Saving file {output_file}")

# def multiparquet2largeparquet(df: pl.DataFrame = None,
#                             list_path_parquet: list = [Path(f'D:\\temp_data\\my2000_onedata_202501231426_{i}.parquet') for i in range(14)],
#                             path_save_dir: Path = Path(r'D:\temp_data')) -> pl.DataFrame:
#     """
#     Merge multiple parquet files into a single DataFrame using Polars

#     Args:
#         df (pl.DataFrame, optional): DataFrame to merge. Defaults to None.
#         list_path_parquet (list): List of parquet file paths to merge.
#         path_save_dir (Path): Directory to save the merged DataFrame.

#     Returns:
#         pl.DataFrame: Merged DataFrame
#     """
#     # Read and concatenate all parquet files
#     merged_df = pl.concat([pl.read_parquet(path) for path in list_path_parquet])

#     # Save the merged DataFrame
#     merged_file_path = os.path.join(path_save_dir, 'merged_data.parquet')
#     merged_df.write_parquet(merged_file_path)
#     print(f"Merged data has been saved to {merged_file_path}")

#     return merged_df

def rename_columns_in_parquet_and_save(df: pl.DataFrame = None,
                                     dict_column_mapping: dict = DICT_COLUMN_MAPPING,
                                     path_save_file: Path = PATH_PARQUET_FILE) -> pl.DataFrame:
    """
    Rename columns

    Args:
        df (pl.DataFrame, optional): Rename data. Defaults to None.
        dict_column_mapping (dict): Rename mapping dict.
        path_save_file (Path): Save file path.

    Returns:
        pl.DataFrame: Renamed data
    """
    if df is None:
        df = pl.read_parquet(PATH_PARQUET_FILE)

    df = df.rename(dict_column_mapping)

    df.write_parquet(path_save_file)
    print(f"Columns have been renamed and the file has been saved to {path_save_file}")

    return df

def label_timeseria_type(df: pl.DataFrame = None, 
                        time_interval: int = 60) -> pl.DataFrame:
    """
    Label time seria type

    Args:
        df (pl.DataFrame, optional): Data to label time seria type. Defaults to None.
        time_interval (int, optional): Time seria interval. Defaults to 60s.

    Returns:
        pl.DataFrame: Labeled time seria type data
    """
    if df is None:
        df = pl.read_parquet(PATH_PARQUET_FILE)

    timeseria_problem_data_dict = {}

    # Sort by turbine_id and time
    df = df.sort(['turbine_id', 'time'])

    # Label idtime null type
    df = df.with_columns([
        pl.col('turbine_id').is_null().or_(pl.col('time').is_null()).alias('label_idtime_null')
    ])
    timeseria_problem_data_dict['idtime_nan_data'] = df.filter(pl.col('label_idtime_null') == 1)

    # Label idtime duplicate type
    df = df.with_columns([
        pl.concat_str([pl.col('turbine_id'), pl.col('time')]).alias('idtime')]).with_columns([
            pl.col('idtime').is_duplicated().alias('idtime_is_duplicated'), 
            pl.col('idtime').is_first_distinct().alias('idtime_is_first_distinct')
            ])

    df = df.with_columns([
        pl.when(~pl.col('idtime_is_duplicated'))
        .then(0)
        .when(pl.col('idtime_is_duplicated') & pl.col('idtime_is_first_distinct'))
        .then(1)
        .when(pl.col('idtime_is_duplicated') & ~pl.col('idtime_is_first_distinct'))
        .then(2)
        .alias('label_idtime_duplicated')
    ])

    timeseria_problem_data_dict['idtime_duplicated_data'] = df.filter(pl.col('label_idtime_duplicated') != 0)

    # todo: fix from here
    # Label time continuity type
    df = df.with_columns([
        pl.col('time').diff().over('turbine_id').alias('time_interval')
    ])

    df = df.with_columns([
        ((pl.col('time_interval') != pl.duration(seconds=time_interval)) & 
         (pl.col('time_interval') != pl.duration(seconds=0)))
        .cast(pl.Int32)
        .alias('label_idtime_continuous_type')
    ])

    # Create not continuity data
    df_notcontinuity_data = df.filter(pl.col('label_idtime_continuous_type') == 1).select([
        'turbine_id', 
        'time', 
        'time_interval',
        (pl.col('time') - pl.col('time_interval')).alias('start_time')
    ])
    
    timeseria_problem_data_dict['idtime_notcontinuity_data'] = df_notcontinuity_data

    print(timeseria_problem_data_dict)

    # Drop temporary columns
    df = df.drop(['time_interval', 'is_duplicate', 'is_duplicate_not_first'])

    return df

# %%
