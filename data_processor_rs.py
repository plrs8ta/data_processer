# %%
import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
import os

# Vars
# File Path
PATH_CSV_FILE = Path(r'D:\temp_data\SCADAdata_23\aggregated_data_60s(test_51turbines_ori).csv')
PATH_PARQUET_FILE = Path(r'D:\temp_data\SCADAdata_23\aggregated_data_60s(test_51turbines_ori)_test.parquet')
# 漫天岭项目对比数据路径
PATH_CSV_FILE_SCADA23 = Path(r'D:\temp_data\SCADAdata_23\aggregated_data_60s(test_51turbines_ori).csv')
PATH_PARQUET_FILE_SCADA23 = Path(r'D:\temp_data\SCADAdata_23\aggregated_data_60s(test_51turbines_ori).parquet')
PATH_CSV_FILE_SCADA24 = Path(r'D:\temp_data\SCADAdata_24\my2000_onedata_202501231426.csv')
PATH_PARQUET_FILE_SCADA24 = Path(r'D:\temp_data\SCADAdata_24\my2000_onedata_202501231426.parquet')

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
    
    # Label time continuity type
    df = df.with_columns([
        pl.col('time').diff().over('turbine_id').alias('time_interval')
    ])

    df = df.with_columns([
        ((pl.col('time_interval') != pl.duration(seconds=time_interval)) & 
         (pl.col('time_interval') != pl.duration(seconds=0)))
        .cast(pl.Boolean)
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
    df = df.drop(['time_interval', 'idtime_is_duplicated', 'idtime', 'idtime_is_first_distinct'])

    return df


def label_outliers_type(df: pl.DataFrame = None,
                       column_float_type: list = ['windspeed_avg', 'winddirection_avg'],
                       continuety_num: int = 3) -> pl.DataFrame:
    """
    Label outliers type

    Args:
        df (pl.DataFrame, optional): Data to label outliers type. Defaults to None.
        column_float_type (list, optional): Float type column list. Defaults to LIST_BASIC_COLUMN.
        continuety_num (int, optional): Value continuety occur times. Defaults to 3.

    Returns:
        pl.DataFrame: Data with labeled outliers type
    """

    if df is None:
        df = pl.read_parquet(PATH_PARQUET_FILE)

    outlier_problem_dict = {}
    
    # Get numeric columns excluding time and turbine_id
    num_col_list = [col for col in df.columns if col not in ['time', 'turbine_id']]
    
    # Label NaN types
    df = df.with_columns([
        pl.when(~pl.any_horizontal(pl.col(num_col_list).is_null()))
        .then(0)
        .when(pl.any_horizontal(pl.col(num_col_list).is_null()) & 
              ~pl.all_horizontal(pl.col(num_col_list).is_null()))
        .then(1)
        .when(pl.all_horizontal(pl.col(num_col_list).is_null()))
        .then(2)
        .otherwise(0)
        .alias('label_NaN_type')
    ])
    
    # Store NaN data in outlier problem dict
    outlier_problem_dict['NaN_data'] = df.filter(pl.col('label_NaN_type') != 0)
    
    # Label continuous duplicate values
    col_float_set = set(df.columns) & set(column_float_type)
    
    # Filter and sort data
    df_filtered = df.filter(
        (pl.col('label_idtime_null') == 0) & 
        (pl.col('label_idtime_duplicated') != 2)
    ).sort(['turbine_id', 'time'])
    
    # Todo: Check from here
    # Check for continuous duplicate values
    for col in col_float_set:
        df = df.with_columns([
            (pl.col(col).eq(pl.col(col).shift(-1)) & 
             pl.col(col).eq(pl.col(col).shift(-2)))
            .fill_null(False)
            .alias(f'continue_duplicate_type_{col}')
        ])
        
        # Update shifted values
        df = df.with_columns([
            pl.when(pl.col(f'continue_duplicate_type_{col}'))
            .then(pl.col(f'continue_duplicate_type_{col}').shift(1))
            .otherwise(pl.col(f'continue_duplicate_type_{col}'))
            .alias(f'continue_duplicate_type_{col}')
        ])
        
        df = df.with_columns([
            pl.when(pl.col(f'continue_duplicate_type_{col}'))
            .then(pl.col(f'continue_duplicate_type_{col}').shift(2))
            .otherwise(pl.col(f'continue_duplicate_type_{col}'))
            .alias(f'continue_duplicate_type_{col}')
        ])
    
    # Label out-of-range wind speed
    df = df.with_columns([
        ((pl.col('windspeed_avg') >= 30) | (pl.col('windspeed_avg') < 0))
        .alias('label_overrange_windspeed')
    ])
    
    print(outlier_problem_dict)

    return df


def label_situations_type(df: pl.DataFrame,
    rated_power:int=RATED_POWER,
    pitch_limit:int=4,
) -> pl.DataFrame:
    """
    打特殊工况标签。
    24.8.12 调频限电目前没有限电标志位，
    24.8.12 惯量调频。

    Args:
        df (pl.DataFrame): 打完异常数据标签后的数据。
        rated_wind_speed (int, optional): 额定风速。 Defaults to 10.
        pitch_limit (int, optional): 变桨限值。 Defaults to 4.

    Returns:
        pl.DataFrame: 打完工况后的标签。
    """
    
    # Create conditions for different limitation levels
    df = df.with_columns([
        pl.when((pl.col('power_avg') <= rated_power) & 
               (pl.col('pitchangle1_avg') >= pitch_limit))
        .then(1)
        .when((pl.col('power_avg') <= rated_power * 0.5) & 
              (pl.col('pitchangle1_avg') >= pitch_limit * 0.5))
        .then(2)
        .when((pl.col('power_avg') <= rated_power * 0.25) & 
              (pl.col('pitchangle1_avg') >= pitch_limit * 0.25))
        .then(3)
        .otherwise(0)
        .alias('label_condiction_limited')
    ])
    
    # Create dictionary of limited situations
    situation_dict = {
        'limited situation data': df.filter(pl.col('label_condiction_limited') != 0)
    }
    
    print("工况字典: \n", situation_dict)
    
    return df

def label_operation(df: pl.DataFrame) -> pl.DataFrame:
    """
    对风向数据进行分类标签化处理。

    Args:
        df (pl.DataFrame): 包含风向数据的 DataFrame。

    Returns:
        pl.DataFrame: 包含风向分类标签的 DataFrame。
    """
    # 创建风向划分的区间（每 22.5 度一个区间），从 0 到 360 度
    wdb = np.arange(0, 382.5, 22.5)
    wdb_label = [i for i in range(16)]
    
    # 使用 polars 的 cut 函数对风向数据进行分箱
    df = df.with_columns([
        pl.col('winddirection_avg')
        .cut(
            breaks=wdb.tolist(),
            labels=wdb_label,
            left_closed=False,
        )
        .alias('label_wdb')
    ])
    
    # 对偏航位置进行同样的处理
    npb = np.arange(0, 382.5, 22.5)
    npb_label = [i for i in range(16)]
    
    df = df.with_columns([
        pl.col('yawposition_avg')
        .cut(
            breaks=npb.tolist(),
            labels=npb_label,
            left_closed=False,
        )
        .alias('label_npb')
    ])

    return df

def integrity_validation(
    df: pl.DataFrame = None,
    validation_windspeedbinvalue: float = 0.5,
    validation_windspeedmax: float = 15.25,
    validation_windspeedmin: float = 2.25,
    validation_wholecumtime: float = 180.0 * 60.0 * 60.0,
    validation_bincumtime: float = 0.5 * 60.0 * 60.0,
) -> pl.DataFrame:
    """
    验证数据完整性，包括风速分箱和累计时间检查。

    Args:
        df (pl.DataFrame): 需要验证的数据集，包含 'windspeed_avg', 'time', 和 'turbine_id' 列。
        validation_windspeedbinvalue (float): 风速分箱的步长。 Defaults to 0.5.
        validation_windspeedmax (float): 风速分箱的最大值。 Defaults to 15.25.
        validation_windspeedmin (float): 风速分箱的最小值。 Defaults to 2.25.
        validation_wholecumtime (float): 累计时间的最小阈值（秒）。 Defaults to 180*60*60 (5 hours).
        validation_bincumtime (float): 每个风速分箱的最小累计时间（秒）。 Defaults to 0.5*60*60 (30 minutes).

    Returns:
        pl.DataFrame: 包含满足条件的 DataFrame。
    """

    # 初始化用于存储完整性问题的字典。
    integrity_problem_dict = {}
    
    # 创建风速分箱区间，并生成对应的标签。
    validation_windspeedbin = np.arange(
        validation_windspeedmin, 
        validation_windspeedmax + validation_windspeedbinvalue, 
        validation_windspeedbinvalue
    )
    validation_bins_label = [
        f'({validation_windspeedbin[i]}, {validation_windspeedbin[i+1]})'
        for i in range(len(validation_windspeedbin)-1)
    ]
    
    # 将 'windspeed_avg' 列的数据划分到指定的风速分箱中
    df = df.with_columns([
        pl.col('windspeed_avg')
        .cut(
            breaks=validation_windspeedbin.tolist(),
            labels=validation_bins_label,
            left_closed=False,
        )
        .alias('label_windspeedbin')
    ])

    # 计算时间间隔
    df = df.with_columns([
        pl.col('time').diff().over('turbine_id').alias('time_interval')
    ])

    # 计算每个机组的累计时间
    wholecumtime = df.group_by('turbine_id').agg([
        pl.col('time_interval').sum().alias('total_time')
    ])
    
    # 找出满足累计时间条件的风机ID
    wholecumtime_satisfy_idlist = wholecumtime.filter(
        pl.col('total_time') >= pl.duration(seconds=validation_wholecumtime)
    ).get_column('turbine_id')

    # 筛选出累计时间满足条件的风机数据
    df_wholecumtime_satisfy = df.filter(
        pl.col('turbine_id').is_in(wholecumtime_satisfy_idlist)
    )

    # 记录不满足累计时间条件的风机数据
    integrity_problem_dict['whole'] = df.filter(
        ~pl.col('turbine_id').is_in(wholecumtime_satisfy_idlist)
    )

    # 计算满足条件的风机数据的风速分箱累计时间
    bincumtime = df_wholecumtime_satisfy.group_by(['turbine_id', 'label_windspeedbin']).agg([
        pl.col('time_interval').sum().alias('bin_total_time')
    ])

    # 找出满足分箱累计时间条件的风机ID
    bincumtime_satisfy_idlist = bincumtime.filter(
        pl.col('bin_total_time') >= pl.duration(seconds=validation_bincumtime)
    ).get_column('turbine_id').unique()

    # 记录不满足风速分箱累计时间条件的风机数据
    integrity_problem_dict['bin'] = df.filter(
        ~pl.col('turbine_id').is_in(bincumtime_satisfy_idlist)
    )

    # 新增完整性验证标签列
    df = df.with_columns([
        pl.col('turbine_id')
        .is_in(wholecumtime_satisfy_idlist)
        .cast(pl.Int8)
        .alias('label_integirity_validation')
    ])

    # 删除临时列
    df = df.drop(['label_windspeedbin', 'time_interval'])

    print(integrity_problem_dict)

    return df

# def filter_data(df: pl.DataFrame = None,
#                 filter_time_range: list = [20231013, 20231216],
#                 average_time_interval: int = 600,
#                 filter_turbine_operation_mode: int = 20, 
#                 filter_pitch_angle: int = 3, 
#                 ) -> pl.DataFrame:
#     """
#     Filter data based on specified conditions
    
#     Args:
#         df (pl.DataFrame, optional): Input DataFrame. Defaults to None.
#         filter_time_range (list): Start and end dates in YYYYMMDD format
#         average_time_interval (int): Time interval in seconds for data averaging
#         filter_turbine_operation_mode (int): Operating mode to filter
#         filter_pitch_angle (int): Maximum pitch angle for filtering
        
#     Returns:
#         pl.DataFrame: Filtered DataFrame
#     """
#     if df is None:
#         df = pl.read_parquet(PATH_PARQUET_FILE)
    
#     # Convert time range to datetime
#     start_time = pl.datetime(
#         year=int(str(filter_time_range[0])[:4]),
#         month=int(str(filter_time_range[0])[4:6]),
#         day=int(str(filter_time_range[0])[6:8])
#     end_time = pl.datetime(
#         year=int(str(filter_time_range[1])[:4]),
#         month=int(str(filter_time_range[1])[4:6]),
#         day=int(str(filter_time_range[1])[6:8])
    
#     # Apply filters
#     filtered_df = df.filter(
#         (pl.col('time') >= start_time) &
#         (pl.col('time') <= end_time) &
#         # (pl.col('operatingmode_cntmax') == filter_turbine_operation_mode) &
#         (pl.col('pitchangle1_avg').abs() < filter_pitch_angle)
#     )
    
#     # Group by time window and turbine_id, then calculate averages
#     if average_time_interval > 0:
#         filtered_df = filtered_df.group_by([
#             pl.col('time').dt.truncate(f'{average_time_interval}s'),
#             'turbine_id'
#         ]).agg([
#             pl.col('windspeed_avg').mean(),
#             pl.col('winddirection_avg').mean(),
#             pl.col('winddirection_relative_avg').mean(),
#             pl.col('yawposition_avg').mean(),
#             pl.col('pitchangle1_avg').mean(),
#             pl.col('generatorspeed_avg').mean(),
#             pl.col('power_avg').mean(),
#             pl.col('airdensity_avg').mean(),
#             # pl.col('operatingmode_cntmax').mode(),
#             # pl.col('yawmode_cntmax').mode(),
#             # pl.col('brakemode_cntmax').mode()
#         ])
    
#     return filtered_df
    
def windspeed_normalized_byairdensity(df = None,
                                    #   R0: float = 287.05,
                                    #   Rw: float = 461.5,
                                    #   B_10min: float = 101325,
                                    #   phi: float = 0.5,
                                      ):
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
    
    # # 标准空气密度（海平面标准大气）
    rho0 = 1.225

    # 计算归一化风速，使用公式：Vn = Va * (ρ/ρ0)^(1/3)
    df = df.with_columns(
        (pl.col('windspeed_avg') * ((pl.col('airdensity_avg')/rho0) ** (1/3))).alias('normalized_windspeed_byairdensity')
    )

    # df['normalized_windspeed_byairdensity'] = df['windspeed_avg'] * ((df['airdensity_avg']/rho0) ** (1/3))
    
    return df

# def compute_wind_speed_distribute(df: pl.DataFrame = None,
#                                 wind_speed_interval: float = 0.5) -> pl.DataFrame:
#     """
#     Compute wind speed probability distribution with interval labeling

#     Args:
#         df (pl.DataFrame, optional): Input DataFrame. Defaults to None.
#         wind_speed_interval (float, optional): Wind speed interval size. Defaults to 0.5.

#     Returns:
#         pl.DataFrame: DataFrame with wind speed distribution analysis
#     """
#     if df is None:
#         df = pl.read_parquet(PATH_PARQUET_FILE)
    
#     # Create wind speed interval labels
#     df = df.with_columns([
#         ((pl.col('windspeed_avg').floor() / wind_speed_interval).floor() * wind_speed_interval)
#         .alias('wind_speed_bin_start'),
#     ])
    
#     df = df.with_columns([
#         (pl.col('wind_speed_bin_start') + wind_speed_interval).alias('wind_speed_bin_end')
#     ])
    
#     # Group by wind speed bins and calculate wind_speed_frequency
#     distribution_df = df.group_by(['wind_speed_bin_start', 'wind_speed_bin_end']).agg([
#         pl.len().alias('wind_speed_frequency')
#     ])
    
#     # Calculate probability density
#     total_count = distribution_df.select(pl.col('wind_speed_frequency').sum())[0, 0]
#     distribution_df = distribution_df.with_columns([
#         (pl.col('wind_speed_frequency') / total_count).alias('wind_speed_probability')
#     ])
    
    # # Calculate cumulative probability density
    # distribution_df = distribution_df.sort('wind_speed_bin_start').with_columns([
    #     pl.col('wind_speed_probability').cum_sum().alias('cumulative_probability')
    # ])
    
    # return distribution_df

def compute_power_curve(
    df: pl.DataFrame,
    normalized_windspeedbinvalue: float = 0.5,
    normalized_windspeedmax: float = 15.25,
    normalized_windspeedmin: float = 2.25,
) -> pl.DataFrame:
    """
    计算功率曲线值，包括风速分箱和功率统计量。

    Args:
        df (pl.DataFrame): 包含归一化风速 ('normalized_windspeed_byairdensity') 和功率 ('power_avg') 数据的 DataFrame。
        normalized_windspeedbinvalue (float): 归一化风速分箱的步长。 Defaults to 0.5.
        normalized_windspeedmax (float): 归一化风速分箱的最大值。 Defaults to 15.25.
        normalized_windspeedmin (float): 归一化风速分箱的最小值。 Defaults to 2.25.

    Returns:
        pl.DataFrame: 包含每个风速分箱的风速均值、功率均值、功率均方根偏差和计数的 DataFrame。
    """
    # 创建分箱边界
    bins = np.arange(
        normalized_windspeedmin,
        normalized_windspeedmax + normalized_windspeedbinvalue,
        normalized_windspeedbinvalue
    )
    
    # 创建完整的条件表达式
    expr = (pl.when(pl.col("normalized_windspeed_byairdensity") <= normalized_windspeedmin)
            .then(pl.lit(f"(-inf, {normalized_windspeedmin})")))
    
    # 添加中间区间的条件
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        label = f"({bin_start}, {bin_end})"
        expr = (expr.when((pl.col("normalized_windspeed_byairdensity") > bin_start) & 
                         (pl.col("normalized_windspeed_byairdensity") <= bin_end))
                   .then(pl.lit(label)))
    
    # 添加最后的条件
    expr = (expr.when(pl.col("normalized_windspeed_byairdensity") > normalized_windspeedmax)
               .then(pl.lit(f"({normalized_windspeedmax}, inf)"))
               .otherwise(None))
    
    # 应用分箱标签
    df = df.with_columns(
        expr.alias("label_normalized_windspeedbin_byairdensity")
    )
    
    # 按风机ID和风速分箱进行分组计算
    result = df.group_by(["turbine_id", "label_normalized_windspeedbin_byairdensity"]).agg([
        pl.col("normalized_windspeed_byairdensity").mean().alias("wind_mean"),
        pl.col("power_avg").mean().alias("power_mean"),
        (((pl.col("power_avg") - pl.col("power_avg").mean())**2).mean().sqrt())
        .alias("power_rmsd"),
        pl.len().alias("cnt")
    ])
    
    return df, result

def compoute_AEP(df: pl.DataFrame, df_windspeedbin_agg: pl.DataFrame):
    """
    计算等效发电性能（AEP）。
    """
    # 计算概率
    counts = df.group_by(['turbine_id', 'label_normalized_windspeedbin_byairdensity']).agg(
        pl.len().alias('bin_count')
    )
    total_counts = df.group_by('turbine_id').agg(
        pl.len().alias('total_count')
    )
    
    # 计算概率并与统计信息合并
    probabilities = counts.join(
        total_counts, 
        on='turbine_id'
    ).with_columns(
        (pl.col('bin_count') / pl.col('total_count')).alias('probability')
    )
    
    # 合并数据并计算AEP
    result = df_windspeedbin_agg.join(
        probabilities.select(['turbine_id', 'label_normalized_windspeedbin_byairdensity', 'probability']),
        on=['turbine_id', 'label_normalized_windspeedbin_byairdensity']
    ).with_columns(
        (pl.col('power_mean') * pl.col('probability')).alias('pp')
    ).group_by('turbine_id').agg(
        pl.col('pp').sum().alias('AEP')
    )
    
    return result

def compute_Cp(
    df_windspeedbin_agg: pl.DataFrame, 
    rho0: float = 1.225, 
    rotor_D: float = 82,
) -> pl.DataFrame:
    """
    计算功率系数Cp。
    """
    # 计算功率系数
    swept_area = np.pi * (rotor_D/2) ** 2
    
    result = df_windspeedbin_agg.with_columns(
        ((pl.col('power_mean') * 1000) / 
         (0.5 * rho0 * swept_area * pl.col('wind_mean').pow(3))).alias('Cp')
    )
    
    return result

def compute_wind_probability(df: pl.DataFrame) -> pl.DataFrame:
    """
    计算风频分布。

    Args:
        df (pl.DataFrame): 输入数据框，需包含 turbine_id 和 label_normalized_windspeedbin_byairdensity 列

    Returns:
        pl.DataFrame: 包含每个风速区间的概率分布
    """
    result = (df.group_by(['turbine_id', 'label_normalized_windspeedbin_byairdensity'])
            .agg(pl.len().alias('count'))  # 添加别名 'cnt'
            .with_columns([
                (pl.col('count') / pl.col('count').sum().over('turbine_id')).alias('probability')  # 按 turbine_id 分组计算概率
            ])
            .sort(['turbine_id', 'label_normalized_windspeedbin_byairdensity']))
    
    return result

def compute_weight_power(
    df_power_curve: pl.DataFrame, 
    df_wind_probability: pl.DataFrame,
) -> pl.Series:
    """
    计算加权功率。
    """
    # 假设df_power_curve中有turbine_id和power_mean列
    # 将power_curve数据与风频分布合并
    result = (df_power_curve
             .join(df_wind_probability, on=['turbine_id', 'label_normalized_windspeedbin_byairdensity'])
             .with_columns((pl.col('power_mean') * pl.col('probability')).alias('weighted_power')))
    
    return result


# %% Main processing
if __name__ == '__main__':
    
    df = pl.read_parquet(PATH_PARQUET_FILE_SCADA24)
    df = filter_data(df, filter_time_range=[20241013, 20241216])
    df = windspeed_normalized_byairdensity(df)
    df, df_pc = compute_power_curve(df)
    df_AEP = compoute_AEP(df, df_pc)
    # df_Cp = compute_Cp(df_pc)
    df_wind_probability = compute_wind_probability(df)
    df_wp = compute_weight_power(df_pc, df_wind_probability)


# %% 
df_SCADA23 = pl.read_csv(r"D:\temp_data\SCADAdata_23\aggregated_data_60s(test_51turbines_ori)_result.csv")
df_SCADA24 = pl.read_csv(r"D:\temp_data\SCADAdata_24\my2000_onedata_202501231426_result.csv")
df_merge = df_SCADA23.join(
    df_SCADA24, 
    on=['turbine_id', 'label_normalized_windspeedbin_byairdensity'], 
    # right_on=['turbine_id_SCADA24', 'label_normalized_windspeedbin_byairdensity_SCADA24'], 
    how='left'
    )
