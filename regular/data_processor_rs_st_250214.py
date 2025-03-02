# %%
import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
from openoa.utils import filters, power_curve, plot

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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

# Main data processor fucntions

def filter_data(df: pl.DataFrame = None,
                filter_time_range: list = None,  # Changed default to None
                average_time_interval: int = 600,
                filter_turbine_operation_mode: int = 20, 
                filter_pitch_angle: int = 3, 
                filter_turbine_speed: int = 1000
                ) -> pl.DataFrame:
    """
    Filter data based on specified conditions
    
    Args:
        df (pl.DataFrame, optional): Input DataFrame. Defaults to None.
        filter_time_range (list, optional): Start and end dates in YYYYMMDD format. If None, all dates are included.
        average_time_interval (int): Time interval in seconds for data averaging
        filter_turbine_operation_mode (int): Operating mode to filter
        filter_pitch_angle (int): Maximum pitch angle for filtering
        filter_turbine_speed (float): Minimum generator speed for filtering
        
    Returns:
        pl.DataFrame: Filtered DataFrame
    """
    if df is None:
        df = pl.read_parquet(PATH_PARQUET_FILE)
    
    # Start with base filters
    filter_conditions = [
        (pl.col('pitchangle1_avg').abs() < filter_pitch_angle),
        (pl.col('generatorspeed_avg').abs() >= filter_turbine_speed)
    ]
    
    # Add time range filter if specified
    if filter_time_range is not None:
        start_time = pl.datetime(
            year=int(str(filter_time_range[0])[:4]),
            month=int(str(filter_time_range[0])[4:6]),
            day=int(str(filter_time_range[0])[6:8])
        )
        end_time = pl.datetime(
            year=int(str(filter_time_range[1])[:4]),
            month=int(str(filter_time_range[1])[4:6]),
            day=int(str(filter_time_range[1])[6:8])
        )
        filter_conditions.extend([
            (pl.col('time') >= start_time),
            (pl.col('time') <= end_time)
        ])
    
    # Apply filters
    filtered_df = df.filter(
        (pl.col('time') >= start_time) &
        (pl.col('time') <= end_time) &
        # (pl.col('operatingmode_cntmax') == filter_turbine_operation_mode) &
        (pl.col('pitchangle1_avg').abs() < filter_pitch_angle) &
        (pl.col('generatorspeed_avg').abs() >= filter_turbine_speed)
    )
    
    # Group by time window and turbine_id, then calculate averages
    if average_time_interval > 0:
        filtered_df = filtered_df.group_by([
            pl.col('time').dt.truncate(f'{average_time_interval}s'),
            'turbine_id'
        ]).agg([
            pl.col('windspeed_avg').mean(),
            pl.col('winddirection_avg').mean(),
            pl.col('winddirection_relative_avg').mean(),
            pl.col('yawposition_avg').mean(),
            pl.col('pitchangle1_avg').mean(),
            pl.col('generatorspeed_avg').mean(),
            pl.col('power_avg').mean(),
            pl.col('airdensity_avg').mean(),
            # pl.col('operatingmode_cntmax').mode(),
            # pl.col('yawmode_cntmax').mode(),
            # pl.col('brakemode_cntmax').mode()
        ])
    
    return filtered_df
    
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
    rho0 = 1.225

    # 计算归一化风速，使用公式：Vn = Va * (ρ/ρ0)^(1/3)
    df = df.with_columns(
        (pl.col('windspeed_avg') * ((pl.col('airdensity_avg')/rho0) ** (1/3))).alias('normalized_windspeed_byairdensity')
    )
    
    return df

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
    
    # First filter out data below minimum wind speed
    df = df.filter(pl.col("normalized_windspeed_byairdensity") >= normalized_windspeedmin)
    
    # 创建分箱边界
    bins = np.arange(
        normalized_windspeedmin,
        normalized_windspeedmax + normalized_windspeedbinvalue,
        normalized_windspeedbinvalue
    )
    
    # 创建条件表达式（从最小风速开始）
    expr = None
    
    # 添加中间区间的条件
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        label = f"({bin_start:.2f}, {bin_end:.2f}]"  # Changed to right-closed interval notation
        new_condition = ((pl.col("normalized_windspeed_byairdensity") > bin_start) & 
                        (pl.col("normalized_windspeed_byairdensity") <= bin_end))  # Already using right-closed intervals
        
        if expr is None:
            expr = pl.when(new_condition).then(pl.lit(label))
        else:
            expr = expr.when(new_condition).then(pl.lit(label))
    
    # 添加最后的条件
    expr = (expr.when(pl.col("normalized_windspeed_byairdensity") > normalized_windspeedmax)
              .then(pl.lit(f"({normalized_windspeedmax:.2f}, inf]"))  # Changed to right-closed notation
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

def compute_wind_probability_farm(df: pl.DataFrame) -> pl.DataFrame:
    """
    计算整个风电场的风频分布，但保持每个风机的输出结构。
    
    首先计算整个风电场的风频分布（所有风机数据合并计算），
    然后复制到每个风机以保持与compute_wind_probability()相同的输出结构。

    Args:
        df (pl.DataFrame): 输入数据框，需包含 turbine_id 和 label_normalized_windspeedbin_byairdensity 列

    Returns:
        pl.DataFrame: 包含每个风速区间的概率分布，但每个风机的概率值相同
    """
    # 首先计算整个风电场的风频分布
    farm_probability = (df.group_by('label_normalized_windspeedbin_byairdensity')
                       .agg(pl.len().alias('farm_count'))
                       .with_columns(
                           (pl.col('farm_count') / pl.col('farm_count').sum()).alias('farm_probability', 
                           pl.col('farm_count'))
                       ))
    
    # 获取所有唯一的turbine_id
    unique_turbines = df.select('turbine_id').unique()
    
    # 为每个风机复制场级概率
    result = (unique_turbines.join(
        farm_probability,
        how='cross'  # 笛卡尔积，确保每个风机都有所有风速区间的概率
    ).with_columns([
        pl.col('farm_probability').alias('probability'),  # 重命名以匹配原函数输出
    ]).drop('farm_count', 'farm_probability')
    .sort(['turbine_id', 'label_normalized_windspeedbin_byairdensity']))
    
    return result



# def compute_Cp(
#     df_windspeedbin_agg: pl.DataFrame, 
#     rho0: float = 1.225, 
#     rotor_D: float = 82,
# ) -> pl.DataFrame:
#     """
#     计算功率系数Cp。
#     """
#     # 计算功率系数
#     swept_area = np.pi * (rotor_D/2) ** 2
    
#     result = df_windspeedbin_agg.with_columns(
#         ((pl.col('power_mean') * 1000) / 
#          (0.5 * rho0 * swept_area * pl.col('wind_mean').pow(3))).alias('Cp')
#     )
    
#     return result



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

# def compoute_equal_AEP(df: pl.DataFrame, df_windspeedbin_agg: pl.DataFrame):
#     """
#     计算等效发电性能（equal_AEP）。
#     """
#     # 计算概率
#     counts = df.group_by(['turbine_id', 'label_normalized_windspeedbin_byairdensity']).agg(
#         pl.len().alias('bin_count')
#     )
#     total_counts = df.group_by('turbine_id').agg(
#         pl.len().alias('total_count')
#     )
    
#     # 计算概率并与统计信息合并
#     probabilities = counts.join(
#         total_counts, 
#         on='turbine_id'
#     ).with_columns(
#         (pl.col('bin_count') / pl.col('total_count')).alias('probability')
#     )
    
#     # 合并数据并计算equal_AEP
#     result = df_windspeedbin_agg.join(
#         probabilities.select(['turbine_id', 'label_normalized_windspeedbin_byairdensity', 'probability']),
#         on=['turbine_id', 'label_normalized_windspeedbin_byairdensity']
#     ).with_columns(
#         (pl.col('power_mean') * pl.col('probability')).alias('pp')
#     ).group_by('turbine_id').agg(
#         pl.col('pp').sum().alias('equal_AEP')
#     )
    
#     return result

def process_scada_data(df: pl.DataFrame,
                      filter_dates: list,
                      avg_time_interval: int,
                      turbine_op_mode: int,
                      pitch_angle: float,
                      turbine_speed:float, 
                      wind_speed_bin: float,
                      max_wind_speed: float,
                      min_wind_speed: float,
                      wind_probality=None, 
                      use_air_density_norm: bool = True) -> pl.DataFrame:
    """
    Process SCADA data with given parameters and return processed results
    
    Args:
        df: Input DataFrame
        filter_dates: List of [start_date, end_date] in YYYYMMDD format
        avg_time_interval: Time interval for averaging in seconds
        turbine_op_mode: Turbine operation mode to filter
        pitch_angle: Maximum pitch angle for filtering
        wind_speed_bin: Wind speed bin size
        max_wind_speed: Maximum wind speed for binning
        min_wind_speed: Minimum wind speed for binning
        wind_probality: Optional pre-calculated wind probability distribution
        use_air_density_norm: Whether to normalize wind speed by air density
    
    Returns:
        Tuple[pl.DataFrame, pl.DataFrame]: (weighted power results, wind probability distribution)
    """
    # Apply filters and processing
    df = filter_data(
        df, 
        filter_time_range=filter_dates,
        average_time_interval=avg_time_interval,
        filter_turbine_operation_mode=turbine_op_mode,
        filter_pitch_angle=pitch_angle, 
        filter_turbine_speed=turbine_speed
    )
    
    if use_air_density_norm:
        df = windspeed_normalized_byairdensity(df)
    else:
        # If not using air density normalization, just copy windspeed_avg
        df = df.with_columns(
            pl.col('windspeed_avg').alias('normalized_windspeed_byairdensity')
        )

    df, df_pc = compute_power_curve(
        df,
        normalized_windspeedbinvalue=wind_speed_bin,
        normalized_windspeedmax=max_wind_speed,
        normalized_windspeedmin=min_wind_speed
    )

    # Calculate or use provided wind probability
    df_wind_probability = wind_probality if wind_probality is not None else compute_wind_probability_farm(df)
    
    # Calculate weighted power
    df_wp = compute_weight_power(df_pc, df_wind_probability)

    return df_wp, df_wind_probability, df

def compare_data():
    df_24 = pl.read_parquet(PATH_PARQUET_FILE_SCADA24)
    df_wp_24, wind_probality_SCADA24= process_scada_data(
                    df_24,
                    filter_dates=[20241013, 20241216],
                    avg_time_interval=600,
                    turbine_op_mode=20,
                    pitch_angle=3,
                    turbine_speed=1000, 
                    wind_speed_bin=0.5,
                    max_wind_speed=15,
                    min_wind_speed=2.5,
                    wind_probality=None, 
                    use_air_density_norm=False
                )
    df_24 = None

    df_23 = pl.read_parquet(PATH_PARQUET_FILE_SCADA23)
    df_wp_23, wind_probality_SCADA23= process_scada_data(
                        df_23,
                        filter_dates=[20231013, 20231216],
                        avg_time_interval=600,
                        turbine_op_mode=20,
                        pitch_angle=10,
                        turbine_speed=0, 
                        wind_speed_bin=0.5,
                        max_wind_speed=15,
                        min_wind_speed=2.5,
                        wind_probality=wind_probality_SCADA24, 
                        use_air_density_norm=False
                    )
    df_23 = None
    
    df_merge = df_wp_23.join(
        df_wp_24,
        on=['turbine_id', 'label_normalized_windspeedbin_byairdensity'],
        how='inner',  # Changed from 'outer' to 'inner' to only keep matching records
        suffix='_24'  # This will add '_24' to all right-side columns
    ).sort(['turbine_id', 'label_normalized_windspeedbin_byairdensity'])

    return df_merge

def draw_power_curve():
    # Get sorted turbine IDs
    sorted_turbine_ids = pl.scan_parquet(PATH_PARQUET_FILE_SCADA24)\
                            .select("turbine_id")\
                            .unique()\
                            .sort("turbine_id")\
                            .collect()\
                            .get_column("turbine_id")\
                            .to_list()
    # sorted_turbine_ids = sorted_turbine_ids.collect()

    for turbine_id in sorted_turbine_ids:
        # Load 2024 data
        df_24 = (
            pl.scan_parquet(PATH_PARQUET_FILE_SCADA24)
            .filter(pl.col("turbine_id") == turbine_id)
            .collect()
        )
        df_24 = filter_data(df_24, 
                         filter_time_range=[20241013, 20241216], 
                         average_time_interval=600, 
                         filter_pitch_angle=3, 
                         filter_turbine_speed=1000
                         )
        df_24 = df_24.filter(pl.col('windspeed_avg') <= 12.5)
        # Load 2023 data
        df_23 = (
            pl.scan_parquet(PATH_PARQUET_FILE_SCADA23)
            .filter(pl.col("turbine_id") == turbine_id)
            .collect()
        )
        df_23 = filter_data(df_23,  # Fixed: Changed df_24 to df_23
                         filter_time_range=[20231013, 20231216], 
                         average_time_interval=600, 
                         filter_pitch_angle=3, 
                         filter_turbine_speed=1000
                         )
        df_23 = df_23.filter(pl.col('windspeed_avg') <= 12.5)
        
        # Convert to pandas for plotting
        df_23 = df_23.to_pandas()
        df_24 = df_24.to_pandas()

        # Calculate spline curves
        spline_curve_23 = power_curve.gam(df_23['windspeed_avg'], df_23['power_avg'], n_splines=20)
        spline_curve_24 = power_curve.gam(df_24['windspeed_avg'], df_24['power_avg'], n_splines=20)

        # Create figure with more space for title
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot scatter points
        ax.scatter(df_23['windspeed_avg'], 
                  df_23['power_avg'],
                  alpha=0.4, 
                  s=10, 
                  color='C0',
                  label='Scatter (Before Optimization)')
        
        ax.scatter(df_24['windspeed_avg'], 
                  df_24['power_avg'], 
                  alpha=0.4, 
                  s=10, 
                  color='C2',
                  label='Scatter (After Optimization)')

        # Plot spline curves
        x = np.linspace(0, 20, 100)
        ax.plot(x, spline_curve_23(x), color="C0", label="Power Curve (Before Optimization)", linewidth=6)
        ax.plot(x, spline_curve_24(x), color="C2", label="Power Curve (After Optimization)", linewidth=6)

        # Customize plot
        ax.set_xlim(-1, 20)
        ax.set_ylim(-100, 2100)
        ax.set_xlabel('Wind Speed (m/s)')
        ax.set_ylabel('Power (kW)')
        ax.legend(loc='upper left')
        
        # Add title with turbine ID
        plt.suptitle(f'Turbine {turbine_id} Power Curve Comparison (Before vs After Optimization)', 
                    fontsize=16, 
                    y=0.95)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        
        # Optional: add a small delay between plots
        plt.pause(0.1)



# # %% Main processing
# if __name__ == '__main__':
    
#     df = pl.read_parquet(PATH_PARQUET_FILE_SCADA24)
#     df = filter_data(df, filter_time_range=[20241013, 20241216])
#     df = windspeed_normalized_byairdensity(df)
#     df, df_pc = compute_power_curve(df)
#     df_AEP = compoute_equal_AEP(df, df_pc)
#     # df_Cp = compute_Cp(df_pc)
#     df_wind_probability = compute_wind_probability(df)
#     df_wp = compute_weight_power(df_pc, df_wind_probability)


# # %% 
# df_SCADA23 = pl.read_csv(r"D:\temp_data\SCADAdata_23\aggregated_data_60s(test_51turbines_ori)_result.csv")
# df_SCADA24 = pl.read_csv(r"D:\temp_data\SCADAdata_24\my2000_onedata_202501231426_result.csv")
# df_merge = df_SCADA23.join(
#     df_SCADA24, 
#     on=['turbine_id', 'label_normalized_windspeedbin_byairdensity'], 
#     # right_on=['turbine_id_SCADA24', 'label_normalized_windspeedbin_byairdensity_SCADA24'], 
#     how='left'
#     )

# # Add Streamlit UI code at the bottom
# def create_streamlit_ui():
#     st.title("Wind Turbine Data Analysis")
    
#     # Sidebar for parameters
#     st.sidebar.header("Analysis Parameters")
    
#     # Time range selection for 2023
#     st.sidebar.subheader("2023 Time Range")
#     start_date_23 = st.sidebar.date_input("2023 Start Date", value=pd.to_datetime("20231013", format="%Y%m%d"))
#     end_date_23 = st.sidebar.date_input("2023 End Date", value=pd.to_datetime("20231216", format="%Y%m%d"))
    
#     # Time range selection for 2024
#     st.sidebar.subheader("2024 Time Range")
#     start_date_24 = st.sidebar.date_input("2024 Start Date", value=pd.to_datetime("20241013", format="%Y%m%d"))
#     end_date_24 = st.sidebar.date_input("2024 End Date", value=pd.to_datetime("20241216", format="%Y%m%d"))
    
#     # Other parameters
#     avg_time_interval = st.sidebar.number_input("Average Time Interval (seconds)", value=600, min_value=0)
#     turbine_op_mode = st.sidebar.number_input("Turbine Operation Mode", value=20)
#     pitch_angle = st.sidebar.number_input("Maximum Pitch Angle", value=3)
    
#     # Wind speed normalization parameters
#     st.sidebar.subheader("Wind Speed Analysis")
#     wind_speed_bin = st.sidebar.number_input("Wind Speed Bin Size", value=0.5, min_value=0.1)
#     max_wind_speed = st.sidebar.number_input("Maximum Wind Speed", value=15.25)
#     min_wind_speed = st.sidebar.number_input("Minimum Wind Speed", value=2.25)
    
#     # Add checkbox for air density normalization in sidebar
#     st.sidebar.subheader("Data Processing Options")
#     use_air_density_norm = st.sidebar.checkbox("Normalize Wind Speed by Air Density", value=True)
    
#     # Process data button
#     if st.sidebar.button("Process Data"):
#         with st.spinner("Processing data..."):
#             try:
#                 # Process SCADA data for 2023


#                 # Process SCADA data for 2024
#                 df_24 = pl.read_parquet(PATH_PARQUET_FILE_SCADA24)
#                 df_wp_24, wind_probality_SCADA24= process_scada_data(
#                     df_24,
#                     filter_dates=[int(start_date_24.strftime("%Y%m%d")), int(end_date_24.strftime("%Y%m%d"))],
#                     avg_time_interval=avg_time_interval,
#                     turbine_op_mode=turbine_op_mode,
#                     pitch_angle=pitch_angle,
#                     wind_speed_bin=wind_speed_bin,
#                     max_wind_speed=max_wind_speed,
#                     min_wind_speed=min_wind_speed,
#                     wind_probality=None, 
#                     use_air_density_norm=use_air_density_norm
#                 )
#                 df_24 = None

#                 df_23 = pl.read_parquet(PATH_PARQUET_FILE_SCADA23)
#                 df_wp_23, wind_probality_SCADA_23= process_scada_data(
#                     df_23,
#                     filter_dates=[int(start_date_23.strftime("%Y%m%d")), int(end_date_23.strftime("%Y%m%d"))],
#                     avg_time_interval=avg_time_interval,
#                     turbine_op_mode=turbine_op_mode,
#                     pitch_angle=pitch_angle,
#                     wind_speed_bin=wind_speed_bin,
#                     max_wind_speed=max_wind_speed,
#                     min_wind_speed=min_wind_speed,
#                     wind_probality=wind_probality_SCADA24, 
#                     use_air_density_norm=use_air_density_norm
#                 )
#                 df_23 = None

#                 # Merge weighted power results with renamed columns
#                 df_merge = df_wp_23.join(
#                     df_wp_24,
#                     on=['turbine_id', 'label_normalized_windspeedbin_byairdensity'],
#                     how='inner',  # Changed from 'outer' to 'inner' to only keep matching records
#                     suffix='_24'  # This will add '_24' to all right-side columns
#                 ).sort(['turbine_id', 'label_normalized_windspeedbin_byairdensity'])

#                 # Display results
#                 st.header("Analysis Results")
                
#                 # Show merged comparison data
#                 st.subheader("SCADA 2023 vs 2024 Power Curve Comparison")
#                 st.dataframe(df_merge)
                
#                 # Create visualizations
#                 st.subheader("Power Curve Analysis")
#                 fig_power = px.scatter(
#                     df_merge.to_pandas(), 
#                     x="wind_mean", 
#                     y="power_mean",
#                     color="turbine_id",
#                     title="Power Curve by Turbine (2023)"
#                 )
#                 st.plotly_chart(fig_power)
                
#                 # Add 2024 power curve
#                 fig_power_24 = px.scatter(
#                     df_merge.to_pandas(), 
#                     x="wind_mean_24", 
#                     y="power_mean_24",
#                     color="turbine_id",
#                     title="Power Curve by Turbine (2024)"
#                 )
#                 st.plotly_chart(fig_power_24)
                
#                 # Wind probability distribution
#                 st.subheader("Wind Speed Probability Distribution")
#                 fig_wind = px.bar(
#                     df_merge.to_pandas(),
#                     x="label_normalized_windspeedbin_byairdensity",
#                     y="probability",
#                     color="turbine_id",
#                     title="Wind Speed Distribution (2023)"
#                 )
#                 st.plotly_chart(fig_wind)
                
#                 fig_wind_24 = px.bar(
#                     df_merge.to_pandas(),
#                     x="label_normalized_windspeedbin_byairdensity",
#                     y="probability_24",
#                     color="turbine_id",
#                     title="Wind Speed Distribution (2024)"
#                 )
#                 st.plotly_chart(fig_wind_24)
                
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")

# if __name__ == '__main__':
#     create_streamlit_ui()


# df = pl.read_parquet(PATH_PARQUET_FILE_SCADA24)
# df = df.with_columns(
#     pl.col('windspeed_avg').alias('normalized_windspeed_byairdensity')
# )
# df, df_pc = compute_power_curve(
#     df,
#     normalized_windspeedbinvalue=0.5,
#     normalized_windspeedmax=100,
#     normalized_windspeedmin=0
# )

# %%
