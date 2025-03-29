# %%
import pandas as pd
import polars as pl 
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# %%
PATH_DEFAULT_DATA = r"C:\Users\EDY\Desktop\手动保存50_20250329150932(1).csv"
TIME_COLUMN = "TimeStamp"

def load_data(path: str = PATH_DEFAULT_DATA):
    df = pd.read_csv(path)
    # df = pl.from_pandas(df)
    return df

def plot_time_series(df: pl.DataFrame, column: str, 
                     time_column: str = TIME_COLUMN):
    fig = px.line(df, x=time_column, y=column)
    st.plotly_chart(fig)

def plot_normalized_time_series(df, selected_columns, time_column=TIME_COLUMN):
    """绘制归一化的时序图，每个选定的列都有自己的Y轴，轴颜色与数据线颜色一致"""
    if not selected_columns:
        st.warning("请至少选择一列进行可视化")
        return
    
    # 创建一个包含时间列和所有选定列的子数据集
    plot_df = df[[time_column] + selected_columns].copy()
    
    # 处理时间列，转换为相对于起始时间的秒数
    try:
        # 尝试将时间列转换为datetime格式
        if not pd.api.types.is_datetime64_any_dtype(plot_df[time_column]):
            try:
                # 尝试将字符串转换为日期时间格式
                plot_df[time_column] = pd.to_datetime(plot_df[time_column])
            except:
                st.warning("无法将时间列转换为日期时间格式，将尝试使用数值方式处理")
        
        # 如果时间列是日期时间格式，转换为相对秒数
        if pd.api.types.is_datetime64_any_dtype(plot_df[time_column]):
            # 计算相对于第一个时间点的秒数差
            start_time = plot_df[time_column].min()
            plot_df['时间(s)'] = (plot_df[time_column] - start_time).dt.total_seconds()
        else:
            # 如果不是日期时间格式，尝试直接计算相对值
            try:
                start_value = plot_df[time_column].astype(float).min()
                plot_df['时间(s)'] = plot_df[time_column].astype(float) - start_value
            except:
                st.error(f"无法处理时间列: {time_column}，请确保它是有效的日期时间或数值格式")
                return
    except Exception as e:
        st.error(f"时间处理出错: {str(e)}")
        return
    
    # 使用新的时间列创建图表
    import plotly.graph_objects as go
    
    # 创建一个空的Figure对象时指定模板
    fig = go.Figure(layout=dict(template='plotly_white'))
    
    # 使用固定的颜色列表，确保一致性
    colors = px.colors.qualitative.Plotly
    
    # 为每个选定的列添加一条线
    for i, col in enumerate(selected_columns):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=plot_df['时间(s)'],
                y=plot_df[col],
                mode='lines',
                name=col,
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{col}</b><br>时间: %{{x:.2f}}s<br>值: %{{y:.2f}}<br><extra></extra>"
            )
        )
    
    # 改进轴位置的生成逻辑
    num_axes = len(selected_columns)

    # 计算左右两侧需要多少轴
    left_axes = (num_axes + 1) // 2  # 左侧放置大约一半的轴（向上取整）
    right_axes = num_axes - left_axes  # 右侧放置剩余的轴

    # 为所有轴生成足够的位置
    left_positions = []
    right_positions = []

    # 生成左侧轴的位置
    for i in range(left_axes):
        if i == 0:
            # 第一个左侧轴固定在最左边
            left_positions.append(0)
        else:
            # 其他左侧轴均匀分布，确保有足够的空间
            step = 0.05
            pos = min(i * step, 0.4)  # 限制最大位置为0.4，避免太靠近中间
            left_positions.append(pos)

    # 生成右侧轴的位置
    for i in range(right_axes):
        # 右侧轴均匀分布
        step = 0.05
        pos = max(1 - i * step, 0.6)  # 限制最小位置为0.6，避免太靠近中间
        right_positions.append(pos)

    # 设置Y轴范围以显示每列的完整数据范围
    left_count = 0
    right_count = 0
    
    # 预先计算每列的范围，以便设置合适的刻度间隔
    y_ranges = {}
    for col in selected_columns:
        min_val = plot_df[col].min()
        max_val = plot_df[col].max()
        data_range = max_val - min_val
        
        # 确保范围不为零，防止除以零错误
        if abs(data_range) < 0.001:
            data_range = 0.001
        
        # 计算刻度间隔为数据范围的20%
        tick_interval = data_range * 0.2
        
        # 确保tick_interval不为零
        if abs(tick_interval) < 0.0001:
            tick_interval = 0.1  # 设置一个默认的最小间隔值
        
        # 四舍五入到合适的小数位
        if tick_interval >= 1:
            tick_interval = round(tick_interval)
        elif tick_interval >= 0.1:
            tick_interval = round(tick_interval, 1)
        else:
            tick_interval = round(tick_interval, 2)
        
        # 再次确保tick_interval不为零（四舍五入可能导致零）
        if tick_interval == 0:
            tick_interval = 0.1  # 设置一个默认的最小间隔值
        
        # 存储范围信息和刻度间隔
        y_ranges[col] = {
            'min': min_val,
            'max': max_val,
            'range': data_range,
            'interval': tick_interval,
            # 确保至少有5个刻度，并防止除零错误
            'nticks': max(5, min(10, 5))  # 简单设置为固定值5，避免除法运算
        }
    
    for i, col in enumerate(selected_columns):
        # 获取颜色 - 使用之前定义的颜色
        color = colors[i % len(colors)]
        
        # 分配轴的位置
        if i == 0 or (i >= 2 and i % 2 == 0):
            # 第一个轴和奇数序号的轴放在左侧
            side = "left"
            # 确保索引不会超出范围
            if left_count < len(left_positions):
                position = left_positions[left_count]
            else:
                # 如果索引超出范围，使用最后一个位置并稍微调整
                position = left_positions[-1] + 0.05 if left_positions else 0
            left_count += 1
            anchor = "free" if position > 0 else "x"
        else:
            # 第二个轴和偶数序号的轴放在右侧
            side = "right"
            # 确保索引不会超出范围
            if right_count < len(right_positions):
                position = right_positions[right_count]
            else:
                # 如果索引超出范围，使用最后一个位置并稍微调整
                position = right_positions[-1] - 0.05 if right_positions else 1
            right_count += 1
            anchor = "free" if position < 1 else "x"
        
        # 获取刻度设置
        y_range = y_ranges[col]
        
        # 为该列添加专用Y轴，并设置与数据线相同的颜色
        fig.update_layout(
            {f"yaxis{i+1 if i > 0 else ''}": {
                # 将轴标题设为空，不显示默认轴标题
                "title": {
                    "text": "",  # 空标题
                    "font": {"color": color, "size": 12},
                    "standoff": 0  # 减少标题和轴的间距
                },
                "side": side,
                "overlaying": "y" if i > 0 else None,
                "position": position,  # 位置在0-1范围内
                "domain": [0.1, 0.9],  # 确保所有Y轴在图表高度范围内
                "anchor": anchor,
                "autorange": False,  # 不使用自动范围，而是设置固定范围
                "range": [y_range['min'] - 0.05 * y_range['range'], 
                        y_range['max'] + 0.05 * y_range['range']],  # 数据范围上下多留5%的空间
                "tickfont": {"color": color, "size": 10},  # 刻度标签颜色和大小
                "tickmode": "linear",  # 使用线性刻度模式
                "tick0": y_range['min'],  # 起始刻度
                "dtick": y_range['interval'],  # 刻度间隔
                "ticklen": 5,  # 刻度线长度
                "tickwidth": 1.5,  # 刻度线宽度
                "tickcolor": color,  # 刻度线颜色
                "linecolor": color,  # 轴线颜色
                "linewidth": 2,  # 轴线宽度
                "showline": True,  # 显示轴线
                "mirror": "all",  # 强制显示轴线
                "zeroline": False,  # 不显示零线
                "showgrid": True,  # 显示网格线
                "gridcolor": "rgba(200,200,200,0.2)",  # 使用安全的网格线颜色
                "gridwidth": 1,  # 网格线宽度
                "tickformat": ".1f" if y_range['interval'] < 1 else "",  # 根据间隔大小决定小数位数
            }}
        )
        
        # 设置线条对应的轴
        if i > 0:
            fig.update_traces(yaxis=f"y{i+1}", selector=dict(name=col))
    
    # 把图例移到图表外部顶部，使其不会挡住图表内容
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            itemsizing="constant"
        )
    )
    
    # 计算时间轴的刻度间隔为5秒
    x_min = plot_df['时间(s)'].min()
    x_max = plot_df['时间(s)'].max()
    
    # 确保起点为0
    tick_start = 0
    
    # 计算刻度点间隔为5秒
    time_range = x_max - x_min
    # 安全地计算刻度
    try:
        # 创建一个从0开始，以5为间隔的刻度序列
        max_seconds = int(time_range) + 5  # 向上取整并加5
        tick_values = list(range(0, max_seconds + 5, 5))
        tick_texts = [f"{t}" for t in tick_values]
    except Exception as e:
        st.warning(f"刻度计算错误: {str(e)}，将使用自动刻度")
        tick_values = None
        tick_texts = None
    
    # 更新布局，设置足够的边距来容纳所有Y轴
    xaxis_config = {
        "domain": [0.1, 0.9],  # 根据轴数量调整图表区域
        "title": "时间(s)",
        "showgrid": True,
        "gridcolor": "rgba(200,200,200,0.2)",  # 使用安全的网格线颜色
        "dtick": 5  # 默认网格线为5秒间隔
    }
    
    # 如果成功计算了刻度，添加到配置中
    if tick_values and tick_texts:
        xaxis_config.update({
            "tickmode": 'array',
            "tickvals": tick_values,
            "ticktext": tick_texts
        })
    
    # 计算合适的页边距
    left_margin = 50 + (left_axes * 40)  # 增加左侧页边距
    right_margin = 50 + (right_axes * 40)  # 增加右侧页边距
    
    # 更新全局布局设置
    fig.update_layout(
        title="选定列的时序图",
        xaxis=xaxis_config,
        legend_title="数据列",
        height=800,  # 图表高度设为800px
        plot_bgcolor="white",  # 白色背景使颜色更明显
        margin=dict(
            l=left_margin, 
            r=right_margin, 
            t=70, 
            b=50
        ),
        autosize=True,  # 自动调整大小
        # 设置垂直线交互模式
        hovermode="x unified",  # 显示一个垂直线并汇总所有数据
        # 自定义悬停信息
        hoverlabel=dict(
            bgcolor="white",  # 悬停标签背景色
            font_size=12,  # 悬停标签字体大小
            font_family="Arial"  # 悬停标签字体
        ),
    )
    
    # 使用全宽容器显示图表，并确保它填满可用空间
    st.plotly_chart(fig, use_container_width=True)

def st_main():
    # 设置为宽屏模式，最大化使用页面空间
    st.set_page_config(layout="wide", page_title="数据记录分析")
    
    # 初始化session_state用于存储点击数据
    if 'plotly_click_data' not in st.session_state:
        st.session_state['plotly_click_data'] = None
    
    # 让标题更紧凑
    st.markdown("# 数据记录分析")

    # 创建侧边栏，减小宽度让主区域有更多空间
    with st.sidebar:
        st.markdown("### 设置")
        
        uploaded_file = st.file_uploader("请选择数据记录文件")
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
        else:
            df = load_data()
            
        st.markdown("### 请选择要分析的数据列")
        
        # 使用多选下拉菜单替代多个复选框
        columns = df.columns.tolist()
        columns_without_time = [col for col in columns if col != TIME_COLUMN]
        
        # 多选下拉菜单 - 确保括号闭合
        selected_columns = st.multiselect(
            "选择数据列",
            options=columns_without_time,
            default=[]
        )
        
        # 添加绘制按钮
        plot_button = st.button("绘制时序图", use_container_width=True)
    
    # 使数据预览更紧凑
    with st.expander("数据预览 (点击展开)", expanded=False):
        st.dataframe(df.head(10), height=200, use_container_width=True)
    
    # 为图表添加一个占满页面宽度的容器
    container = st.container()
    
    # 主区域显示图表
    if plot_button:
        if TIME_COLUMN not in df.columns:
            st.error(f"数据中缺少时间列: {TIME_COLUMN}")
        elif not selected_columns:
            st.warning("请至少选择一列进行可视化")
        else:
            # 使用占据页面全宽的容器来显示图表
            with container:
                plot_normalized_time_series(df, selected_columns)

if __name__ == "__main__":
    st_main()
