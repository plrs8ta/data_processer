# %%
import streamlit as st
import polars as pl
import numpy as np
import plotly.graph_objects as go

def load_data(file_path: str = r"C:\Users\EDY\Desktop\PROJECT\PLC\frequency_analysis\2025_02_27_11_42_52.txt") -> pl.DataFrame:
    """加载时序数据文件"""
    try:
        df = pl.read_csv(file_path, separator=" ")
        return df
    except Exception as e:
        st.error(f"文件加载错误: {str(e)}")
        return None

def plot_time_series(signal: np.ndarray, t: np.ndarray) -> go.Figure:
    """绘制时序图"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t,
        y=signal,
        mode='lines',
        name='时域信号'
    ))
    fig.update_layout(
        title="时域信号",
        xaxis_title="时间 (s)",
        yaxis_title="振幅",
        showlegend=True
    )
    return fig

def plot_spectrum(signal: np.ndarray, Fs: float) -> go.Figure:
    """根据时序数据计算并绘制频谱图"""
    # 计算FFT
    N = len(signal)
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1/Fs)
    
    # 只保留正频率部分
    freqs = freqs[:N//2]
    amplitudes = np.abs(fft_result)[:N//2]

    # 绘制频谱图
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs,
        y=amplitudes,
        mode='lines',
        name='频谱'
    ))
    fig.update_layout(
        title="频谱图",
        xaxis_title="频率 (Hz)",
        yaxis_title="幅值",
        showlegend=True
    )
    return fig

def plot_log_spectrum(signal: np.ndarray, Fs: float) -> go.Figure:
    """计算对数频谱图"""
    # 计算FFT
    N = len(signal)
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, d=1/Fs)
    
    # 只保留正频率部分
    freqs = freqs[:N//2]
    amplitudes = np.abs(fft_result)[:N//2]

    # 计算对数幅度（dB）
    log_amplitudes = 20 * np.log10(amplitudes)

    # 绘制对数频谱图
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs,
        y=log_amplitudes,
        mode='lines',
        name='对数频谱'
    ))
    fig.update_layout(
        title="对数频谱图",
        xaxis_title="频率 (Hz)",
        yaxis_title="幅值 (dB)",
        showlegend=True
    )
    return fig

def main():
    st.title("时序数据频谱分析工具")
    
    # 文件上传
    uploaded_file = st.file_uploader("请上传数据文件 (.txt)", type=["txt"])

    if uploaded_file == None:
        uploaded_file = r"C:\Users\EDY\Desktop\PROJECT\PLC\frequency_analysis\2025_02_27_11_42_52.txt"
    
    if uploaded_file is not None:
        # 加载数据
        df = load_data(uploaded_file)
        
        if df is not None:
            # 显示数据预览
            st.subheader("数据预览")
            st.write(df.head())
            
            # 选择要分析的列
            columns = df.columns
            selected_column = st.selectbox("选择要分析的列", columns)
            
            # 设置采样间隔
            sample_interval = st.number_input(
                "采样间隔 (秒)",
                value=0.02,
                min_value=0.001,
                max_value=1.0,
                step=0.001
            )
            
            # 设置采样频率
            Fs = 1 / sample_interval  # 采样频率 = 1 / 采样间隔

            # if st.button("生成时序图与频谱图"):
                # 获取选定列的数据
            signal = df[selected_column].to_numpy()
            t = np.arange(0, len(signal) * sample_interval, sample_interval)

            # 绘制时序图
            time_series_fig = plot_time_series(signal, t)
            
            # 绘制频谱图
            spectrum_fig = plot_spectrum(signal, Fs)
            
            # 绘制对数频谱图
            log_spectrum_fig = plot_log_spectrum(signal, Fs)
            
            # 显示图表
            st.plotly_chart(time_series_fig)
            st.plotly_chart(spectrum_fig)
            st.plotly_chart(log_spectrum_fig)

if __name__ == "__main__":
    main()

# %%
