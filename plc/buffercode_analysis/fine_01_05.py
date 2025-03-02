# %%
import pandas as pd
import polars as pl
import os
from pathlib import Path

PATH_FLODER = Path(r"C:\Users\EDY\Desktop\mmc_QP10beifen\data\statuscode")

def combine_txt_2pldf(folder_path):
    
    all_data = []

    # 遍历文件夹中的所有txt文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            try:
                # 读取文件，从第13行开始，使用正则表达式处理多个空格作为分隔符
                df = pd.read_csv(file_path, skiprows=12, sep=r'\s+', engine='python')
                df = df.reset_index()
                all_data.append(df)
                print(f"成功处理文件：{filename}")
            except Exception as e:
                print(f"处理文件 {filename} 时发生错误：{e}")

    if all_data:
        combined_df = pd.concat(all_data)
        combined_df = pl.from_pandas(combined_df)
        return combined_df
    
    else:
        print("没有成功处理任何文件。")
        return None

df = combine_txt_2pldf(PATH_FLODER)

if df is not None:
    print(df)

# %%
