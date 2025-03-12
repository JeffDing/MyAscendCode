import pandas as pd

# 读取JSON文件
json_file = 'input.json'  # 替换为你的JSON文件路径
df = pd.read_json(json_file)

# 将数据写入Parquet文件
parquet_file = 'output.parquet'  # 替换为你想保存的Parquet文件路径
df.to_parquet(parquet_file, engine='pyarrow')

print(f"JSON文件已成功转换为Parquet文件: {parquet_file}")