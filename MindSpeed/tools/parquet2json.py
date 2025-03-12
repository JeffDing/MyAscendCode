import pyarrow.parquet as pq
import pandas as pd

# 读取Parquet文件
table = pq.read_table('train-00000-of-00001-a09b74b3ef9c3b56.parquet')

# 将Parquet数据转换为DataFrame
df = table.to_pandas()

# 将DataFrame转换为JSON格式
json_data = df.to_json(orient='records', lines=True)

# 将JSON数据写入文件
with open('train-00000-of-00001-a09b74b3ef9c3b56.json', 'w') as f:
    f.write(json_data)