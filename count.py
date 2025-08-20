import pandas as pd

# 1. 读取数据
df = pd.read_csv('all_data.csv')

# 2. 定义需要检查的参数列
required_params = ['gain', 'drive', 'treble', 'mid', 'bass']

# 3. 判断四列都非空
mask = df[required_params].notnull().all(axis=1)

# 4. 统计满足条件的行数
count = mask.sum()

print(f"同时具有 gain、treble、mid、bass 这四项的项目数量为: {count}")
