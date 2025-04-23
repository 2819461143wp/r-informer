import pandas as pd

# 读取原始 Excel 数据，指定 openpyxl 引擎
df = pd.read_excel('./data/ETT/new_data.xlsx', engine='openpyxl')

# 重命名列，移除特殊字符
column_mapping = {
    '检测时间': 'date',
    '水温(℃)': 'water_temp',
    'pH': 'ph',
    '溶解氧(mg/L)': 'O2',
    '浊度(NTU)': 'turbidity',
    '电导率(uS/cm)': 'conductivity'
}

# 重命名列
df = df.rename(columns=column_mapping)

# 确保时间列格式正确
df['date'] = pd.to_datetime(df['date'])

# 按时间排序
df = df.sort_values('date')

# 保存为 CSV，确保正确的格式
df.to_csv('./data/ETT/new_data.csv', index=False, encoding='utf-8-sig')