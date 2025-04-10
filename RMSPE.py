import numpy as np
import pandas as pd

# 读取数据
data_file = 'AMD_garch_forecast.csv'
df = pd.read_csv(data_file, parse_dates=['Date'])

# 剔除预测波动率异常低的行（预测波动率小于阈值的行）
threshold = 10
df = df[df['Forecasted_Volatility'] >= threshold].copy()

# 定义一个极小值，避免除零错误
epsilon = 1e-6

# 计算百分比误差和均方根百分比误差（RMSPE）
df['Percentage_Error'] = (df['Actual_Volatility'] - df['Forecasted_Volatility']) / (df['Actual_Volatility'] + epsilon)
df['Squared_Percentage_Error'] = df['Percentage_Error'] ** 2
rmspe = np.sqrt(df['Squared_Percentage_Error'].mean())

print("RMSPE:", rmspe)
