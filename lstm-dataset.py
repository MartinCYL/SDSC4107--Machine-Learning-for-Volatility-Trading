from ta.momentum import RSIIndicator
from ta.trend import MACD
import ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from arch import arch_model

symbol="NFLX"
df=pd.read_csv(f"{symbol}.csv", parse_dates=True, index_col='Date')


# MACD - 移动平均收敛/发散
df['Log_Returns'] = (np.log(df.Close) - np.log(df.Close.shift(1)))*100
df['Log_Trading_Range'] = (np.log(df.High) - np.log(df.Low))*100
df['Log_Volume_Change'] =(np.log(df.Volume) - np.log(df.Volume.shift(1))) *100
df['Previous_10_Day_Volatility'] = df['Log_Returns'].rolling(window = 60).std()*np.sqrt(252)
df['Previous_30_Day_Volatility'] = df['Log_Returns'].rolling(window = 80).std()*np.sqrt(252)
df['Next_10_Days_Volatility'] = df['Log_Returns'].rolling(window = 60).std().shift(-1)*np.sqrt(252)
df.dropna(inplace=True)

train_size=int(df.shape[0] * 0.8)
# X=df.iloc[:train_size]

import arch_forecast
from arch_forecast import arch_model
import pandas as pd
import numpy as np

# 假设 X 是您的数据，'Log_Returns' 是您要建模的列名
# 并且您已经将数据提取到名为 log_returns 的 pandas Series 中

# 如果 X 是 DataFrame，则提取 'Log_Returns' 列
# 确保 X 是 DataFrame 并且 'Log_Returns' 列存在
# 示例：X = pd.DataFrame({'Log_Returns': np.random.randn(100)})  # 创建一个示例 DataFrame
log_returns = df['Log_Returns'].values
# 定义滚动窗口的大小
window_size = 42
# 创建一个列表来存储滚动预测
rolling_predictions = []
# 循环遍历数据，进行滚动预测
for i in range(window_size, len(log_returns)):
    # 使用滚动窗口中的数据拟合 GARCH(1,1) 模型

    model = arch_model(log_returns[i-window_size:i], p=1, q=1,
                      mean='Zero', vol='GARCH', dist='Normal')
    results = model.fit(disp='off')

    # 进行单步预测
    forecast = results.forecast(horizon=1)
    forecasted_vol = np.sqrt(forecast.variance.iloc[-1, 0]) * np.sqrt(252)

    # 将预测值添加到列表中
    rolling_predictions.append(forecasted_vol) # 使用方差作为预测值

# 将预测值转换为 pandas Series
GARCH_rolling_predictions = pd.DataFrame(rolling_predictions, index=df.iloc[window_size:].index, columns=['GARCH_rolling_predictions'])

log_returns = df['Log_Returns']
print(log_returns)


# 定义 GARCH(1,1) 模型
model = arch_model(log_returns, p=1, q=1,
                  mean='Zero', vol='GARCH', dist='Normal')

# 拟合模型
results = model.fit(disp='off')

# 进行前瞻性预测
forecast = results.forecast(horizon=df.shape[0]-train_size)





print(GARCH_rolling_predictions)
df = pd.concat([df, GARCH_rolling_predictions], axis=1)
df['GARCH_rolling_predictions'] =  df['GARCH_rolling_predictions'].fillna(0)
df.drop(['Low','High','Close','Open','Volume'],axis=1,inplace=True)
print(df)
df.to_csv(f"{symbol}_features.csv")


