import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

# 假设 train_data 和 test_data 已经加载好，并且包含收益率数据
# 我们假设收益率列名为 'Return'，如果不是，请替换为实际的列名
symbol = "AMD"
train_data = pd.read_csv(f'{symbol}_train.csv', index_col='Date', parse_dates=True)
test_data = pd.read_csv(f'{symbol}_test.csv', index_col='Date', parse_dates=True)
returns_train = train_data['Return']
returns_test = test_data['Return']

# 1. 在训练数据上拟合 GARCH(1,1) 模型
garch_model = arch_model(returns_train, vol='ARCH', mean='zero', p=1)
garch_fit = garch_model.fit(disp='off')

# 打印模型摘要
print(garch_fit.summary())

# 获取 AIC 和 BIC
aic = garch_fit.aic
bic = garch_fit.bic
print(f"AIC: {aic}")
print(f"BIC: {bic}")

# 2. 在测试数据上进行滚动预测
forecasts = []
actual_vol = []
rolling_window = returns_train.copy()
rolling_window= rolling_window.iloc[-42:]

plt.hist(train_data['Return'], bins = 100)
plt.show()

for i in range(len(returns_test)):
    # 使用当前窗口拟合模型
    vol_realized= np.std(rolling_window)*np.sqrt(252)
    model = arch_model(rolling_window, vol='ARCH', mean='zero', p=1)
    fit_model = model.fit(disp='off', show_warning=False)

    # 预测下一步的波动率
    forecast = fit_model.forecast(horizon=1)
    forecasted_vol = np.sqrt(forecast.variance.iloc[-1,0])*np.sqrt(252)
    forecasts.append(forecasted_vol)

    # 更新滚动窗口，添加新的观测值
    rolling_window = pd.concat([rolling_window, pd.Series([returns_test.iloc[i]],
                                                          index=[returns_test.index[i]])])
    # 移除最早的观测值，保持窗口大小不变
    rolling_window = rolling_window.iloc[1:]
    actual_vol.append(vol_realized)


# 3. 评估预测结果
# 将预测结果和实际波动率转换为DataFrame
forecast_df = pd.DataFrame({
    'Forecasted_Volatility': forecasts,
    'Actual_Volatility': actual_vol,
    'Return': returns_test
}, index=returns_test.index.values)

# 计算均方根误差 (RMSE)
rmse = math.sqrt(mean_squared_error(forecast_df['Actual_Volatility'],
                                    forecast_df['Forecasted_Volatility'] ))
print(f"RMSE: {rmse}")

# 4. 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(forecast_df.index, forecast_df['Actual_Volatility'],
         label='Realized Volatility (|Return|)')
plt.plot(forecast_df.index, forecast_df['Forecasted_Volatility'],
         label='Forecasted Volatility', color='red')
# plt.title(f'GARCH(1,1) Volatility Forecast vs Realized Volatility\nRMSE: {rmse:.6f}')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. 保存预测结果
forecast_df.to_csv(f'{symbol}_arch_forecast.csv', date_format='%Y-%m-%d', index_label='Date')