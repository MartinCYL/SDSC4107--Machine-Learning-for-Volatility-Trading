import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2. 计算对数收益率
symbol= "AMD"
df=pd.read_csv(f"{symbol}.csv")
prices= df['Close']

log_returns = np.log(prices / prices.shift(1))*100
log_returns=log_returns.fillna(0)
log_returns.index=df['Date']
log_returns.name = "Return"


# 3. 生成波动率时间序列（30天滚动窗口，年化）
window =  60 # 窗口天数
rolling_vol = log_returns.rolling(window).std() * np.sqrt(252)


all_df = pd.DataFrame({
    'Volatility': rolling_vol,
    'Return': log_returns
}, index=df['Date'])


all_df.to_csv(f"{symbol}_rolling_vol.csv")
print(log_returns)

# train_test_split

# 4. 训练集和测试集划分
train_size = int(len(log_returns) * 0.8)
train = all_df.iloc[:train_size]
test = all_df.iloc[train_size:]

# 5. 训练集和测试集保存
train.to_csv(f"{symbol}_train.csv")
test.to_csv(f"{symbol}_test.csv")
print(train)
print(test)

