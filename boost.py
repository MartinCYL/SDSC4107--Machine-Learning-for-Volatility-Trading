from ta.momentum import RSIIndicator
from ta.trend import MACD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



symbol="AMD"

data=pd.read_csv(f"{symbol}.csv")
data['Return'] = np.log(data['Close'] / data['Close'].shift(1))*100
data['Volatility']=data['Return'].rolling(window=60).std() * np.sqrt(252)
data['MA_21'] = data['Close'].rolling(window=21).mean()
rsi = RSIIndicator(data['Close'], window=14)
data['RSI'] = rsi.rsi()
macd = MACD(data['Close'])
data['MACD'] = macd.macd_diff()
data.dropna(inplace=True)
# Create target variable
data['Future_Volatility'] = data['Volatility'].shift(-1)  # Forecast 21 days ahead

# Drop NaN values
data.dropna(inplace=True)

# Features and target
X = data[['MA_21', 'RSI', 'MACD', 'Volatility']]

y = data['Future_Volatility']

# Split data into training and testing sets
train_size = int(len(data) * 0.8)
X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]


# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Actual Volatility', color='blue')
plt.plot(y_pred, label='Predicted Volatility', color='red')
plt.title('Actual vs Predicted Volatility')
plt.legend()
plt.show()

# save test_prediction
test_prediction = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Return': data['Return'].iloc[train_size:]
},  index=data['Date'].iloc[train_size:])



# save test_prediction to csv
test_prediction.to_csv(f"{symbol}_boost_forecast.csv", date_format='%Y-%m-%d', index_label='Date')


