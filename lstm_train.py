import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import EarlyStopping
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import math
from keras.callbacks import ModelCheckpoint




symbol="AMD"
df=pd.read_csv(f"{symbol}_features.csv", parse_dates=True, index_col='Date')
print(df)
X = np.array(df.drop(["Next_10_Days_Volatility"], axis=1).values)
#,'Low','High','Close','Open','Volume','MACD_h','MACD_sl','RSI14','SMA14','EMA14'
y = np.array(df["Next_10_Days_Volatility"].values).reshape(-1, 1)
print(X.shape)

train_size=int(df.shape[0] * 0.8)

X_train = X[:train_size,]
X_test = X[train_size:,]
y_train = y[:train_size]
y_test = y[train_size:]
print(y_test)


def create_sequences(x, y, time_steps):
    xs, ys = [], []
    for i in range(len(x) - time_steps):
        xs.append(x[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(xs), np.array(ys)
T =30

X_train, y_train = create_sequences(X_train, y_train,T)
X_test, y_test =create_sequences(X_test, y_test, T)



inputLSTM = Input(shape=(X_train.shape[1], X_train.shape[2]))
y = LSTM(200, return_sequences=True)(inputLSTM)
y = LSTM(200)(y)
y = Dense(1)(y)
lstm = Model(inputs=inputLSTM, outputs=y)
early_stopping = EarlyStopping(
    monitor='val_loss',     # 监控验证集损失
    patience=20,            # 如果20个epoch内val_loss没有改善，则停止训练
    min_delta=0.0001,       # 被视为改善的最小变化量
    mode='min',             # 监控的指标应该是越小越好
    restore_best_weights=True
    # 恢复训练期间具有最佳性能的模型权重
)

lstm.compile(optimizer=keras.optimizers.Adam(lr=0.001),
             loss=tf.keras.losses.MeanSquaredError(),
             metrics=[tf.keras.metrics.RootMeanSquaredError()])
checkpoint = ModelCheckpoint(
    filepath=f'{symbol}_best_lstm_model.h5',  # 保存路径
    monitor='val_loss',             # 监控验证损失
    save_best_only=True,            # 只保存最佳模型
    mode='min',                     # 监控指标越小越好
    verbose=1                       # 显示保存信息
)

hist = lstm.fit(X_train, y_train,batch_size=32
                 ,epochs=400,verbose=1,validation_split=0.2,shuffle=False,callbacks=[checkpoint]
             )
from tensorflow.keras.models import load_model
lstm = load_model('best_lstm_model.h5')
plt.plot(hist.history['root_mean_squared_error'])
plt.plot(hist.history['val_root_mean_squared_error'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
for ind, i in enumerate(lstm.predict(X_test)):
    print('Prediction: ' + str('{:.2f}'.format(round(round(i[0], 4), 3))) + ',    ' + 'Actual Value: ' + str(
        '{:.2f}'.format(round( round(y_test[ind][0], 4), 2))))

# plot the prediction
plt.figure(figsize=(10, 6))
plt.plot(lstm.predict(X_test), label='Predicted')
plt.plot(y_test, label='Actual')
plt.title('Prediction vs Actual')
plt.legend()
plt.show()

def printing_out_results_of_a_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Print the R2 score

    print("R2 score:\n")
    print(('{:.2f}'.format((100 * (r2_score(y_test, y_pred))))) + " %")

    print("\n")

    # Print the RMSE

    print("RMSE:\n")
    print(math.sqrt(mean_squared_error(y_test, y_pred)))

    print('\n')

    # Print the mean squared error

    print("Mean Squared Error:\n")
    print(mean_squared_error(y_test, y_pred))
    results_df = pd.DataFrame({
        'Actual': y_test.flatten(),
        'Predicted': y_pred.flatten(),
        'Return': df['Log_Returns'].iloc[-len(y_test):]
    },index=df.index[-len(y_test):])

    results_df.to_csv(f"{symbol}_lstm_results.csv", date_format='%Y-%m-%d', index_label='Date')


printing_out_results_of_a_model(lstm, X_test, y_test)