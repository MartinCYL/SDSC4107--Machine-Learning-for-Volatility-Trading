import pandas as pd
import numpy as np



symbol= "AMD"
option=pd.read_csv(f"{symbol}_option.csv",index_col='Date', parse_dates=True)
prediction=pd.read_csv(f"{symbol}_lstm_results.csv",index_col='Date', parse_dates=True)

tmp = option.join(
    prediction,
    how='left'  # 只保留option中存在的索引
)

tmp['diff']=tmp['Forecasted_Volatility']-tmp['IVOL_MID']

action = [0]
state = 0  # 0:无持仓, 1:做多期权, -1:做空期权
snumber = []  # 对冲股票数量

for r, i in enumerate(tmp["diff"]):
    if i > 0 and state == 0:
        # 隐含波动率高估，做空期权 + 用股票对冲
        action.append(-1)
        state = -1
        snumber.append(tmp['DELTA_MID'][r])
    elif i < 0 and state == 0:
        # 隐含波动率低估，做多期权 + 用股票对冲
        action.append(1)
        state = 1
        snumber.append(-tmp['DELTA_MID'][r])
    elif state == -1 and i < 0:
        # 平空仓
        action.append(1)
        state = 0
        snumber.append(snumber[-1])
    elif state == 1 and i > 0:
        # 平多仓
        action.append(-1)
        state = 0
        snumber.append(snumber[-1])
    else:
        action.append(0)
        snumber.append(snumber[-1])

tmp["action"] = action[1:]
tmp["snumber"] = snumber
tmp = tmp[tmp["action"]!=0]

    # pnl
if len(tmp)%2 == 1:tmp=tmp.iloc[:-1,:]
res_h = []
for i in range(1,len(tmp),2):
    opt = -tmp['OP_MID'][i]*tmp["action"][i]-tmp['OP_MID'][i-1]*tmp["action"][i-1]
    stock = (tmp['PX_MID'][i]-tmp['PX_MID'][i-1])*tmp["snumber"][i]
    res_h.append(opt+stock)


print("cumulative return of the strategy:",round(sum( res_h),3))
print("average return of the strategy:",round(np.mean( res_h),3))
print("total trade:",len( res_h))
