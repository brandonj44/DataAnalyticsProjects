
import pandas as pd
import numpy as np

df=pd.read_csv("C:/Users/velez/OneDrive/Desktop/School/D213/medical_time_series .csv", parse_dates=True)
df=df.dropna()
print('DF shape:',df.shape)
print(df.head())
print(df.tail())

# datetime index original dataset
index_df = pd.date_range(start='2015-1-01', end='2016-12-31')
df.index = index_df

# line graph revenue over time 
df['Revenue'].plot(figsize=(12,5), title='Daily Revenue (in Mil) over Two Years LIne chart', xlabel='Day', ylabel='$MM of Revenue')

# ADfuller on df['Revenue']
from statsmodels.tsa.stattools import adfuller
def ad_test(dataset):
    dftest=adfuller(dataset, autolag='AIC')
    print("1 ADF: ", dftest[0])
    print("2 P value:", dftest[1])
    print("3 num of lags:", dftest[2])
    print("4 num of obsvs for adf regr & critical values calc:", dftest[3])
    print("5 critical values: ")
    for key, val in dftest[4].items():
        print("\t",key,": ",val)
ad_test(df['Revenue'])

# Export Clean Dataset 
df.to_csv("C:/Users/velez/OneDrive/Desktop/School/D213/CleanData.csv")

# Initial ARIMA model on ORIGINAL DATASET
#needed to run 'pip3 install pmdarima' in anaconda prompt 
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
stepwise_fit=auto_arima(df['Revenue'], trace=True, suppress_warnings=True)
stepwise_fit.summary()

#best model from last step was :  ARIMA(1,1,0)(0,0,0)[0]
from statsmodels.tsa.arima.model import ARIMA
print(df.shape)

#define train and test
train = df.iloc[:-30]
test = df.iloc[-30:]
print(train.shape,test.shape)

train.to_csv("C:/Users/velez/OneDrive/Desktop/School/D213/train.csv")
test.to_csv("C:/Users/velez/OneDrive/Desktop/School/D213/test.csv",)

# model ARIMA using recommended order on train
model=ARIMA(train['Revenue'],order=(1,1,0))
model = model.fit()
print("Train model:", model.summary())

# define pred
start=len(train)
end=len(train)+len(test)-1
pred=model.predict(start=start,end=end,type='levels')

# set pred index 1 after df ends
pred.index=df.index[start:end+1]
print(pred)

print("Pred:", pred)

#plot test data
pred.plot(legend=True)
test['Revenue'].plot(legend=True)

print("Test Revenue mean:", test['Revenue'].mean())

#calculate RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(pred,test['Revenue']))
print(rmse)

# ARIMA model on OG data
model2 = ARIMA(df['Revenue'],order=(1,1,0))
model2 = model2.fit()
print(df.tail())

# set forecast datetime index
index_future_dates = pd.date_range(start= '2017-1-01' , end='2017-03-31')
print(index_future_dates)

#define pred and match to forecast datetime index
pred=model2.predict(start=len(df), end=len(df)+89, type='levels').rename('Arima pred')
pred.index=index_future_dates
print(pred.head(10))
# plot forecasted values
pred.plot(legend=True, use_index=90)

## decompose
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(df['Revenue'], model='additive', period=12)
decomp.seasonal.plot()
decomp.plot()

from statsmodels.tsa.seasonal import STL
stl = STL(df['Revenue'], period = 731)
res = stl.fit()
fig = res.plot()

# PSD
import matplotlib.pyplot as plt
plt.psd(df['Revenue'])
plt.show()

#spectral density cycle and freq graphs
np.random.seed(696969)
diff = 0.01
ax = np.arange(0,10,diff)
n = np.random.randn(len(ax))
by = np.exp(-ax/0.05)
cn = np.convolve(n, by) * diff
cn = cn[:len(ax)]
s = 0.1 * np.sin(2*np.pi*ax) +cn
plt.subplot(211)
plt.plot(ax,s)
plt.subplot(212)
plt.psd(s, 512, 1/diff)
plt.show()

# ARIMA presentation 2
from matplotlib.pylab import rcParams
rcParams['figure.figsize']= 10,6
indexedDataset = df

#monthly rol mean and std
rolmean = indexedDataset.rolling(window=12).mean() 
rolstd = indexedDataset.rolling(window=12).std()
print(rolmean, rolstd)

#plot rol mean and std vs rev
orig = plt.plot(indexedDataset, color='blue', label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Std Dev vs Revenue')
plt.show(block=False)

#check the stationarity
print ('Results of Dickey-Fuller test: ')
dftest = adfuller(indexedDataset['Revenue'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','No. of Observations'])
for key,value in dftest[4].items():
	dfoutput['Critical Value (%s) '%key] = value
print(dfoutput)

# plot the log of df
indexedDataset_logScale=np.log(indexedDataset)
plt.plot(indexedDataset_logScale) 

# plot Moving avg and STD of log
movingAverage = indexedDataset_logScale.rolling(window=12).mean()
movingSTD = indexedDataset_logScale.rolling(window=12).std()
plt.plot(indexedDataset_logScale)
plt.plot(movingAverage,label='Moving Average Logscale', color='red')
# define logscale - mvng avg
datasetLogScaleMinusMovingAverage = indexedDataset_logScale - movingAverage  
datasetLogScaleMinusMovingAverage.head(12)              
#need to Remove NaN valuesafter prev calc
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)
#adfuller stationarity test on log df
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    std = plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation Shows a Trend')
    plt.show(block=False)
    print ('Results of Dickey-Fuller test: ')
    dftest = adfuller(timeseries['Revenue'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','No. of Observations'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s) '%key] = value # Critical Values should always be more than the test statistic
    print(dfoutput)

# graphs from above calcs
test_stationarity(datasetLogScaleMinusMovingAverage)
exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(indexedDataset_logScale)
plt.plot(exponentialDecayWeightedAverage)

# Exponential Decay
datasetLogScaleMinusMovingExponentialDecayAverage = indexedDataset_logScale - exponentialDecayWeightedAverage
datasetLogScaleMinusMovingExponentialDecayAverage = datasetLogScaleMinusMovingExponentialDecayAverage.dropna()
test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)

datasetLogDiffShifting = indexedDataset_logScale - indexedDataset_logScale.shift()
plt.plot(datasetLogDiffShifting)

datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)
## End of ARIMA 2
## Decomposing logscale data
indexedDataset_logScale['Revenue'] = indexedDataset_logScale['Revenue'].replace(-np.inf, np.nan)
indexedDataset_logScale = indexedDataset_logScale.dropna()

# index needs set to list and empty column dropped
indexedDataset_logScale = pd.DataFrame(indexedDataset_logScale, index= indexedDataset_logScale.index, columns=['Drop', 'Day', 'Revenue'])
indexedDataset_logScale = indexedDataset_logScale.drop(columns=['Drop', 'Day'])

## decomp logscale data
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(indexedDataset_logScale['Revenue'], model='additive', period=12)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.subplot(411)
plt.plot(indexedDataset_logScale, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

#ACF, PACF
datasetLogDiffShifting = pd.DataFrame(datasetLogDiffShifting, columns=['Drop', 'Day', 'Revenue'])
datasetLogDiffShifting = datasetLogDiffShifting.drop(columns=['Drop', 'Day'])

from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(datasetLogDiffShifting, nlags=20)
lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols')

# plot acf
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y= 1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')

#Plot pacf
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y= 1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

# ARIMA analysis 
# OG clean data
modelm= auto_arima(df['Revenue'],trace=True)
print(modelm.summary())

# plot log Scale dataset against AR fitted values
model = ARIMA(indexedDataset_logScale, order=(1,1,0))
results_AR = model.fit()
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('AR Plot')
model_fit = model.fit()
print("AR Model fit: ", model_fit.summary())

# plot MA fit against logscale
model = ARIMA(indexedDataset_logScale, order=(0,1,2))
results_MA = model.fit()
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('MA Model')
model = ARIMA(indexedDataset_logScale, order=(0,1,2))
model_fit = model.fit()
print("MA Model fit: ",  model_fit.summary())

#Build time series model
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(df['Revenue'],order=(0,1,2),seasonal_order=(1,1,0,90))
results = model.fit()
print(results.summary())

#Diagnostic plots
results.plot_diagnostics(figsize=[16,10]).show()

# slide 17 D3 dr elleh
# Generate predictions
diff_forecast=results.get_forecast(steps=90)
mean_forecast = diff_forecast.predicted_mean
confidence_intervals = diff_forecast.conf_int()
lower_limits = confidence_intervals.loc[:,'lower Revenue']
upper_limits = confidence_intervals.loc[:, 'upper Revenue']

prediction = results.get_prediction(start=732, end= 822)
mean_prediction=prediction.predicted_mean
confidence_intervals = prediction.conf_int()
lower_limits = confidence_intervals.loc[:,'lower Revenue']
upper_limits = confidence_intervals.loc[:, 'upper Revenue']

#plot forecast vs og dataset
plt.figure(figsize=(12,4))
plt.plot(train.index,train['Revenue'], color='g',label='observed (train set)')
plt.plot(test.index, test['Revenue'], label='observed (test set)')
plt.plot(mean_prediction.index, mean_prediction, color = 'r', label ='forecast')
plt.fill_between(lower_limits.index, lower_limits, upper_limits, color='pink', label='confidence interval')
plt.title('Forecast VS Test data')
plt.xlabel('Day')
plt.ylabel('Revenue ($MM)')
plt.legend()
plt.show()


# MAE
mae = np.mean(np.abs(results.resid))
print("MAE", mae)

