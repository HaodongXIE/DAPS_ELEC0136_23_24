#!/usr/bin/env python
# coding: utf-8

# ## Acquisition of Data

# In[1]:


from pandas_datareader import data as web
import pandas as pd
import datetime 
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yfin
yfin.pdr_override()
import warnings 
warnings.filterwarnings('ignore')


# ## Task 1: Data Acquisition
# ### Task 1.1 
# Select Microsoft and get the stock data from the beginning of April 2019 up to the end of March
# 2023.

# In[2]:


df = web.DataReader('MSFT', start = datetime.date(2019,4,1), end = datetime.date(2023,4,30))
df.head()


# In[3]:


df[df.columns[:-1]].plot(figsize=(10,5),alpha=0.7)
plt.grid()
plt.savefig('pricehistory.pdf',dpi=100)
plt.close()


# In[4]:


df[df.columns[-1]].plot(figsize=(10,5),alpha=0.7)
plt.grid()
plt.savefig('volumehistory.pdf',dpi=100)
plt.close()


# In[5]:


df['Ret'] = np.log(df['Close']).diff()
df['Ret_square'] = np.log(df['Close']).diff()**2


# In[6]:


df.corr()


# In[7]:


df.to_csv('MSFT.csv')


# ### Task 1.2  Alternative Data

# In[8]:


# Import Meteostat library and dependencies
import matplotlib.pyplot as plt
from meteostat import Point, Daily

# Set time period
start = datetime.datetime(2019, 4, 1)
end = datetime.datetime(2023, 4, 30)

# Create Point for Vancouver, BC
location = Point(40.69, -74, 70)

# Get daily data for 2018
data = Daily(location, start, end)
data = data.fetch()

# Plot line chart including average, minimum and maximum temperature
data.plot(y=['tavg', 'tmin', 'tmax'])
plt.savefig('weather.pdf',dpi=100)
plt.close()


# In[9]:


data.to_csv('NY_weather.csv')
df.to_csv('MSFT.csv')


# ## Feature Engineering

# In[10]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
figure, ax = plt.subplots(2,1,figsize=(15,10))
plot_acf(df['Close'],ax=ax[0])
plot_pacf(df['Close'],ax=ax[1])
plt.savefig('acf_pacf.pdf',dpi=100)
plt.close()
#plot_pacf(df['Volume'])


# ## Task 2: Data Storage

# ### Task2.1

# In[11]:


data.to_csv('NY_weather.csv')


# In[12]:


whole_data = pd.concat([df,data],axis=1)
whole_data.describe()


# In[13]:


whole_data.isnull().mean(0)


# In[14]:


whole_data = whole_data.drop(['wpgt','tsun'],axis=1).dropna()
whole_data.describe()


# In[15]:


whole_data.corr()['Close']


# In[16]:


whole_data.to_csv('wholedata.csv')


# In[17]:


whole_data


# ### Task2.2

# In[18]:


from api import *


# ## Task 3: Data Preprocessing

# ### Task 3.1 [5%] Check the data for missing values and outliers in the time series. Clean the data from missing values and outliers.

# In[19]:


whole_data.isnull().any()


# In[20]:


((whole_data-whole_data.mean())/whole_data.std()).boxplot(figsize=(12,5))
plt.savefig('boxplot.pdf',dpi=100)
plt.close()


# In[21]:


fig,ax=plt.subplots(figsize=(15,15))
im=ax.matshow(((whole_data-whole_data.mean())/whole_data.std()).corr())
plt.xticks(range(len(whole_data.columns)), whole_data.columns, fontsize=14, rotation=45)
plt.yticks(range(len(whole_data.columns)), whole_data.columns, fontsize=14)
fig.colorbar(im, orientation='vertical')
plt.savefig('correlationmatrix.pdf',dpi=100)
plt.close()


# In[22]:


((whole_data-whole_data.mean())/whole_data.std()).corr()['Close'].plot.bar()
plt.grid()
plt.savefig('correlationwithclose.pdf',dpi=100)
plt.close()


# ## Task 4: Data Exploration

# ### Task 4.1 Perform exploratory data analysis on your data

# In[23]:


whole_data['Adj Close'].plot()
plt.grid()
plt.savefig('historicalclose.pdf',dpi=100)
plt.close()


# In[24]:


moving_average = whole_data['Adj Close'].rolling(
    window=365,       # 365-day window
    center=True,      # puts the average at the center of the window
    min_periods=183,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)
ax = whole_data['Adj Close'].plot(style=".", color="0.5",label='Close')
moving_average.plot(
    ax=ax, linewidth=3, title="Tunnel Traffic - 365-Day Moving Average", legend=False,label='Trend');
plt.grid()
plt.legend()
plt.savefig('trend.pdf',dpi=100)
plt.close()


# In[25]:


def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("365D") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax


# In[26]:


plot_periodogram(whole_data['Close'])
plt.grid()
plt.grid()
plt.savefig('periodogram.pdf',dpi=100)
plt.close()


# ### Task 4.2 [5%] Calculate some of the known financial indicators for stock data. You can find a list of the most common indicators here. Then, plot and describe the indicators that you calculated. Use the indicators to derive insights about the stock trends, and identify the most significant days that it could be convenient to sell or buy stocks.

# In[27]:


# Define function to calculate the RSI

def calc_rsi(over: pd.Series, fn_roll) -> pd.Series:
    delta = over.diff()
    delta = delta[1:] 
    up, down = delta.clip(lower=0), delta.clip(upper=0).abs()
    roll_up, roll_down = fn_roll(up), fn_roll(down)
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:] = np.select([roll_down == 0, roll_up == 0, True], [100, 0, rsi])
    rsi.name = 'rsi'
    valid_rsi = rsi[length - 1:]
    assert ((0 <= valid_rsi) & (valid_rsi <= 100)).all()
    return rsi
# Calculate RSI using MA of choice
# Reminder: Provide ≥ `1 + length` extra data points!
length = 14
rsi_ema = calc_rsi(whole_data['Close'], lambda s: s.ewm(span=length).mean())
#rsi_sma = calc_rsi(whole_data['Close'], lambda s: s.rolling(length).mean())
#rsi_rma = calc_rsi(whole_data['Close'], lambda s: s.ewm(alpha=1 / length).mean())  # Approximates TradingView.
# Compare graphically
plt.figure(figsize=(8, 6))
rsi_ema.plot()#, rsi_sma.plot(), rsi_rma.plot()
plt.legend(['RSI via EMA/EWMA', 'RSI via SMA', 'RSI via RMA/SMMA/MMA (TradingView)'])
plt.grid()
plt.grid()
plt.savefig('rsi.pdf',dpi=100)
plt.close()


# In[28]:


#OBV
obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
((obv-obv.mean())/obv.std()).plot()
((df['Close']-df['Close'].mean())/df['Close'].std()).plot()
plt.grid()
plt.grid()
plt.savefig('obv.pdf',dpi=100)
plt.close()


# In[29]:


#MACD
# Calculate the 12-period EMA
EMA12 = whole_data['Close'].ewm(span=12, adjust=False).mean()
# Calculate the 26-period EMA
EMA26 = whole_data['Close'].ewm(span=26, adjust=False).mean()
# Calculate MACD (the difference between 12-period EMA and 26-period EMA)
MACD= EMA12 - EMA26


# In[30]:


plt.plot(MACD)
plt.grid()
plt.grid()
plt.savefig('MACD.pdf',dpi=100)
plt.close()


# In[31]:


test = df[['Ret']]
test.loc[:,'MACD'] = MACD
test.loc[:,'RSI'] = rsi_ema
test.loc[:,'obv'] = obv
test.loc[:,'Ret_next'] = test['Ret'].shift(-1) 
test.corr()


# ## Task 5: Forecasting

# In[32]:


from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score


# In[33]:


#On-balance volume (OBV)
#Accumulation/distribution (A/D) line
#Average directional index
#Aroon oscillator
#Moving average convergence divergence (MACD)
#Relative strength index (RSI)
#Stochastic oscillator


# In[34]:


#Develop ONE model that takes the previous n stock movements as inputs and outputs
#the closing price of the next day. Choose n appropriately, justify the choice, and use the model recursively
#to predict the true closing price for each day from the 1st of April 2023 to the 30th of April 2023 (the
#evaluation period). For example, if we assume n = 2, to predict the closing price of the 3rd of April
#the model takes the prices from the 2nd and the 1st of April. However, since these days are within the
#evaluation window, we must consider them not available, and we must forecast them as well. Evaluate the
#model on the true closing price of each day from 1st April 2023 to the 30th of April 2023.


# ### Task 5.1

# In[35]:


prices =  whole_data[['Close']]#web.DataReader('MSFT', start = datetime.date(2019,4,1), end = datetime.date(2023,4,30))[['Close']]
window = 2
for i in range(1,window+1):
    prices['Close_t-%s'%i] = prices['Close'].shift(i) 
prices = prices.dropna()
#plot_acf(prices['Close'])
plot_pacf(prices['Close'])
plt.grid()
plt.savefig('PACF.pdf',dpi=100)
plt.close()


# In[36]:


from sklearn.linear_model import  LinearRegression
test_period = pd.to_datetime(datetime.date(2023,3,31))
trainset = prices[prices.index<=test_period]
testset = prices[prices.index>test_period]
X_train, y_train = trainset.drop(['Close'],axis=1),trainset['Close']
X_test, y_test = testset.drop(['Close'],axis=1),testset['Close']
model=LinearRegression().fit(X_train, y_train )
trainset['Pred'] = model.predict(X_train)
testset['Pred'] = model.predict(X_test)


# In[37]:


trainset[['Pred','Close']].plot()
plt.grid()
plt.grid()
plt.savefig('CloseVSPred_model1_train.pdf',dpi=100)
plt.close()


# In[38]:


testset[['Close','Pred']].plot()
plt.grid()
plt.savefig('CloseVSPred_model1_test.pdf',dpi=100)
plt.close()


# In[39]:


mean_squared_error(trainset['Close'] ,trainset['Pred']),mean_absolute_error(trainset['Close'] ,trainset['Pred']), r2_score(trainset['Close'] ,trainset['Pred'])


# In[40]:


mean_squared_error(testset['Close'] ,testset['Pred']),mean_absolute_error(testset['Close'] ,testset['Pred']), r2_score(testset['Close'] ,testset['Pred'])


# In[41]:


plt.plot(testset['Close']-testset['Pred'])
plt.grid()
plt.savefig('residual_test_model1.pdf',dpi=100)
plt.close()


# ### Model 2

# In[42]:


aux_data = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd',
       'pres']
prices = whole_data[aux_data+['Close']]
prices


# In[43]:


window = 2
for col in aux_data+['Close']:
    for i in range(1,window+1):
        prices['%s_t-%s'%(col,i)] = prices[col].shift(i) 
prices = prices.dropna()
prices = prices.drop(aux_data,axis=1)
prices


# In[44]:


from sklearn.linear_model import  LinearRegression
test_period = pd.to_datetime(datetime.date(2023,3,31))
trainset = prices[prices.index<=test_period]
testset = prices[prices.index>test_period]
X_train, y_train = trainset.drop(['Close'],axis=1), trainset['Close']
X_test, y_test = testset.drop(['Close'],axis=1), testset['Close']
model = LinearRegression().fit(X_train, y_train)
trainset['Pred'] = model.predict(X_train)
testset['Pred'] = model.predict(X_test)


# In[45]:


testset[['Close','Pred']].plot()
plt.savefig('CloseVSPred_model2_train.pdf',dpi=100)
plt.close()


# In[46]:


testset[['Close','Pred']].plot()
plt.savefig('CloseVSPred_model2_test.pdf',dpi=100)
plt.close()


# In[47]:


mean_squared_error(trainset['Close'] ,trainset['Pred']),mean_absolute_error(trainset['Close'] ,trainset['Pred']), r2_score(trainset['Close'] ,trainset['Pred'])


# In[48]:


plt.plot(trainset['Close']-trainset['Pred'])
plt.grid()
plt.savefig('residual_train.pdf',dpi=100)
plt.close()


# In[49]:


mean_squared_error(testset['Close'] ,testset['Pred']),mean_absolute_error(testset['Close'] ,testset['Pred']), r2_score(testset['Close'] ,testset['Pred'])


# In[50]:


plt.plot(testset['Close']-testset['Pred'])
plt.grid()
plt.savefig('residual_test.pdf',dpi=100)
plt.close()


# ## Task 6: Decision-Making

# In[51]:


def agent(model,X):
    pred = model.predict(X)
    return pred[-1]-X['Close'].iloc[-1]

