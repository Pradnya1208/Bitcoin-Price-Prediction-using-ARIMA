<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Bitcoin price prediction using ARIMA</div>
<div align="center"><img src="https://github.com/Pradnya1208/Bitcoin-Price-Prediction-using-ARIMA/blob/main/output/intro.gif?raw=true"></div>


## Overview:
Bitcoin is the longest running and most well known cryptocurrency, first released as open source in 2009 by the anonymous Satoshi Nakamoto. Bitcoin serves as a decentralized medium of digital exchange, with transactions verified and recorded in a public distributed ledger (the blockchain) without the need for a trusted record keeping authority or central intermediary. Transaction blocks contain a SHA-256 cryptographic hash of previous transaction blocks, and are thus "chained" together, serving as an immutable record of all transactions that have ever occurred. As with any currency/commodity on the market, bitcoin trading and financial instruments soon followed public adoption of bitcoin and continue to grow. Included here is historical bitcoin market data at 1-min intervals for select bitcoin exchanges where trading takes place.
### Time series analysis:
Time Series is a series of observations taken at specified time intervals usually equal intervals. Analysis of the series helps us to predict future values based on previous observed values. In Time series, we have only 2 variables, time & the variable we want to forecast.

### Components of Time Series:
There are 4 components:<br>
- **Trend** - Upward & downward movement of the data with time over a large period of time. Eq: Appreciation of Dollar vs rupee.
- **Seasonality** - seasonal variances. Eq: Icecream sales increases in Summer only
- **Noise or Irregularity** - Spikes & troughs at random intervals
- **Cyclicity** - behavior that repeats itself after large interval of time, like months, years etc.

### ARIMA Model:
ARIMA(Auto Regressive Integrated Moving Average) is a combination of 2 models AR(Auto Regressive) & MA(Moving Average). It has 3 hyperparameters - P(auto regressive lags),d(order of differentiation),Q(moving avg.) which respectively comes from the AR, I & MA components. The AR part is correlation between prev & current time periods. To smooth out the noise, the MA part is used. The I part binds together the AR & MA parts.
<br>
In order to find the values of P and Q, We need to take help of ACF(Auto Correlation Function) & PACF(Partial Auto Correlation Function) plots. ACF & PACF graphs are used to find value of P & Q for ARIMA. We need to check, for which value in x-axis, graph line drops to 0 in y-axis for 1st time.
From PACF(at y=0), get P
From ACF(at y=0), get Q
## Dataset:
[Bitcoin Historical Data](https://www.kaggle.com/mczielinski/bitcoin-historical-data)

CSV files for select bitcoin exchanges for the time period of Jan 2012 to December March 2021, with minute to minute updates of OHLC (Open, High, Low, Close), Volume in BTC and indicated currency, and weighted bitcoin price. Timestamps are in Unix time. Timestamps without any trades or activity have their data fields filled with NaNs. If a timestamp is missing, or if there are jumps, this may be because the exchange (or its API) was down, the exchange (or its API) did not exist, or some other unforeseen technical error in data reporting or gathering.
## Implementation:

**Libraries:**  `NumPy` `pandas` `datetime` `matplotlib` `sklearn` `seaborn` `statsmodels`
## Data Exploration:
<img src="https://github.com/Pradnya1208/Bitcoin-Price-Prediction-using-ARIMA/blob/main/output/eda1.PNG?raw=true">

#### Stationarity:
Time Series(TS) need to be stationary because,
- If a TS has a particular behavior over a time interval, then there's a high probability that over a different interval, it will have same behavior, provided TS is stationary. This helps in forecasting accurately.
- Theories & Mathematical formulas ae more mature & easier to apply for as TS which is stationary.

Before applying any statistical model on a Time Series, the series has to be staionary, which means that, over different time periods,
- It should have constant mean.
- It should have constant variance or standard deviation.
- Auto-covariance should not depend on time.

Trend & Seasonality are two reasons why a Time Series is not stationaru & hence need to be corrected.
<br>
There are 2 ways to check for Stationarity of a TS:<br>
- **Rolling Statistics** - Plot the moving avg or moving standard deviation to see if it varies with time. Its a visual technique.
- **ADCF Test** - Augmented Dickey–Fuller test is used to gives us various values that can help in identifying stationarity. The Null hypothesis says that a TS is non-stationary. It comprises of a Test Statistics & some critical values for some confidence levels. If the Test statistics is less than the critical values, we can reject the null hypothesis & say that the series is stationary. THE ADCF test also gives us a p-value. Acc to the null hypothesis, lower values of p is better.

#### Stationarity check and seasonal decomposition:
We will try to decompose seasonal component of the time series.
```
plt.figure(figsize=[15,7])
sm.tsa.seasonal_decompose(df_month.Weighted_Price).plot()
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df_month.Weighted_Price)[1])
```
<img src="https://github.com/Pradnya1208/Bitcoin-Price-Prediction-using-ARIMA/blob/main/output/eda2.PNG?raw=true">
<br>
The series are not stationary.

#### Regular differentiation:
<img src="https://github.com/Pradnya1208/Bitcoin-Price-Prediction-using-ARIMA/blob/main/output/diff.PNG?raw=true">
<br>
The series are stationary.

### Model Selection
<img src="https://github.com/Pradnya1208/Bitcoin-Price-Prediction-using-ARIMA/blob/main/output/model.PNG?raw=true">
<br>

```
parameters         aic
21  (1, 0, 2, 0)  220.331987
20  (1, 0, 1, 1)  220.386077
18  (1, 0, 0, 1)  220.741648
19  (1, 0, 1, 0)  221.262547
12  (0, 2, 0, 1)  221.834795
                                 Statespace Model Results                                 
==========================================================================================
Dep. Variable:                 Weighted_Price_box   No. Observations:                   66
Model:             SARIMAX(1, 1, 0)x(2, 1, 0, 12)   Log Likelihood                -106.166
Date:                            Tue, 14 Nov 2017   AIC                            220.332
Time:                                    10:17:07   BIC                            229.091
Sample:                                12-31-2011   HQIC                           223.793
                                     - 05-31-2017                                         
Covariance Type:                              opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ar.L1          0.5049   1.02e+10   4.93e-11      1.000   -2.01e+10    2.01e+10
ar.S.L12      -0.8275   2.84e+10  -2.91e-11      1.000   -5.57e+10    5.57e+10
ar.S.L24      -0.2863   3.03e+10  -9.45e-12      1.000   -5.94e+10    5.94e+10
sigma2         2.7269   9.45e+10   2.88e-11      1.000   -1.85e+11    1.85e+11
===================================================================================
Ljung-Box (Q):                       31.99   Jarque-Bera (JB):                12.38
Prob(Q):                              0.81   Prob(JB):                         0.00
Heteroskedasticity (H):               0.49   Skew:                             0.74
Prob(H) (two-sided):                  0.14   Kurtosis:                         4.84
===================================================================================
```
### Analysis of residues
<img src="https://github.com/Pradnya1208/Bitcoin-Price-Prediction-using-ARIMA/blob/main/output/resi.PNG?raw=true">

### Predictions:
<img src = "https://github.com/Pradnya1208/Bitcoin-Price-Prediction-using-ARIMA/blob/main/output/predict.PNG?raw=true">












### Lessons Learned
`Time Series`
`ARIMA model`
`Differentiation and Stationarity`








## References:
[Time series](https://www.kaggle.com/thebrownviking20/everything-you-can-do-with-a-time-series)
### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner



















[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

