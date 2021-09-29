```python
import pandas as pd
import numpy as np
```


```python
df = pd.read_json('../data/raw/applerev.json')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>symbol</th>
      <th>reportedCurrency</th>
      <th>fillingDate</th>
      <th>acceptedDate</th>
      <th>period</th>
      <th>revenue</th>
      <th>costOfRevenue</th>
      <th>grossProfit</th>
      <th>grossProfitRatio</th>
      <th>...</th>
      <th>incomeBeforeTaxRatio</th>
      <th>incomeTaxExpense</th>
      <th>netIncome</th>
      <th>netIncomeRatio</th>
      <th>eps</th>
      <th>epsdiluted</th>
      <th>weightedAverageShsOut</th>
      <th>weightedAverageShsOutDil</th>
      <th>link</th>
      <th>finalLink</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-09-26</td>
      <td>AAPL</td>
      <td>USD</td>
      <td>2020-10-30</td>
      <td>2020-10-29 18:06:25</td>
      <td>FY</td>
      <td>274515000000</td>
      <td>169559000000</td>
      <td>104956000000</td>
      <td>0.382332</td>
      <td>...</td>
      <td>0.244398</td>
      <td>9680000000</td>
      <td>57411000000</td>
      <td>0.209136</td>
      <td>3.3100</td>
      <td>3.2800</td>
      <td>17352119000</td>
      <td>17528214000</td>
      <td>https://www.sec.gov/Archives/edgar/data/320193...</td>
      <td>https://www.sec.gov/Archives/edgar/data/320193...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-09-28</td>
      <td>AAPL</td>
      <td>USD</td>
      <td>2019-10-31 00:00:00</td>
      <td>2019-10-30 18:12:36</td>
      <td>FY</td>
      <td>260174000000</td>
      <td>161782000000</td>
      <td>98392000000</td>
      <td>0.378178</td>
      <td>...</td>
      <td>0.252666</td>
      <td>10481000000</td>
      <td>55256000000</td>
      <td>0.212381</td>
      <td>2.9925</td>
      <td>2.9725</td>
      <td>18471336000</td>
      <td>18595652000</td>
      <td>https://www.sec.gov/Archives/edgar/data/320193...</td>
      <td>https://www.sec.gov/Archives/edgar/data/320193...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-09-29</td>
      <td>AAPL</td>
      <td>USD</td>
      <td>2018-11-05 00:00:00</td>
      <td>2018-11-05 08:01:40</td>
      <td>FY</td>
      <td>265595000000</td>
      <td>163756000000</td>
      <td>101839000000</td>
      <td>0.383437</td>
      <td>...</td>
      <td>0.274489</td>
      <td>13372000000</td>
      <td>59531000000</td>
      <td>0.224142</td>
      <td>3.0025</td>
      <td>2.9775</td>
      <td>19821508000</td>
      <td>20000436000</td>
      <td>https://www.sec.gov/Archives/edgar/data/320193...</td>
      <td>https://www.sec.gov/Archives/edgar/data/320193...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-09-30</td>
      <td>AAPL</td>
      <td>USD</td>
      <td>2017-11-03 00:00:00</td>
      <td>2017-11-03 08:01:37</td>
      <td>FY</td>
      <td>229234000000</td>
      <td>141048000000</td>
      <td>88186000000</td>
      <td>0.384699</td>
      <td>...</td>
      <td>0.279579</td>
      <td>15738000000</td>
      <td>48351000000</td>
      <td>0.210924</td>
      <td>2.3175</td>
      <td>2.3025</td>
      <td>20868968000</td>
      <td>21006768000</td>
      <td>https://www.sec.gov/Archives/edgar/data/320193...</td>
      <td>https://www.sec.gov/Archives/edgar/data/320193...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-09-24</td>
      <td>AAPL</td>
      <td>USD</td>
      <td>2016-10-26 00:00:00</td>
      <td>2016-10-26 16:42:16</td>
      <td>FY</td>
      <td>215639000000</td>
      <td>131376000000</td>
      <td>84263000000</td>
      <td>0.390760</td>
      <td>...</td>
      <td>0.284605</td>
      <td>15685000000</td>
      <td>45687000000</td>
      <td>0.211868</td>
      <td>2.0875</td>
      <td>2.0775</td>
      <td>21883280000</td>
      <td>22001124000</td>
      <td>https://www.sec.gov/Archives/edgar/data/320193...</td>
      <td>https://www.sec.gov/Archives/edgar/data/320193...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>



# Clean Data


```python
df = df.drop(columns=['reportedCurrency', 
                      'fillingDate', 
                      'acceptedDate', 
                      'period', 
                      'link', 
                      'finalLink', 
                      'symbol', 
                      'grossProfitRatio', 
                      'incomeBeforeTaxRatio', 
                      'netIncomeRatio', 
                      'eps', 
                      'epsdiluted'])

```


```python
df = df.sort_values("date")
```


```python
df.to_csv('../data/interim/AppleInterimRevenue.csv', index = False)
```

# Data analysis


```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
```


```python
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 10)
```

Metrics that I need to analyze.

#### First 
* Revenue
* CostOfRevenue
* grossProfit

#### Second - Expenses
* researchAndDevelopmentExpenses 	
* generalAndAdministrativeExpenses 	
* sellingAndMarketingExpenses 	
* sellingGeneralAndAdministrativeExpenses 	
* otherExpenses 	
* operatingExpenses

#### Third - Earnings before
* ebitda
* operatingIncome
* netIncome


## Creating new frames

#### Revenue Frame


```python
revdf = df[["revenue","costOfRevenue", "grossProfit"]] / 1000000000
```




    1000000000




```python
revdf.to_csv('../data/interim/RevAppleDf.csv', index = False)
```


```python
revdf = pd.read_csv('../data/interim/RevAppleDf.csv')
```


```python
revdf.plot(kind='line');
```


    
![png](output_15_0.png)
    



```python
revdf.plot(subplots=True);
```


    
![png](output_16_0.png)
    


#### Expense Frame


```python
expdf = df[[
    "researchAndDevelopmentExpenses",
    "generalAndAdministrativeExpenses",
    "sellingAndMarketingExpenses",
    "sellingGeneralAndAdministrativeExpenses",
    "otherExpenses",
    "operatingExpenses"
]] / 1000000000
```


```python
expdf.to_csv('../data/interim/ExpAppleDf.csv', index = False)
```


```python
expdf = pd.read_csv('../data/interim/ExpAppleDf.csv')
```


```python
expdf.plot(kind='line');
```


    
![png](output_21_0.png)
    



```python
expdf.plot(subplots=True);
```


    
![png](output_22_0.png)
    


#### Earnings frame


```python
eardf = df[[
    "ebitda",
    "operatingIncome",
    "netIncome"]] / 1000000000
```


```python
eardf.to_csv('../data/interim/EarAppleDf.csv', index = False)
```


```python
eardf = pd.read_csv('../data/interim/EarAppleDf.csv')
```


```python
eardf.plot(kind='line');
```


    
![png](output_27_0.png)
    



```python
eardf.plot(subplots=True);
```


    
![png](output_28_0.png)
    


## Linear Regression


```python
from sklearn import linear_model
```


```python
appledf = pd.read_csv('../data/processed/DataProcessed.csv')
```


```python
appledf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>revenue</th>
      <th>costOfRevenue</th>
      <th>grossProfit</th>
      <th>researchAndDevelopmentExpenses</th>
      <th>generalAndAdministrativeExpenses</th>
      <th>sellingAndMarketingExpenses</th>
      <th>sellingGeneralAndAdministrativeExpenses</th>
      <th>otherExpenses</th>
      <th>operatingExpenses</th>
      <th>ebitda</th>
      <th>operatingIncome</th>
      <th>netIncome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.9183</td>
      <td>1.0760</td>
      <td>0.8423</td>
      <td>0.0</td>
      <td>0.6532</td>
      <td>0</td>
      <td>0.6950</td>
      <td>0.0</td>
      <td>0.6950</td>
      <td>0.1891</td>
      <td>0.1473</td>
      <td>0.0612</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.9019</td>
      <td>0.8400</td>
      <td>1.0619</td>
      <td>0.0</td>
      <td>0.7373</td>
      <td>0</td>
      <td>0.7884</td>
      <td>0.0</td>
      <td>0.7884</td>
      <td>0.3246</td>
      <td>0.2735</td>
      <td>0.1540</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.6611</td>
      <td>1.2257</td>
      <td>1.4354</td>
      <td>0.0</td>
      <td>0.9934</td>
      <td>0</td>
      <td>1.0639</td>
      <td>0.0</td>
      <td>1.0639</td>
      <td>0.4420</td>
      <td>0.3715</td>
      <td>0.2175</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0714</td>
      <td>1.9132</td>
      <td>2.1582</td>
      <td>0.0</td>
      <td>1.4602</td>
      <td>0</td>
      <td>1.5379</td>
      <td>0.0</td>
      <td>1.5379</td>
      <td>0.6980</td>
      <td>0.6203</td>
      <td>0.4003</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.2840</td>
      <td>2.5700</td>
      <td>2.7140</td>
      <td>0.0</td>
      <td>1.9549</td>
      <td>0</td>
      <td>2.0797</td>
      <td>0.0</td>
      <td>2.0797</td>
      <td>0.7591</td>
      <td>0.6343</td>
      <td>0.4540</td>
    </tr>
  </tbody>
</table>
</div>




```python
appledf.plot(kind='line');
```


    
![png](output_33_0.png)
    



```python
appledf.plot(kind = 'scatter', x='revenue', y='costOfRevenue')
plt.show()
```


    
![png](output_34_0.png)
    



```python
appledf.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>revenue</th>
      <th>costOfRevenue</th>
      <th>grossProfit</th>
      <th>researchAndDevelopmentExpenses</th>
      <th>generalAndAdministrativeExpenses</th>
      <th>sellingAndMarketingExpenses</th>
      <th>sellingGeneralAndAdministrativeExpenses</th>
      <th>otherExpenses</th>
      <th>operatingExpenses</th>
      <th>ebitda</th>
      <th>operatingIncome</th>
      <th>netIncome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>revenue</th>
      <td>1.000000</td>
      <td>0.999496</td>
      <td>0.998771</td>
      <td>0.939552</td>
      <td>0.993723</td>
      <td>NaN</td>
      <td>0.993040</td>
      <td>-0.536602</td>
      <td>0.979462</td>
      <td>0.994519</td>
      <td>0.991342</td>
      <td>0.996547</td>
    </tr>
    <tr>
      <th>costOfRevenue</th>
      <td>0.999496</td>
      <td>1.000000</td>
      <td>0.996693</td>
      <td>0.945411</td>
      <td>0.994014</td>
      <td>NaN</td>
      <td>0.993240</td>
      <td>-0.523494</td>
      <td>0.982310</td>
      <td>0.991379</td>
      <td>0.987306</td>
      <td>0.993701</td>
    </tr>
    <tr>
      <th>grossProfit</th>
      <td>0.998771</td>
      <td>0.996693</td>
      <td>1.000000</td>
      <td>0.928510</td>
      <td>0.991264</td>
      <td>NaN</td>
      <td>0.990724</td>
      <td>-0.555987</td>
      <td>0.973040</td>
      <td>0.997418</td>
      <td>0.995645</td>
      <td>0.998982</td>
    </tr>
    <tr>
      <th>researchAndDevelopmentExpenses</th>
      <td>0.939552</td>
      <td>0.945411</td>
      <td>0.928510</td>
      <td>1.000000</td>
      <td>0.954967</td>
      <td>NaN</td>
      <td>0.954492</td>
      <td>-0.327682</td>
      <td>0.986751</td>
      <td>0.902942</td>
      <td>0.890807</td>
      <td>0.917839</td>
    </tr>
    <tr>
      <th>generalAndAdministrativeExpenses</th>
      <td>0.993723</td>
      <td>0.994014</td>
      <td>0.991264</td>
      <td>0.954967</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.999941</td>
      <td>-0.487263</td>
      <td>0.990423</td>
      <td>0.980859</td>
      <td>0.976486</td>
      <td>0.985974</td>
    </tr>
    <tr>
      <th>sellingAndMarketingExpenses</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sellingGeneralAndAdministrativeExpenses</th>
      <td>0.993040</td>
      <td>0.993240</td>
      <td>0.990724</td>
      <td>0.954492</td>
      <td>0.999941</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>-0.486459</td>
      <td>0.990233</td>
      <td>0.980240</td>
      <td>0.975812</td>
      <td>0.985350</td>
    </tr>
    <tr>
      <th>otherExpenses</th>
      <td>-0.536602</td>
      <td>-0.523494</td>
      <td>-0.555987</td>
      <td>-0.327682</td>
      <td>-0.487263</td>
      <td>NaN</td>
      <td>-0.486459</td>
      <td>1.000000</td>
      <td>-0.417830</td>
      <td>-0.595038</td>
      <td>-0.603350</td>
      <td>-0.559701</td>
    </tr>
    <tr>
      <th>operatingExpenses</th>
      <td>0.979462</td>
      <td>0.982310</td>
      <td>0.973040</td>
      <td>0.986751</td>
      <td>0.990423</td>
      <td>NaN</td>
      <td>0.990233</td>
      <td>-0.417830</td>
      <td>1.000000</td>
      <td>0.955384</td>
      <td>0.947302</td>
      <td>0.965128</td>
    </tr>
    <tr>
      <th>ebitda</th>
      <td>0.994519</td>
      <td>0.991379</td>
      <td>0.997418</td>
      <td>0.902942</td>
      <td>0.980859</td>
      <td>NaN</td>
      <td>0.980240</td>
      <td>-0.595038</td>
      <td>0.955384</td>
      <td>1.000000</td>
      <td>0.999195</td>
      <td>0.997926</td>
    </tr>
    <tr>
      <th>operatingIncome</th>
      <td>0.991342</td>
      <td>0.987306</td>
      <td>0.995645</td>
      <td>0.890807</td>
      <td>0.976486</td>
      <td>NaN</td>
      <td>0.975812</td>
      <td>-0.603350</td>
      <td>0.947302</td>
      <td>0.999195</td>
      <td>1.000000</td>
      <td>0.997429</td>
    </tr>
    <tr>
      <th>netIncome</th>
      <td>0.996547</td>
      <td>0.993701</td>
      <td>0.998982</td>
      <td>0.917839</td>
      <td>0.985974</td>
      <td>NaN</td>
      <td>0.985350</td>
      <td>-0.559701</td>
      <td>0.965128</td>
      <td>0.997926</td>
      <td>0.997429</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
x=appledf.drop(['revenue'], axis=1).values
y=appledf['revenue'].values
```


```python
from sklearn.model_selection import train_test_split
```


```python
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=0)
```


```python
from sklearn.linear_model import LinearRegression
ml=LinearRegression()
ml.fit(x_train,y_train)
```




    LinearRegression()




```python
y_pred=ml.predict(x_test)
print(y_pred)
```

    [215.639   13.931    5.363  233.715   24.006    7.983   11.062    2.6611
       9.833  182.795  156.508 ]
    


```python
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
```




    1.0




```python
plt.figure(figsize=(15, 10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
```




    Text(0.5, 1.0, 'Actual vs Predicted')




    
![png](output_42_1.png)
    



```python
pred_y_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred, 'Difference': y_test-y_pred})
pred_y_df[0:20]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual Value</th>
      <th>Predicted Value</th>
      <th>Difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>215.6390</td>
      <td>215.6390</td>
      <td>-5.684342e-14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13.9310</td>
      <td>13.9310</td>
      <td>3.552714e-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.3630</td>
      <td>5.3630</td>
      <td>-3.552714e-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>233.7150</td>
      <td>233.7150</td>
      <td>-1.421085e-13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.0060</td>
      <td>24.0060</td>
      <td>7.105427e-15</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7.9830</td>
      <td>7.9830</td>
      <td>-6.217249e-15</td>
    </tr>
    <tr>
      <th>6</th>
      <td>11.0620</td>
      <td>11.0620</td>
      <td>7.105427e-15</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.6611</td>
      <td>2.6611</td>
      <td>-4.440892e-15</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.8330</td>
      <td>9.8330</td>
      <td>1.065814e-14</td>
    </tr>
    <tr>
      <th>9</th>
      <td>182.7950</td>
      <td>182.7950</td>
      <td>-2.842171e-14</td>
    </tr>
    <tr>
      <th>10</th>
      <td>156.5080</td>
      <td>156.5080</td>
      <td>-5.684342e-14</td>
    </tr>
  </tbody>
</table>
</div>


