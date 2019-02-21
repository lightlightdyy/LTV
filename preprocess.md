

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime, timedelta
```


```python
orders_df = pd.read_csv('data/passenger_orders_zhengzhou_2017_2018.tsv',sep='\t')
#orders_df
```


```python
#order_df_partial = orders_df.loc[:1000].copy()
order_df_partial = orders_df.copy()
```


```python
def get_call_and_finish_orders(x):
    x = x[2:]
    x = x[:-2]
    if(',' in x):
        tj = x.split('","')
        tj.sort(key = lambda x: x.split('|')[0])
        time_tj = ','.join([s.split('|')[0] for s in tj])
        call_tj = ','.join([s.split('|')[1] for s in tj])
        finish_tj = ','.join([s.split('|')[2] for s in tj])
    else:
        time_tj = x.split('|')[0]
        call_tj = x.split('|')[1]
        finish_tj = x.split('|')[2]
    return pd.Series({'time':time_tj, 'call_orders':call_tj, 'finish_orders':finish_tj})
```


```python
new_df = order_df_partial['_c1'].apply(get_call_and_finish_orders)

new_df['time']=new_df['time'].apply(lambda x: [datetime.strptime(str_time, '%Y-%m-%d') for str_time in x.split(',')])
new_df['call_orders']=new_df['call_orders'].apply(lambda x: [int(g) for g in x.split(',')])
new_df['finish_orders']=new_df['finish_orders'].apply(lambda x: [int(g) for g in x.split(',')])
```


```python
def datetime_range(start=None, end=None):
    span = end - start
    for i in range(span.days + 1):
        yield start + timedelta(days=i)


days = list(datetime_range(start=datetime(2017, 2, 15), end=datetime(2018, 2, 10)))
```


```python
new_df.shape
```


```python
row = new_df.loc[10]
```


```python
def get_one_data(row):
    days_count = len(days)
    times = row['time'] ; call_orders = row['call_orders'] ; finish_orders = row['finish_orders']
    call_vector = np.zeros(364) ; finish_vector = np.zeros(364)
    t_current = 0
    for d in range(days_count):
        if(days[d]==times[t_current]):
            call_vector[d] = call_orders[t_current]
            finish_vector[d] = finish_orders[t_current]
            t_current += 1
            if(t_current > len(times)-1):
                break
  
    call_weeks = np.reshape(call_vector,(52,-1))
    call_week_sum = np.sum(call_weeks,axis=1)
    
    finish_weeks = np.reshape(finish_vector,(52,-1))
    finish_week_sum = np.sum(finish_weeks,axis=1)
    
    return call_week_sum, finish_week_sum
```


```python
def get_result(row):
    c,f = get_one_data(row)
    r = np.std(f)/np.mean(f)
    return r
```


```python
result = []
for i in range(new_df.shape[0]):
    row = new_df.loc[i]
    r = get_result(row)
    if(r>0):
        result.append(r)
```


```python
plt.hist(result)
```




    (array([  32347.,  227635.,  395658.,  506731.,  428614.,  592839.,
             624463.,  299499.,   33880., 2443866.]),
     array([0.19006693, 0.88520308, 1.58033923, 2.27547538, 2.97061153,
            3.66574768, 4.36088383, 5.05601998, 5.75115613, 6.44629228,
            7.14142843]),
     <a list of 10 Patch objects>)




![png](output_11_1.png)



```python
len(result)
```




    5585532


