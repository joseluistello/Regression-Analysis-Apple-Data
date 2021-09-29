import pandas as pd

### Revenue Frame Visualization 

revdf.plot(kind='line');

revdf.plot(subplots=True);

### Expense Frame Visualization

expdf.plot(kind='line');

expdf.plot(subplots=True);


### Earning Frame Visualization


eardf.plot(kind='line');

eardf.plot(subplots=True);


### Apple Frame Visualization

appledf.plot(kind='line');


appledf.plot(kind = 'scatter', x='revenue', y='costOfRevenue')
plt.show()


### Model Visualization


plt.figure(figsize=(15, 10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')