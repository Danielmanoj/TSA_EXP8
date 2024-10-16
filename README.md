# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 
### Developed by : MANOJ G
### Register no : 212222240060

### AIM:
To implement Moving Average Model and Exponential smoothing Using Astrobiological dataset.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the CSV file
df = pd.read_csv('/content/astrobiological_activity_monitoring.csv')

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Display shape and first 20 rows (or the available data if fewer rows)
print("Dataset shape:", df.shape)
print("First rows of dataset:\n", df.head(20))

# Plot the original data (Soil_Microbial_Activity)
plt.figure(figsize=(10, 6))
plt.plot(df['Soil_Microbial_Activity'], label='Original Data', marker='o')
plt.title('Original Time Series Data (Soil Microbial Activity)')
plt.ylabel('Soil Microbial Activity')
plt.xlabel('Date')
plt.legend()
plt.show()

# Moving Average with window size 5 and 10
rolling_mean_5 = df['Soil_Microbial_Activity'].rolling(window=5).mean()
rolling_mean_10 = df['Soil_Microbial_Activity'].rolling(window=10).mean()

# Plot original data and rolling means (5 and 10)
plt.figure(figsize=(10, 6))
plt.plot(df['Soil_Microbial_Activity'], label='Original Data', marker='o')
plt.plot(rolling_mean_5, label='Rolling Mean (Window=5)', marker='x')
plt.plot(rolling_mean_10, label='Rolling Mean (Window=10)', marker='^')
plt.title('Original Data vs Rolling Means')
plt.ylabel('Soil Microbial Activity')
plt.xlabel('Date')
plt.legend()
plt.show()

# Perform Exponential Smoothing
exp_smoothing = SimpleExpSmoothing(df['Soil_Microbial_Activity']).fit(smoothing_level=0.2, optimized=False)
exp_smoothed = exp_smoothing.fittedvalues

# Plot Original Data and Exponential Smoothing
plt.figure(figsize=(10, 6))
plt.plot(df['Soil_Microbial_Activity'], label='Original Data', marker='o')
plt.plot(exp_smoothed, label='Exponential Smoothing', marker='s')
plt.title('Original Data vs Exponential Smoothing')
plt.ylabel('Soil Microbial Activity')
plt.xlabel('Date')
plt.legend()
plt.show()

# Plot ACF and PACF
plt.figure(figsize=(10, 6))
plt.subplot(121)
plot_acf(df['Soil_Microbial_Activity'], lags=10, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.subplot(122)
plot_pacf(df['Soil_Microbial_Activity'], lags=10, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# Generate Predictions using Exponential Smoothing (Predict next 3 values)
prediction_steps = 3
forecast = exp_smoothing.forecast(steps=prediction_steps)

# Plot original data and predictions
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Soil_Microbial_Activity'], label='Original Data', marker='o')
plt.plot(pd.date_range(start=df.index[-1], periods=prediction_steps + 1, freq='D')[1:], forecast, label='Predictions', marker='x')
plt.title('Original Data vs Predictions (Exponential Smoothing)')
plt.ylabel('Soil Microbial Activity')
plt.xlabel('Date')
plt.legend()
plt.show()

```

### OUTPUT:

![image](https://github.com/user-attachments/assets/8fe957c6-0409-49a3-bd5a-54ebf8bc1d38)

![image](https://github.com/user-attachments/assets/e19f9aef-8b7c-49ea-886e-cee8df8fb7f9)

![image](https://github.com/user-attachments/assets/02c32c42-564d-49ee-bb46-514bd3d778e8)

![image](https://github.com/user-attachments/assets/00ee10a7-fe47-4f0e-9498-1d6201fa4da9)


![image](https://github.com/user-attachments/assets/7fbc9b62-77ad-4214-a0bb-9abba2f8abdd)



### RESULT:
Thus the python code successfully implemented for the Moving Average Model and Exponential smoothing using astrobiological dataset.
