import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.cluster import KMeans

# --- Data Preparation ---
# Define the data for Dutch Bros Inc. including revenues, shop count, AUV, and sales growth
data = {
    'Year': ['2020', '2021', '2022', '2023_9M'],
    'Total_Revenues': [327413, 497876, 739012, 711653],  # in thousands
    'Total_Shop_Count_End_of_Period': [441, 538, 671, 794],
    'Systemwide_AUV': [1679, 1850, 1924, 1950],  # Placeholder for 2023 AUV as N/A
    'Company_Operated_Same_Shop_Sales_Growth': [0.8, 9.0, 0.6, 0.5],  # Percentage
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Calculate year-over-year growth for revenues and shop count
df['Revenue_Growth'] = df['Total_Revenues'].pct_change() * 100
df['Shop_Count_Growth'] = df['Total_Shop_Count_End_of_Period'].pct_change() * 100

# --- Visualization ---
# Setting seaborn style for better aesthetics
sns.set(style="whitegrid")

# Plotting Total Revenues and Shop Count over the years
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Year')
ax1.set_ylabel('Total Revenues (in thousands)', color='tab:blue')
ax1.plot(df['Year'], df['Total_Revenues'], label='Total Revenues', color='tab:blue', marker='o')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Shop Count & AUV', color='tab:red')
ax2.plot(df['Year'], df['Total_Shop_Count_End_of_Period'], label='Shop Count', color='tab:red', linestyle='--', marker='x')
ax2.plot(df['Year'], df['Systemwide_AUV'], label='Systemwide AUV', color='tab:orange', linestyle='--', marker='x')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
plt.title('Dutch Bros Inc. Financial & Operational Overview')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.show()

# --- Time Series Forecasting for Revenue ---
# Assuming 'Total_Revenues' is in thousands and indexed by 'Year'
revenue_data = df.set_index('Year')['Total_Revenues']

# Model the time series with Exponential Smoothing
model = ExponentialSmoothing(revenue_data, trend="add", seasonal="add", seasonal_periods=4).fit()

# Forecast the next period
forecast = model.forecast(1)
print("Revenue Forecast for Next Period:", forecast)

# --- Clustering for Shop Performance ---
# Prepare data for clustering (using 'Systemwide_AUV' and 'Company_Operated_Same_Shop_Sales_Growth' as examples)
X = df[['Systemwide_AUV', 'Company_Operated_Same_Shop_Sales_Growth']].values

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# Add cluster labels to the original dataframe
df['Cluster'] = kmeans.labels_

# Display the dataframe with clusters
print(df[['Year', 'Systemwide_AUV', 'Company_Operated_Same_Shop_Sales_Growth', 'Cluster']])

# --- Additional Visualizations ---
# Plotting Revenue Growth Trend
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Revenue_Growth', data=df, marker='o', label='Revenue Growth (%)')
plt.title('Yearly Revenue Growth Trend')
plt.ylabel('Revenue Growth (%)')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Plotting Systemwide AUV Trend
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Systemwide_AUV', data=df, marker='o', color='orange', label='Systemwide AUV')
plt.title('Systemwide Average Unit Volume (AUV) Trend')
plt.ylabel('AUV ($)')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# Extend the DataFrame to include the years 2023 and 2024
df_extended = df.copy()
df_extended = df_extended.append({'Year': '2023', 'Revenue_Growth': None}, ignore_index=True)
df_extended = df_extended.append({'Year': '2024', 'Revenue_Growth': None}, ignore_index=True)

# Assuming 'Revenue_Growth' is in percentage and indexed by 'Year'
revenue_growth_data = df_extended.set_index('Year')['Revenue_Growth']

# Model the time series with Exponential Smoothing
model_extended = ExponentialSmoothing(revenue_growth_data, trend="add", seasonal="add", seasonal_periods=4).fit()

# Forecast the revenue growth for 2023 and 2024
forecast_2023 = model_extended.forecast(1)
forecast_2024 = model_extended.forecast(2)

# Update the DataFrame with forecasted revenue growth for 2023 and 2024
df_extended.loc[df_extended['Year'] == '2023', 'Revenue_Growth'] = forecast_2023.values[0]
df_extended.loc[df_extended['Year'] == '2024', 'Revenue_Growth'] = forecast_2024.values[0]

# Plotting Revenue Growth Trend including forecast
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Revenue_Growth', data=df_extended, marker='o', label='Revenue Growth (%)')
plt.title('Yearly Revenue Growth Trend including Forecast')
plt.ylabel('Revenue Growth (%)')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Print the forecasted revenue growth for 2023 and 2024
print("Forecasted Revenue Growth for 2023:", forecast_2023.values[0])
print("Forecasted Revenue Growth for 2024:", forecast_2024.values[0])


import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing

# Recreating the DataFrame with the necessary data
data = {
    'Year': ['2020', '2021', '2022', '2023_9M'],
    'Total_Revenues': [327413, 497876, 739012, 711653],  # in thousands
}

df = pd.DataFrame(data)

# Assuming the 2023_9M value is an incomplete value for 2023, let's extrapolate to get an estimated full year value
# Using a simple method: (711653 / 9) * 12 to estimate 2023 full year revenue
df.at[3, 'Total_Revenues'] = (df.at[3, 'Total_Revenues'] / 9) * 12

# Calculate year-over-year growth for revenues
df['Revenue_Growth'] = df['Total_Revenues'].pct_change() * 100

# Since we need to forecast for 2023 and 2024, let's correct the 'Year' for forecasting purposes
df['Year'] = [2020, 2021, 2022, 2023]  # Correcting year for forecasting

# Preparing the time series data
revenue_data = df.set_index('Year')['Revenue_Growth'].dropna()  # Dropping NaN values for modeling

# Model the time series with Exponential Smoothing
model = ExponentialSmoothing(revenue_data, trend="add", seasonal=None).fit()

# Forecast the revenue growth for the next 2 years (2024 and 2025, to get growth for 2023 and 2024)
forecast_years = [2024, 2025]
forecast = model.forecast(len(forecast_years))

# Prepare the forecasted growth for display
forecast_df = pd.DataFrame({'Year': forecast_years, 'Forecasted_Revenue_Growth': forecast.values})
forecast_df


import matplotlib.pyplot as plt
import seaborn as sns

# Integrating forecasted data with the original DataFrame for visualization
forecast_years = [2024, 2025]  # To align with the forecast years
forecast_growth = [19.297977, 7.464711]  # Forecasted growth rates

# Creating a new DataFrame to hold both actual and forecasted data
df_vis = pd.concat([df[['Year', 'Revenue_Growth']].dropna(), pd.DataFrame({'Year': forecast_years, 'Revenue_Growth': forecast_growth})])

# Setting seaborn style for better aesthetics
sns.set(style="whitegrid")

# Plotting Revenue Growth Trend
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Revenue_Growth', data=df_vis, marker='o', label='Revenue Growth (%)')
plt.axvline(x=2023, linestyle='--', color='gray', label='Forecast Start')  # Marking the start of forecasted data
plt.title('Yearly Revenue Growth Trend with Forecast')
plt.ylabel('Revenue Growth (%)')
plt.xlabel('Year')
plt.xticks(df_vis['Year'], rotation=45)
plt.legend()
plt.show()
# Addressing the issue and attempting a correct visualization approach
# Manually creating a combined DataFrame for actual and forecasted revenue growth to avoid previous errors

# Existing years and their revenue growth
years = [2020, 2021, 2022, 2023]
revenue_growth = df['Revenue_Growth'].tolist()

# Adding forecasted years and their predicted revenue growth
years += forecast_years
revenue_growth += forecast_growth

# Creating a new DataFrame for visualization
df_combined = pd.DataFrame({
    'Year': years,
    'Revenue Growth (%)': revenue_growth
})

# Plotting
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Revenue Growth (%)', data=df_combined, marker='o')
plt.title('Yearly Revenue Growth Trend with Forecast')
plt.xlabel('Year')
plt.ylabel('Revenue Growth (%)')
plt.axvline(x=2023, color='grey', linestyle='--', label='Forecast Start')
plt.legend()
plt.grid(True)
plt.show()
