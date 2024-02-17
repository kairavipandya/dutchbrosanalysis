import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.cluster import KMeans

# Data Preparation
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

# Visualization
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

# Time Series Forecasting for Revenue
revenue_data = df.set_index('Year')['Total_Revenues']
model = ExponentialSmoothing(revenue_data, trend="add", seasonal="add", seasonal_periods=4).fit()
forecast = model.forecast(1)

# Clustering for Shop Performance
X = df[['Systemwide_AUV', 'Company_Operated_Same_Shop_Sales_Growth']].values
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
df['Cluster'] = kmeans.labels_

# Additional Visualizations
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Revenue_Growth', data=df, marker='o', label='Revenue Growth (%)')
plt.title('Yearly Revenue Growth Trend')
plt.ylabel('Revenue Growth (%)')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Systemwide_AUV', data=df, marker='o', color='orange', label='Systemwide AUV')
plt.title('Systemwide Average Unit Volume (AUV) Trend')
plt.ylabel('AUV ($)')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Extend the DataFrame to include the years 2023 and 2024 for forecasted growth
df_extended = df.copy()
df_extended = df_extended.append({'Year': '2023', 'Revenue_Growth': None}, ignore_index=True)
df_extended = df_extended.append({'Year': '2024', 'Revenue_Growth': None}, ignore_index=True)
revenue_growth_data = df_extended.set_index('Year')['Revenue_Growth']
model_extended = ExponentialSmoothing(revenue_growth_data, trend="add", seasonal="add", seasonal_periods=4).fit()
forecast_2023 = model_extended.forecast(1)
forecast_2024 = model_extended.forecast(2)
df_extended.loc[df_extended['Year'] == '2023', 'Revenue_Growth'] = forecast_2023.values[0]
df_extended.loc[df_extended['Year'] == '2024', 'Revenue_Growth'] = forecast_2024.values[0]
# Define forecast years and growth
forecast_years = [2023, 2024]  # Add the years you want to forecast
forecast_growth = [forecast_2023.values[0], forecast_2024.values[0]]  # Add the forecasted growth for each year

# Systemwide AUV Trend and Yearly Revenue Growth Trend
plt.figure(figsize=(12, 6))
years = [2020, 2021, 2022, 2023]
revenue_growth = df['Revenue_Growth'].tolist()
years += forecast_years  # Use the defined forecast years
revenue_growth += forecast_growth  # Use the defined forecasted growth
df_combined = pd.DataFrame({'Year': years, 'Revenue Growth (%)': revenue_growth})
sns.lineplot(x='Year', y='Revenue Growth (%)', data=df_combined, marker='o')
plt.title('Yearly Revenue Growth Trend with Forecast')
plt.xlabel('Year')
plt.ylabel('Revenue Growth (%)')
plt.axvline(x=2023, color='grey', linestyle='--', label='Forecast Start')
plt.legend()
plt.grid(True)
plt.show()
print("Forecasted Revenue Growth for 2023:", forecast_2023.values[0])
print("Forecasted Revenue Growth for 2024:", forecast_2024.values[0])

# Systemwide AUV Trend and Yearly Revenue Growth Trend
plt.figure(figsize=(12, 6))
years = [2020, 2021, 2022, 2023]
revenue_growth = df['Revenue_Growth'].tolist()
years += forecast_years
revenue_growth += forecast_growth
df_combined = pd.DataFrame({'Year': years, 'Revenue Growth (%)': revenue_growth})
sns.lineplot(x='Year', y='Revenue Growth (%)', data=df_combined, marker='o')
plt.title('Yearly Revenue Growth Trend with Forecast')
plt.xlabel('Year')
plt.ylabel('Revenue Growth (%)')
plt.axvline(x=2023, color='grey', linestyle='--', label='Forecast Start')
plt.legend()
plt.grid(True)
plt.show()

# Operational Efficiency Analysis
df['Cost_of_Sales_Percentage'] = (df['Total_Costs_and_Expenses'] / df['Total_Revenues']) * 100
df['SGA_Percentage'] = df['Cost_of_Sales_Percentage']
df['Net_Income_Margin'] = (df['Net_Income_Loss'] / df['Total_Revenues']) * 100
plt.figure(figsize=(14, 7))
sns.lineplot(x='Year', y='Cost_of_Sales_Percentage', data=df, marker='o', label='Cost of Sales % of Total Revenues')
sns.lineplot(x='Year', y='SGA_Percentage', data=df, marker='o', label='SG&A % of Total Revenues', color='green')
sns.lineplot(x='Year', y='Net_Income_Margin', data=df, marker='s', label='Net Income Margin %', color='red')
plt.title('Operational Efficiency Metrics Over Time')
plt.ylabel('Percentage')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.legend()
plt.show()
