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
