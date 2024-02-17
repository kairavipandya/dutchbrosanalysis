#Systemwide AUV Trend and Yearly Revenue Growth Trend.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Recreating the DataFrame with the necessary data
data = {
    'Year': ['2020', '2021', '2022', '2023_9M'],
    'Total_Revenues': [327413, 497876, 739012, 711653],  # in thousands
    'Company_Operated_Shops_Revenue': [244514, 403746, 639710, 630588],  # in thousands
    'Franchising_and_Other_Revenue': [82899, 94130, 99302, 81065],  # in thousands
    'Total_Costs_and_Expenses': [316413, 609102, 741624, 667610],  # in thousands
    'Net_Income_Loss': [6058, -117931, -19253, 13721],  # in thousands
    'Total_Shop_Count_End_of_Period': [441, 538, 671, 794],
    'Systemwide_AUV': [1679, 1850, 1924, 1950],  # Placeholder for 2023 AUV as N/A
    'Company_Operated_Same_Shop_Sales_Growth': [0.8, 9.0, 0.6, 0.5],  # Percentage
}
df = pd.DataFrame(data)

# Calculate year-over-year growth for revenues
df['Revenue_Growth'] = df['Total_Revenues'].pct_change() * 100

# Setting seaborn style for better aesthetics
sns.set(style="whitegrid")

# Plotting Revenue Growth
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Revenue_Growth', data=df, marker='o', label='Revenue Growth (%)')
plt.title('Yearly Revenue Growth Trend')
plt.ylabel('Revenue Growth (%)')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Plotting AUV Trend
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Systemwide_AUV', data=df, marker='o', color='orange', label='Systemwide AUV')
plt.title('Systemwide Average Unit Volume (AUV) Trend')
plt.ylabel('AUV ($)')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Operational Efficiency Analysis

# Calculate Cost of Sales as a percentage of Total Revenues
df['Cost_of_Sales_Percentage'] = (df['Total_Costs_and_Expenses'] / df['Total_Revenues']) * 100

# SG&A is not explicitly separated in our dataset, let's assume Total Costs and Expenses include SG&A for this example
# Calculate SG&A as a percentage of Total Revenues
# This is a simplification; in real analysis, you'd separate SG&A from total costs if detailed data is available
df['SGA_Percentage'] = df['Cost_of_Sales_Percentage']  # Simplified assumption

# Calculate Net Income Margin
df['Net_Income_Margin'] = (df['Net_Income_Loss'] / df['Total_Revenues']) * 100

# Visualization
plt.figure(figsize=(14, 7))

# Plotting Cost of Sales Percentage
sns.lineplot(x='Year', y='Cost_of_Sales_Percentage', data=df, marker='o', label='Cost of Sales % of Total Revenues')

# Plotting SG&A Percentage
sns.lineplot(x='Year', y='SGA_Percentage', data=df, marker='o', label='SG&A % of Total Revenues', color='green')

# Plotting Net Income Margin
sns.lineplot(x='Year', y='Net_Income_Margin', data=df, marker='s', label='Net Income Margin %', color='red')

plt.title('Operational Efficiency Metrics Over Time')
plt.ylabel('Percentage')
plt.xlabel('Year')
plt.xticks(rotation=45)
plt.legend()
plt.show()
