from analysis_00s import best_00s_features, best_00s_features_df, data_00s
from analysis_10s import best_10s_features, best_10s_features_df, data_10s
from analysis_90s import best_90s_features, best_90s_features_df, data_90s
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Test print statements to verify data loading and feature extraction
print("Best features from 90s: \n")
print(best_90s_features)
print("Best features from 00s: \n")
print(best_00s_features)
print("Best features from 10s: \n")
print(best_10s_features)
print("Combined DataFrame:")
#print(combined_df.head())

# Get the top 6 features for each decade and combine them to get unique features across all decades
top6_90s = best_90s_features.index[1:7]
top6_00s = best_00s_features.index[1:7]
top6_10s = best_10s_features.index[1:7]
print("Top 6 features for 90s:", top6_90s)
print("Top 6 features for 00s:", top6_00s)
print("Top 6 features for 10s:", top6_10s)
joined_top_features = list(set(top6_90s).union(set(top6_00s)).union(set(top6_10s))) #get the top unique features across all decades
print("Joined top features across all decades:", joined_top_features)

# Create combined DataFrame for the joined top features across all decades
joined_features_00s = data_00s[joined_top_features]
joined_features_10s = data_10s[joined_top_features]
joined_features_90s = data_90s[joined_top_features]

combined_df = pd.concat([joined_features_90s, joined_features_00s, joined_features_10s], ignore_index=True)
combined_df['decade'] = ['90s'] * len(joined_features_90s) + ['00s'] * len(joined_features_00s) + ['10s'] * len(joined_features_10s)

"""
# Outlier detection and removal for 'time_signature' feature using IQR method
Q1 = combined_df["time_signature"].quantile(0.25)
Q3 = combined_df["time_signature"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = combined_df[
    (combined_df["time_signature"] < lower) |
    (combined_df["time_signature"] > upper)
]

percent = len(outliers) / len(combined_df) * 100

print(f"Outliery: {percent:.2f}% danych")


df_clean = combined_df[
    (combined_df["time_signature"] >= lower) &
    (combined_df["time_signature"] <= upper)
]
"""


fig, axes = plt.subplots(2, 4, figsize=(15, 10))
axes = axes.flatten()
#test = combined_df[combined_df['target'] == 1]
for idx, feature in enumerate(joined_top_features):
    sns.boxplot(x='decade', y=feature, data=combined_df, ax=axes[idx])
    axes[idx].set_title(f'Box Plot of {feature} by Decade')
plt.tight_layout()
#plt.savefig('decade_comparison_box_plots_fulldata.png')
plt.show()

# Prepare bump chart for the top features ranking across decades
bump_data = pd.DataFrame({
    'feature': joined_top_features,
    '90s': [best_90s_features[feature] for feature in joined_top_features],
    '00s': [best_00s_features[feature] for feature in joined_top_features],
    '10s': [best_10s_features[feature] for feature in joined_top_features]
})
bump_data = bump_data.set_index('feature')
bump_data = bump_data.rank(ascending=False)
# Plot the bump chart
plt.figure(figsize=(10, 6))
for feature in joined_top_features:
    plt.plot(['90s', '00s', '10s'], bump_data.loc[feature], marker='o', label=feature)
plt.gca().invert_yaxis()  # Invert y-axis to have rank 1 at the top
plt.title('Bump Chart of Top Feature Rankings Across Decades')
plt.xlabel('Decade')
plt.ylabel('Rank')
plt.legend(title='Feature')
#plt.savefig('decade_comparison_bump_chart.png')
plt.show()

# Prepare visualization of the distribution of the top features in hits across decades using kde plots
fig, axes = plt.subplots(2, 4, figsize=(15, 10))
axes = axes.flatten()
for idx, feature in enumerate(joined_top_features):
    sns.kdeplot(data=combined_df, x=feature, hue='decade', fill=True, ax=axes[idx])
    axes[idx].set_title(f'Distribution of {feature} in Hits Across Decades')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Density')
plt.tight_layout()
#plt.savefig('decade_comparison_kde_plots.png')
plt.show()