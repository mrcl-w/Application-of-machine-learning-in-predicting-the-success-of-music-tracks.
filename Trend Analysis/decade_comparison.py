from analysis_00s import best_00s_features, best_00s_features_df, data_00s
from analysis_10s import best_10s_features, best_10s_features_df, data_10s
from analysis_90s import best_90s_features, best_90s_features_df, data_90s
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Combine the best features from each decade into a single DataFrame
#combined_df = pd.concat([best_00s_features_df, best_10s_features_df, best_90s_features_df], ignore_index=True)


# Add a decade column to differentiate the data
#combined_df['decade'] = ['00s'] * len(best_00s_features_df) + ['10s'] * len(best_10s_features_df) + ['90s'] * len(best_90s_features_df)


# Visualize the combined data using pairplot
#sns.pairplot(combined_df, hue='decade')
#plt.show() # can be uncommented to visualize pairplot comparing best features across decades

# Visualize only same features across decades using box plots
'''
features_to_compare = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, feature in enumerate(features_to_compare):
    sns.boxplot(x='decade', y=feature, data=combined_df, ax=axes[idx])
    axes[idx].set_title(f'Box Plot of {feature} by Decade')
plt.tight_layout()
plt.show()
'''

print("Best features from 90s: \n")
print(best_90s_features)
print("Best features from 00s: \n")
print(best_00s_features)
print("Best features from 10s: \n")
print(best_10s_features)
print("Combined DataFrame:")
#print(combined_df.head())
top6_90s = best_90s_features.index[1:7]
top6_00s = best_00s_features.index[1:7]
top6_10s = best_10s_features.index[1:7]
print("Top 6 features for 90s:", top6_90s)
print("Top 6 features for 00s:", top6_00s)
print("Top 6 features for 10s:", top6_10s)
joined_top_features = list(set(top6_90s).union(set(top6_00s)).union(set(top6_10s)))
print("Joined top features across all decades:", joined_top_features)
joined_features_00s = data_00s[joined_top_features]
joined_features_10s = data_10s[joined_top_features]
joined_features_90s = data_90s[joined_top_features]


combined_df = pd.concat([joined_features_00s, joined_features_10s, joined_features_90s], ignore_index=True)
combined_df['decade'] = ['00s'] * len(joined_features_00s) + ['10s'] * len(joined_features_10s) + ['90s'] * len(joined_features_90s)

fig, axes = plt.subplots(2, 4, figsize=(15, 10))
axes = axes.flatten()
test = combined_df[combined_df['target'] == 1]
for idx, feature in enumerate(test):
    sns.boxplot(x='decade', y=feature, data=combined_df, ax=axes[idx])
    axes[idx].set_title(f'Box Plot of {feature} by Decade')
plt.tight_layout()
plt.show()