import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset for 10s songs
data = pd.read_csv('./Data/dataset-of-10s.csv')
columns_corr = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                'duration_ms', 'time_signature', 'chorus_hit', 'sections', 'target'] 


# Calculate correlations and identify best features
data_num = data[columns_corr]
corr = abs(data_num.corr(numeric_only=True)['target']).sort_values(ascending=False)
best_features = corr[1:7].index
best_features_df = data[best_features]
best_features_df['target'] = data['target']

#sns.pairplot(best_features_df, hue='target') # can be uncommented to visualize pairplot for 10s songs
#plt.show()


# Box plots for all features in 10s songs
fig, axes = plt.subplots(3, 5, figsize=(15, 10))
axes = axes.flatten()
for idx, col in enumerate(columns_corr[:-1]):  # exclude target
    axes[idx].boxplot(data[col])
    axes[idx].set_title(col)

#plt.tight_layout()
#plt.title('Box Plots of Features for 10s Songs')
#plt.show() # can be uncommented to visualize box plots for 10s songs

best_10s_features = corr # can be used in decade_comparison.py to compare with 00s and 90s features
best_10s_features_df = best_features_df
data_10s = data