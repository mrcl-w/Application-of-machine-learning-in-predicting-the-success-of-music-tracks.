import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_song_billboard_placement(song_title, song_artist, data):
    song_data = data[(data['Normalized Title'] == song_title) & (data['Artist'] == song_artist)]
    if not song_data.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Date', y='Rank', data=song_data)
        plt.gca().invert_yaxis()  # Invert y-axis to show rank 1 at the top
        plt.title(f'Billboard Placement Over Time for "{song_title.title()}"')
        plt.xlabel('Date')
        plt.ylabel('Rank')
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()
    else:
        print(f'No data found for the song "{song_title}".')

def plot_song_prediction_comparison(test_data, song_name, y_pred):
    test_data['predicted_next_rank'] = y_pred
    plot_data = test_data[test_data['Song'] == song_name]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_data, x='Date', y='target_next_rank', label='Actual Rank')
    sns.lineplot(data=plot_data, x='Date', y='predicted_next_rank', label='Predicted Rank')
    plt.gca().invert_yaxis()  # Invert y-axis to have rank 1 at the top
    plt.title(f'Actual vs Predicted Rank Over Time for {song_name}')

def plot_song_prediction_comparison_lstm(test_meta, song_name, y_pred):
    plot_df = test_meta.copy()

    plot_df['predicted_next_rank'] = np.asarray(y_pred).reshape(-1)

    plot_data = (
        plot_df[plot_df['Song'].str.lower() == song_name.lower()]
        .sort_values('Date')
    )

    if plot_data.empty:
        print(f"No data found for song: {song_name}")
        return

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=plot_data,
        x='Date',
        y='target_next_rank',
        marker='o',
        label='Actual next week rank'
    )

    sns.lineplot(
        data=plot_data,
        x='Date',
        y='predicted_next_rank',
        marker='o',
        label='Predicted next week rank'
    )

    plt.gca().invert_yaxis()
    plt.title(f'Actual vs Predicted Next Week Rank: {song_name}')
    plt.xlabel('Date')
    plt.ylabel('Billboard Rank')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()