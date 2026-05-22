import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def _prepare_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10


def plot_song_billboard_placement(song_title, song_artist, data):
    _prepare_plot_style()

    song_data = data[
        (data['Normalized Title'] == song_title) &
        (data['Artist'] == song_artist)
    ].copy()

    song_data = song_data.sort_values('Date')

    if song_data.empty:
        print(f'No data found for the song "{song_title}".')
        return

    plt.figure()

    sns.lineplot(
        x='Date',
        y='Rank',
        data=song_data,
        marker='o',
        label='Actual rank'
    )

    plt.gca().invert_yaxis()
    plt.title(f'Billboard Placement Over Time: {song_title.title()}')
    plt.xlabel('Date')
    plt.ylabel('Billboard Rank')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_song_prediction_comparison(test_data, song_name, y_pred):
    _prepare_plot_style()

    plot_df = test_data.copy()
    plot_df['predicted_next_rank'] = np.asarray(y_pred).reshape(-1)

    plot_data = (
        plot_df[
            plot_df['Song']
            .astype(str)
            .str.strip()
            .str.lower()
            .str.contains(song_name.strip().lower(), na=False)
        ]
        .sort_values('Date')
    )

    if plot_data.empty:
        print(f"No data found for song: {song_name}")
        return

    plt.figure()

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
        label='XGBoost prediction'
    )

    plt.gca().invert_yaxis()
    plt.title(f'Actual vs XGBoost Prediction: {song_name}')
    plt.xlabel('Date')
    plt.ylabel('Billboard Rank')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_song_prediction_comparison_lstm(test_meta, song_name, y_pred):
    _prepare_plot_style()

    plot_df = test_meta.copy()
    plot_df['predicted_next_rank'] = np.asarray(y_pred).reshape(-1)

    plot_data = (
        plot_df[
            plot_df['Song']
            .astype(str)
            .str.strip()
            .str.lower()
            .str.contains(song_name.strip().lower(), na=False)
        ]
        .sort_values('Date')
    )

    if plot_data.empty:
        print(f"No data found for song: {song_name}")
        return

    plt.figure()

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
        label='LSTM prediction'
    )

    plt.gca().invert_yaxis()
    plt.title(f'Actual vs LSTM Prediction: {song_name}')
    plt.xlabel('Date')
    plt.ylabel('Billboard Rank')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_xgboost_lstm_comparison(
    xgb_test_data,
    lstm_test_meta,
    song_name,
    xgb_pred,
    lstm_pred
):
    _prepare_plot_style()

    xgb_df = xgb_test_data.copy()
    lstm_df = lstm_test_meta.copy()

    xgb_df['xgb_predicted_next_rank'] = np.asarray(xgb_pred).reshape(-1)
    lstm_df['lstm_predicted_next_rank'] = np.asarray(lstm_pred).reshape(-1)

    song_key = song_name.strip().lower()

    xgb_plot = (
        xgb_df[
            xgb_df['Song']
            .astype(str)
            .str.strip()
            .str.lower()
            .str.contains(song_key, na=False)
        ][['Song', 'Date', 'target_next_rank', 'xgb_predicted_next_rank']]
        .copy()
    )

    lstm_plot = (
        lstm_df[
            lstm_df['Song']
            .astype(str)
            .str.strip()
            .str.lower()
            .str.contains(song_key, na=False)
        ][['Song', 'Date', 'lstm_predicted_next_rank']]
        .copy()
    )

    if xgb_plot.empty and lstm_plot.empty:
        print(f"No data found for song: {song_name}")
        return

    plot_data = pd.merge(
        xgb_plot,
        lstm_plot,
        on=['Song', 'Date'],
        how='outer'
    ).sort_values('Date')

    plt.figure()

    if 'target_next_rank' in plot_data.columns:
        sns.lineplot(
            data=plot_data,
            x='Date',
            y='target_next_rank',
            marker='o',
            label='Actual next week rank'
        )

    if 'xgb_predicted_next_rank' in plot_data.columns:
        sns.lineplot(
            data=plot_data,
            x='Date',
            y='xgb_predicted_next_rank',
            marker='o',
            label='XGBoost prediction'
        )

    if 'lstm_predicted_next_rank' in plot_data.columns:
        sns.lineplot(
            data=plot_data,
            x='Date',
            y='lstm_predicted_next_rank',
            marker='o',
            label='LSTM prediction'
        )

    plt.gca().invert_yaxis()
    plt.title(f'Actual vs XGBoost vs LSTM: {song_name}')
    plt.xlabel('Date')
    plt.ylabel('Billboard Rank')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()