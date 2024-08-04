import sys
import pandas as pd 
import numpy as np
sys.path.append('D:/betting/esport')

from func import date_formating, expand_df, cleansing_df, adjust_scores, filter_by_match_count

df = pd.read_csv("D:\\betting\\esport\\valorant.csv")

df = date_formating(df)
expanded_df = expand_df(df)
expanded_df = cleansing_df(expanded_df)
expanded_df = adjust_scores(expanded_df)
filtered_df = filter_by_match_count (expanded_df, match_count = 20)






test = filtered_df.copy()
test

test[test["result"] == "draw"]



def calculate_overall_win_rate(df):
    # Calculate the win rate for each team
    df['team_wins'] = df.apply(lambda row: row['score'] > row['opponent_score'], axis=1).astype(int)
    team_stats = df.groupby('team')['team_wins'].agg(['sum', 'count']).reset_index()
    team_stats.rename(columns={'sum': 'total_wins', 'count': 'total_games'}, inplace=True)
    team_stats['win_rate'] = team_stats['total_wins'] / team_stats['total_games']
    
    # Merge win rate back into the main DataFrame
    df = df.merge(team_stats[['team', 'win_rate']], on='team', how='left')
    
    return df


test = calculate_overall_win_rate(filtered_df)



def calculate_moving_rolling_win_rates(df, window_size=3, max_shifts=6):
    # Ensure 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'], format='%m-%d-%Y')
    
    # Ensure data is sorted by team and date
    df = df.sort_values(by=['team', 'date'])
    
    # Create a binary win indicator for each game
    df['win'] = (df['score'] > df['opponent_score']).astype(int)
    
    # Compute rolling win rates with a fixed window size
    for shift in range(max_shifts):
        start_col = f'rolling_win_rate_start_{shift + 1}'
        df[start_col] = df.groupby('team')['win'].transform(lambda x: x.shift(shift).rolling(window=window_size, min_periods=1).mean())
    
    # Drop the temporary 'win' column as it's no longer needed
    df.drop(columns=['win'], inplace=True)
    
    return df




def calculate_cumulative_win_rate(df):
    # Ensure 'date' is in datetime format and sort by team and date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['team', 'date']).reset_index(drop=True)
    
    # Calculate whether the team won the match
    df['team_wins'] = df['score'] > df['opponent_score']
    
    # Calculate cumulative wins and games for each team
    df['cumulative_wins'] = df.groupby('team')['team_wins'].cumsum()
    df['cumulative_games'] = df.groupby('team').cumcount() + 1
    
    # Calculate win rate
    df['win_rate'] = df['cumulative_wins'] / df['cumulative_games']
    
    # Calculate win rate before this match (one match back)
    df['win_rate_before'] = df.groupby('team')['win_rate'].shift(1)
    df['cumulative_games_before'] = df.groupby('team')['cumulative_games'].shift(1)
    df['cumulative_wins_before'] = df.groupby('team')['cumulative_wins'].shift(1)

    return df

def add_opponent_win_rate(df):
    # Calculate cumulative win rates for each team
    df = calculate_cumulative_win_rate(df)
    
    # Create a dictionary to quickly lookup win rates by team and date
    win_rate_lookup = df.set_index(['team', 'date'])['win_rate'].to_dict()
    
    # Function to get the opponent's win rate for each match
    def get_opponent_win_rate(row):
        opponent_team = row['opponent']
        match_date = row['date']
        # Find the most recent match of the opponent before the current match date
        opponent_matches = df[(df['team'] == opponent_team) & (df['date'] < match_date)]
        if not opponent_matches.empty:
            opponent_win_rate = opponent_matches.iloc[-1]['win_rate']
        else:
            opponent_win_rate = 0
        return opponent_win_rate

    df['win_rate_opponent'] = df.apply(get_opponent_win_rate, axis=1)
    
    return df











def add_last_game_columns(df):
    """
    Adds 'last_game' and 'opponent_last_game' columns to the DataFrame which indicate whether the team and their opponent won
    their last match before the current one, grouped by each team.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing match data with 'team', 'opponent', 'date', and 'team_wins' columns.
    
    Returns:
    pd.DataFrame: DataFrame with additional 'last_game' and 'opponent_last_game' columns.
    """
    # Ensure DataFrame is sorted by 'team' and 'date'
    df = df.sort_values(by=['team', 'date']).reset_index(drop=True)
    
    # Define a function to apply to each group for team's last game
    def add_last_game_for_team(group):
        group['last_game'] = group['team_wins'].shift(1)
        group['last_game'] = group['last_game'].fillna(0).astype(int)
        return group
    
    # Define a function to apply to each group for opponent's last game
    def add_last_game_for_opponent(group):
        group['opponent_last_game'] = group['team_wins'].shift(1)
        group['opponent_last_game'] = group['opponent_last_game'].fillna(0).astype(int)
        return group
    
    # Apply the function to each group for the team's last game
    df = df.groupby('team').apply(add_last_game_for_team).reset_index(drop=True)    
    # Create a temporary DataFrame to calculate opponent's last game
    df_opponent = df.rename(columns={'team': 'opponent', 'opponent': 'team', 'team_wins': 'opponent_wins'})    
    # Apply the function to each group for the opponent's last game
    df_opponent = df_opponent.groupby('opponent').apply(add_last_game_for_opponent).reset_index(drop=True)    
    # Merge the opponent's last game results back into the original DataFrame
    df = df.merge(df_opponent[['date', 'team', 'opponent_last_game']], left_on=['date', 'team'], right_on=['date', 'team'], how='left')
    
    return df



test = calculate_cumulative_win_rate(test)
test = add_opponent_win_rate(test)


test


vita_df = test[test["team"] == "TEAM VITALITY"]

trace_df=test[test["team"] == "TRACE ESPORTS"]

trace_df.tail(1)

vita_df.tail(1)

vita_df

jou = add_last_game_columns(jou)



trace_df


jou

[]
vita_df
import seaborn as sns
import matplotlib.pyplot as plt

vita_df = jou[jou["team"] == "TEAM VITALITY"]

vita_df.columns
# Encode categorical variables
vita_df['day_of_week'] = pd.Categorical(vita_df['day_of_week']).codes
vita_df['month'] = pd.Categorical(vita_df['month']).codes
vita_df['hour'] = pd.Categorical(vita_df['hour']).codes
vita_df['hour'] = pd.Categorical(vita_df['hour']).codes
vita_df['is_weekend'] = vita_df['is_weekend'].astype(int)

# Calculate correlation matrix
corr_matrix = vita_df[['team_wins', 'cumulative_wins', 'cumulative_games', 'win_rate', 'day_of_week', 'month', 'hour', 'is_weekend','win_rate_before_this_match',
                       "win_rate_opponent"]].corr()

# Set up the matplotlib figure
plt.figure(figsize=(14, 12))

# Define the color palette
cmap = sns.color_palette("coolwarm", as_cmap=True)

# Draw the heatmap
sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, center=0,
            cbar_kws={'shrink': .8, 'label': 'Correlation Coefficient'})

# Add titles and labels
plt.title('Correlation Matrix Heatmap with Additional Features', fontsize=16)
plt.xticks(ticks=np.arange(len(corr_matrix.columns)) + 0.5, labels=corr_matrix.columns, rotation=45, ha='right')
plt.yticks(ticks=np.arange(len(corr_matrix.columns)) + 0.5, labels=corr_matrix.columns, rotation=0, va='center')

# Show plot
plt.tight_layout()
plt.show()