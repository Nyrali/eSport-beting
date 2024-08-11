import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('D:/betting/esport')

from func import date_formating, expand_df, cleansing_df, adjust_scores, filter_by_match_count, calculate_cumulative_win_rate, calculate_rolling_sums, calculate_win_streaks, drop_rolling_columns, make_opponent_df, create_match_id



df = pd.read_csv("D:\\betting\\esport\\valorant_raw_enhanced.csv")


df = date_formating(df)
expanded_df = expand_df(df)
cleansed_df = cleansing_df(expanded_df)
score_df = adjust_scores(cleansed_df)
filtered_df = filter_by_match_count (score_df, match_count = 11)
filtered_df = filtered_df.dropna()


winrate_df = calculate_cumulative_win_rate(filtered_df)
rolling_win_rate_df = calculate_rolling_sums(winrate_df)
win_streak_df = calculate_win_streaks(rolling_win_rate_df)
win_streak_df = drop_rolling_columns(win_streak_df)


bm_cols = [i for i in win_streak_df.columns if "bm" in i]
opp_df = make_opponent_df(win_streak_df, bm_cols, prefix ="opp_")
opp_df = opp_df.drop_duplicates(subset=['opponent', 'datetime'])

duplicates = opp_df[opp_df.duplicated(subset=['opponent', 'datetime'], keep=False)]
merged_df = pd.merge(win_streak_df,opp_df, on =["opponent", "datetime", "date"], how = "left" )

# TODO  making custom match_id - is needed?
merged_df['match_id_1'] = merged_df.apply(create_match_id, axis=1)

# droping one team record from the match for model
filtered_df = merged_df.drop_duplicates(subset=['match_id'])

features_df = filtered_df[bm_cols+["team_wins"]]


bm_cols = [i for i in filtered_df.columns if "bm" in i]
merged_df[bm_cols+["team_wins"]].to_csv("D:\\eSport-betting\\esport\\valorant_features_all.csv")
filtered_df[bm_cols+["team_wins"]].to_csv("D:\\eSport-betting\\esport\\valorant_features_one_row_per_match.csv")


# corr
bm_cols = [i for i in filtered_df.columns if "bm" in i]
corr_df = filtered_df[bm_cols + ["team_wins"]]

corr_df['team_wins'] = corr_df['team_wins'].astype(int)
corr_df = corr_df.dropna()

correlations = corr_df.corr()
team_wins_correlations = correlations['team_wins'].drop('team_wins')

team_wins_correlations


correlations

# Create a bar plot
correlations_series = pd.Series(team_wins_correlations)

# Create a horizontal bar plot
plt.figure(figsize=(12, 8))
correlations_series.plot(kind='barh', color='skyblue')
plt.xlabel('Correlation with team_wins')
plt.title('Correlation of Different Features with team_wins')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()


import seaborn as sns
# Plotting the heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()



merged_df

team1_columns = [col for col in merged_df.columns if 'team' in col]
team2_columns = [col for col in merged_df.columns if 'opponent' in col]

# Ensure you have columns for both teams
assert len(team1_columns) == len(team2_columns), "Mismatch in number of team1 and team2 columns"

cleaned_df = pd.DataFrame()
test = merged_df.copy()


merged_df

merged_df['datetime'] = pd.to_datetime(merged_df['datetime']).astype(str)



merged_df['match_id'] = merged_df.apply(create_match_id, axis=1)
filtered_df = merged_df.drop_duplicates(subset=['match_id'])

filtered_df


merged_df[merged_df["match_id"] == "2024-07-16 18:00:00_TEAM HERETICS_TEAM VITALITY"]

merged_df[merged_df["match_id"] == "2024-08-03 12:30:00_TEAM VITALITY_TRACE ESPORTS"]

filtered_df[filtered_df["match_id"] == "2024-08-03 12:30:00_TEAM VITALITY_TRACE ESPORTS"]


pe_df = merged_df[merged_df["team"] == "PCIFIC ESPOR"]
expanded_df[expanded_df["team"] == "PCIFIC ESPOR"]

ista_df = merged_df[merged_df["team"] == "İSTANBUL WILDCATS"]

vita_df_merged = merged_df[merged_df["team"] == "TEAM VITALITY"]

trace_df_merged = merged_df[merged_df["team"] == "TRACE ESPORTS"]

vita_df = win_streak_df[win_streak_df["team"] == "TEAM VITALITY"]

trace_df_opp=opp_df[(opp_df["opponent"] == "TRACE ESPORTS") & (opp_df["datetime"] == "2024-08-03 12:30:00")]

vita_df_merged.iloc[135]
trace_df_merged.iloc[47]



def expand_df(df):

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    expanded_records = []

    # Identify all columns and columns to exclude
    all_cols = df.columns.tolist()
    col_exclude = ["team", "opponent", "score", "opponent_score", "team_1", "team_2",
                   "score_1", "score_2"]

    # Determine which columns to include
    columns_to_include = [col for col in all_cols if col not in col_exclude]

    # Iterate through each row in the original DataFrame
    for row in df.itertuples(index=False):
        # Create a row for team_1
        record_team_1 = {
            'team': row.team_1,
            'opponent': row.team_2,
            'score': row.score_1,
            'opponent_score': row.score_2,
        }
        
        # Add additional columns
        record_team_1.update({
            col: getattr(row, col) for col in columns_to_include
        })

        expanded_records.append(record_team_1)
        
        # Create a row for team_2
        record_team_2 = {
            'team': row.team_2,
            'opponent': row.team_1,
            'score': row.score_2,
            'opponent_score': row.score_1,
        }
        
        # Add additional columns
        record_team_2.update({
            col: getattr(row, col) for col in columns_to_include
        })

        expanded_records.append(record_team_2)

    # Create a new DataFrame from the expanded records
    expanded_df = pd.DataFrame(expanded_records)
    
    return expanded_df

def create_match_id(row):
    # Sort the teams to ensure consistent ordering
    teams = sorted([row['team'], row['opponent']])
    return f"{row['datetime']}_{teams[0]}_{teams[1]}"
# Apply the function to create standardized match IDs


def drop_rolling_columns(df):
    # Drop columns containing the substring 'rolling'
    columns_to_drop = df.filter(regex='rolling').columns
    df = df.drop(columns=columns_to_drop)
    return df



def calculate_cumulative_win_rate(df):
    # Ensure 'date' is in datetime format and sort by team and date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['team', 'date']).reset_index(drop=True)
    
    # Calculate whether the team won the match
    df['team_wins'] = df['score'] > df['opponent_score']
    
    # Calculate cumulative wins and games for each team
    df['games_won'] = df.groupby('team')['team_wins'].cumsum()
    df['games_played'] = df.groupby('team').cumcount() + 1
    
    # Calculate win rate
    df['win_rate'] = df['games_won'] / df['games_played']
    
    # Calculate win rate before this match (one match back)
    df['win_rate_bm'] = df.groupby('team')['win_rate'].shift(1)
    df['games_played_bm'] = df.groupby('team')['games_played'].shift(1)
    df['games_won_bm'] = df.groupby('team')['games_won'].shift(1)

    
    return df



def calculate_rolling_sums(df):
    # Ensure 'date' is in datetime format and sort by team and date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['team', 'date']).reset_index(drop=True)
    
    # Calculate whether the team won the match
    df['team_wins'] = df['score'] > df['opponent_score']
    
    # Calculate rolling sums with the correct min_periods
    df['rolling_sum_3'] = df.groupby('team')['team_wins'].rolling(window=3, min_periods=3).sum().reset_index(level=0, drop=True)
    df['rolling_sum_5'] = df.groupby('team')['team_wins'].rolling(window=5, min_periods=5).sum().reset_index(level=0, drop=True)
    df['rolling_sum_8'] = df.groupby('team')['team_wins'].rolling(window=8, min_periods=8).sum().reset_index(level=0, drop=True)
    df['rolling_sum_10'] = df.groupby('team')['team_wins'].rolling(window=10, min_periods=10).sum().reset_index(level=0, drop=True)
    
    # Correct win rate calculations
    df["win_rate_last_3"] = df["rolling_sum_3"] / 3
    df["win_rate_last_5"] = df["rolling_sum_5"] / 5
    df["win_rate_last_8"] = df["rolling_sum_8"] / 8
    df["win_rate_last_10"] = df["rolling_sum_10"] / 10
    
    # Shift win rates by one row to reflect the previous window
    df["win_rate_last_3_bm"] = df.groupby('team')["win_rate_last_3"].shift(1)
    df["win_rate_last_5_bm"] = df.groupby('team')["win_rate_last_5"].shift(1)
    df["win_rate_last_8_bm"] = df.groupby('team')["win_rate_last_8"].shift(1)
    df["win_rate_last_10_bm"] = df.groupby('team')["win_rate_last_10"].shift(1)

    return df


def calculate_win_streaks(df):
    # Ensure 'date' is in datetime format and sort by team and date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['team', 'date']).reset_index(drop=True)
    
    # Calculate whether the team won the match
    df['team_wins'] = df['score'] > df['opponent_score']
    
    # Define a function to check for streaks
    def streak_check(series, length):
        return series.rolling(window=length, min_periods=length).apply(lambda x: x.all(), raw=True)
    
    # Calculate rolling streaks
    df['win_streak_3'] = df.groupby('team')['team_wins'].apply(lambda x: streak_check(x, 3)).reset_index(level=0, drop=True)
    df['win_streak_5'] = df.groupby('team')['team_wins'].apply(lambda x: streak_check(x, 5)).reset_index(level=0, drop=True)
    
    df['win_streak_3_bm'] = df.groupby('team')["win_streak_3"].shift(1)
    df['win_streak_5_bm'] = df.groupby('team')["win_streak_5"].shift(1)

    return df


def make_opponent_df(df, col_lst, prefix ="opp_"):
    opp_df = df.copy()  
    
    opp_df["opponent"] = opp_df["team"]
    rename_dict = {col: f"{prefix}{col}" for col in col_lst}
    opp_df = opp_df.rename(columns=rename_dict)

    new_cols = [prefix + i for i in col_lst]
    selected_cols = ["opponent", "date", "datetime"] + new_cols
    selected_cols = [col for col in selected_cols if col in opp_df.columns]
    
    opp_df = opp_df[selected_cols]

    return opp_df











##LEGACY ###


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


def add_prefix_to_columns(df, col_lst, prefix='opp_'):
    opp_df = df.copy()  
    # Iterate over each column name in the list
    for col in col_lst:
        # Create a new column with the prefix
        opp_df[prefix + col] = opp_df[col]
        new_cols = [prefix + i for i in bm_cols]

        new_cols = [prefix + i for i in bm_cols]
        selected_cols = ["opponent", "date", "datetime"] + new_cols
        selected_cols = [col for col in selected_cols if col in opp_df.columns]
        
        opp_df = opp_df[selected_cols]


    return df


# Define the function to process each group
def opponent_stats(group):
    team = group['team'].iloc[0]
    opponent = group['opponent'].iloc[0]
    date = group['date'].iloc[0]

    temp_df = winrate_df[(winrate_df['team'] == team) & (winrate_df['opponent'] == opponent) & (winrate_df['date'] == date)]
    opponent_df = winrate_df[(winrate_df['team'] == opponent) & (winrate_df['opponent'] == team) & (winrate_df['date'] == date)]

    if not opponent_df.empty:
        opponent_df = opponent_df.rename(columns={
            "games_won": "opponent_games_won",
            "games_played": "opponent_games_played",
            "win_rate": "opponent_win_rate",
            "win_rate_bm": "opponent_win_rate_bm",
            "games_played_bm": "opponent_games_played_bm",
            "games_won_bm": "opponent_games_won_bm"
        })

        temp_df["opponent_games_won"] = opponent_df["opponent_games_won"].values[0]
        temp_df["opponent_games_played"] = opponent_df["opponent_games_played"].values[0]
        temp_df["opponent_win_rate"] = opponent_df["opponent_win_rate"].values[0]
        temp_df["opponent_win_rate_bm"] = opponent_df["opponent_win_rate_bm"].values[0]
        temp_df["opponent_games_played_bm"] = opponent_df["opponent_games_played_bm"].values[0]
        temp_df["opponent_games_won_bm"] = opponent_df["opponent_games_won_bm"].values[0]
    else:
        temp_df["opponent_games_won"] = None
        temp_df["opponent_games_played"] = None
        temp_df["opponent_win_rate"] = None
        temp_df["opponent_win_rate_bm"] = None
        temp_df["opponent_games_played_bm"] = None
        temp_df["opponent_games_won_bm"] = None
    
    return temp_df

def make_opponent_df(df):
    # Step 1: Rename the columns that do not conflict
    temp_opponent_df = df.copy()  

    temp_opponent_df["opponent"] = temp_opponent_df["team"]

    temp_opponent_df["opponent_games_won"] = temp_opponent_df["games_won"]
    temp_opponent_df["opponent_games_played"] = temp_opponent_df["games_played"]
    temp_opponent_df["opponent_win_rate"] = temp_opponent_df["win_rate"]
    temp_opponent_df["opponent_win_rate_bm"] = temp_opponent_df["win_rate_bm"]
    temp_opponent_df["opponent_games_played_bm"] = temp_opponent_df["games_played_bm"]
    temp_opponent_df["opponent_games_won_bm"] = temp_opponent_df["games_won_bm"]
    
    temp_opponent_df = temp_opponent_df[["opponent", "date", "datetime","opponent_games_played", "opponent_games_won", "opponent_win_rate",
                       "opponent_win_rate_bm", "opponent_games_played_bm", "opponent_games_won_bm"]]

    return temp_opponent_df
