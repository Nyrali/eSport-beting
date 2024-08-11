import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os
import numpy as np

def scrap_results(url):
    # Fetch the webpage content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Initialize a list to hold match details
    match_records = []

    # Extract all match cards directly
    match_cards = soup.find_all('div', class_='wf-card')

    for card in match_cards:
        # Extract date from the card header
        date_header = card.find_previous('div', class_='wf-label mod-large')
        date_text = date_header.get_text(strip=True) if date_header else "Unknown Date"

        # Extract match items within the card
        match_items = card.find_all('a', class_='match-item')

        for item in match_items:
            # Extract time
            time_element = item.find('div', class_='match-item-time')
            time_text = time_element.get_text(strip=True) if time_element else "Unknown Time"

            # Extract team names
            team_elements = item.find_all('div', class_='match-item-vs-team-name')
            teams = [team_element.get_text(strip=True) for team_element in team_elements]

            # Extract scores
            score_elements = item.find_all('div', class_='match-item-vs-team-score')
            scores = [score_element.get_text(strip=True) for score_element in score_elements]

            # Extract event text
            event_element = item.find('div', class_='match-item-event')
            event_text = event_element.get_text(strip=True) if event_element else "Unknown Event"

            # Extract full match_id from the <a> tag
            link_tag = item
            match_url = link_tag['href'] if link_tag and 'href' in link_tag.attrs else "Unknown URL"
            match_url = "https://www.vlr.gg" + match_url
            match_id = link_tag['href'].split('/')[-1] if link_tag and 'href' in link_tag.attrs else "Unknown Match ID"
            
            
            # Ensure we have teams and scores in pairs
            if len(teams) % 2 == 0 and len(scores) == len(teams):  # Ensure teams and scores are in pairs
                num_matches = len(teams) // 2
                for i in range(num_matches):
                    # Create match record with correct indices
                    match_record = {
                        'date': date_text,
                        'time': time_text,
                        'team_1': teams[i * 2] if i * 2 < len(teams) else "Unknown Team",
                        'team_2': teams[i * 2 + 1] if i * 2 + 1 < len(teams) else "Unknown Team",
                        'score_1': scores[i * 2] if i * 2 < len(scores) else "Unknown Score",
                        'score_2': scores[i * 2 + 1] if i * 2 + 1 < len(scores) else "Unknown Score",
                        'event': event_text,
                        'match_id': match_id,
                        'url': match_url 
                    }
                    match_records.append(match_record)

    # Create a DataFrame from the match records
    df = pd.DataFrame(match_records)

    # Drop duplicates and reset index
    df = df.drop_duplicates().reset_index(drop=True)  
    
    return df


def scrape_multiple_pages(base_url, num_pages):
    all_dfs = []
    # Scrape the first page
    print(f"Scraping first page: {base_url}")
    df_first_page = scrap_results(base_url)
    all_dfs.append(df_first_page)

    # Scrape subsequent pages
    for page in range(2, num_pages + 1):
        url = f"{base_url}?page={page}"
        print(f"Scraping page: {url}")
        df = scrap_results(url)
        all_dfs.append(df)
    
    # Concatenate all DataFrames
    full_df = pd.concat(all_dfs, ignore_index=True)
    return full_df


def date_formating(df):
    # Replace 'Today' and 'Yesterday' in the date column
    df['date'] = df['date'].str.replace('Today', '').str.replace('Yesterday', '')
    # Parse the 'date' column
    df['date'] = pd.to_datetime(df['date'], format='%a, %B %d, %Y', errors='coerce')
    # Extract month from 'date'
    df['month'] = df['date'].dt.month
    # Replace 'TBD' in 'time' with NaN
    df['time'] = df['time'].replace('TBD', np.nan)
    # Create 'datetime' column
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
    # Extract day of the week and hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['hour'] = df['datetime'].dt.hour
    # Determine if it's a weekend
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)
    # Convert dates to 'mm-dd-yyyy' format
    df['date'] = df['date'].dt.strftime('%m-%d-%Y')

    return df

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


def cleansing_df (expanded_df):
    
    expanded_df['team'] = expanded_df['team'].str.strip().str.upper()
    expanded_df['opponent'] = expanded_df['opponent'].str.strip().str.upper()
    expanded_df['score'] = pd.to_numeric(expanded_df['score'], errors='coerce')
    expanded_df['opponent_score'] = pd.to_numeric(expanded_df['opponent_score'], errors='coerce')
    
    return expanded_df


def adjust_scores(expanded_df):      

    # Create temporary columns to store new values
    new_score = np.where(
        (expanded_df['score'] >= 6) | (expanded_df['opponent_score'] >= 6), 
        np.where(expanded_df['score'] > expanded_df['opponent_score'], 1, 0), 
        expanded_df['score']
    )

    new_opponent_score = np.where(
        (expanded_df['score'] >= 6) | (expanded_df['opponent_score'] >= 6), 
        np.where(expanded_df['score'] > expanded_df['opponent_score'], 0, 1), 
        expanded_df['opponent_score']
    )

    expanded_df['score'] = new_score
    expanded_df['opponent_score'] = new_opponent_score

    expanded_df["result"] = np.where(expanded_df["score"] > expanded_df["opponent_score"], 1, 0)
    expanded_df["result"] = np.where(expanded_df["score"] == expanded_df["opponent_score"], np.nan, expanded_df["result"])

    return expanded_df


def filter_by_match_count (expanded_df, match_count = 10):
    team_counts_df = expanded_df.groupby("team").size().reset_index()
    enough_matches_df = team_counts_df[team_counts_df[0] >= match_count]
    enough_matches_team_lst = [i for i in enough_matches_df["team"] ]
    filtered_df = expanded_df[expanded_df["team"].isin(enough_matches_team_lst)]
    
    return filtered_df


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


def drop_rolling_columns(df):
    # Drop columns containing the substring 'rolling'
    columns_to_drop = df.filter(regex='rolling').columns
    df = df.drop(columns=columns_to_drop)
    return df


def create_match_id(row):
    # Sort the teams to ensure consistent ordering
    teams = sorted([row['team'], row['opponent']])
    return f"{row['datetime']}_{teams[0]}_{teams[1]}"
# Apply the function to create standardized match IDs


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
