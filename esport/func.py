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
        # Extract date
        date_header = card.find_previous('div', class_='wf-label mod-large')
        date_text = date_header.get_text(strip=True) if date_header else "Unknown Date"
        
        # Extract time
        time_elements = card.find_all('div', class_='match-item-time')
        time_texts = [time_element.get_text(strip=True) for time_element in time_elements]
        
        # Extract team names
        team_elements = card.find_all('div', class_='match-item-vs-team-name')
        teams = [team_element.get_text(strip=True) for team_element in team_elements]
        
        # Extract scores
        score_elements = card.find_all('div', class_='match-item-vs-team-score')
        scores = [score_element.get_text(strip=True) for score_element in score_elements]

        
        
        # Check how to group teams and scores
        if len(teams) % 2 == 0 and len(scores) == len(teams):  # Ensure teams and scores are in pairs
            num_matches = len(teams) // 2
            for i in range(num_matches):
                # Adjust time handling if there are multiple times
                time_text = time_texts[i] if i < len(time_texts) else "Unknown Time"
                
                match_record = {
                    'date': date_text,
                    'time': time_text,
                    'team_1': teams[i * 2],
                    'team_2': teams[i * 2 + 1],
                    'score_1': scores[i * 2],
                    'score_2': scores[i * 2 + 1]
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
    expanded_records = []

    # Iterate through each row in the original DataFrame
    for _, row in df.iterrows():
        # Create a row for team_1
        record_team_1 = {
            'date': row['date'],
            'time': row['time'],
            'team': row['team_1'],
            'opponent': row['team_2'],
            'score': row['score_1'],
            'opponent_score': row['score_2'],
            'month': row['month'],
            'datetime': row['datetime'],
            'day_of_week': row['day_of_week'],
            'hour': row['hour'],
            'is_weekend': row['is_weekend']
        }
        expanded_records.append(record_team_1)
        
        # Create a row for team_2
        record_team_2 = {
            'date': row['date'],
            'time': row['time'],
            'team': row['team_2'],
            'opponent': row['team_1'],
            'score': row['score_2'],
            'opponent_score': row['score_1'],
            'month': row['month'],
            'datetime': row['datetime'],
            'day_of_week': row['day_of_week'],
            'hour': row['hour'],
            'is_weekend': row['is_weekend']
        }
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
