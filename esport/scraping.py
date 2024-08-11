import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

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

# Fetch the webpage
base_url = 'https://www.vlr.gg/matches/results'
num_pages = 525
df = scrape_multiple_pages(base_url, num_pages)

df.to_csv("D:\\betting\\esport\\valorant_raw_enhanced.csv")