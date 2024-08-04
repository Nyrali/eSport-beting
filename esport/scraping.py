import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

# Fetch the webpage
url = 'https://www.vlr.gg/379340/reta-esports-vs-all-knights-challengers-league-2024-latam-regional-playoffs-winners/?game=180564&tab=overview'
url = 'https://www.vlr.gg/matches/results'



response = requests.get(url)
webpage = response.content

# Parse HTML content
soup = BeautifulSoup(webpage, 'html.parser')


# Locate the section containing the match details
date_raw = soup.find('div', class_='match-header-date').get_text(strip=True)
teams = soup.find_all('div', class_='wf-title-med')
score = soup.find('div', class_='js-spoiler').get_text(strip=True)

# Get team names
team1 = teams[0].get_text(strip=True)
team2 = teams[1].get_text(strip=True)

# Convert the date to mm-dd-yyyy format
# Example format for extracted date (assuming it is like "Saturday, July 27th")
# Adjust the format string based on the actual date format you get
date_formats = ['%A, %B %d', '%A %B %d', '%d %B %Y']
for fmt in date_formats:
    try:
        date_obj = datetime.strptime(date_raw, fmt)
        date_text = date_obj.strftime('%m-%d-%Y')
        break
    except ValueError:
        continue

# Create a dictionary with the extracted details
match_data = {
    'Date': [date_text],
    'Team 1': [team1],
    'Team 2': [team2],
    'Score': [score]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(match_data)

# Print the DataFrame
print(df)


date_raw = 'Saturday, July 27th'  # Example date, replace with actual date extraction if needed

# Define date formats to try
date_formats = ['%A, %B %d', '%A %B %d', '%d %B %Y']
date_text = 'Unknown Date'
for fmt in date_formats:
    try:
        date_obj = datetime.strptime(date_raw, fmt)
        date_text = date_obj.strftime('%m-%d-%Y')
        break
    except ValueError:
        continue

# Add the converted date to the DataFrame
df['Date'] = df['Date'].replace('Unknown Date', date_text)

# Split the Score column into two columns: 'Team 1 Score' and 'Team 2 Score'
df[['Team 1 Score', 'Team 2 Score']] = df['Score'].str.split(':', expand=True)

# Convert scores to integer
df['Team 1 Score'] = pd.to_numeric(df['Team 1 Score'], errors='coerce')
df['Team 2 Score'] = pd.to_numeric(df['Team 2 Score'], errors='coerce')

# Determine win/loss results
def determine_results(row):
    if row['Team 1 Score'] > row['Team 2 Score']:
        return pd.Series(['win', 'loss'])
    elif row['Team 1 Score'] < row['Team 2 Score']:
        return pd.Series(['loss', 'win'])
    else:
        return pd.Series(['draw', 'draw'])

df[['Team 1 Result', 'Team 2 Result']] = df.apply(determine_results, axis=1)




import requests
from bs4 import BeautifulSoup
import pandas as pd


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

base_url = 'https://www.vlr.gg/matches/results'
num_pages = 2
jou = scrape_multiple_pages(base_url, num_pages)


chm = scrap_results('https://www.vlr.gg/matches/results=3')

chm

jou.head(50)
jou.tail(50)