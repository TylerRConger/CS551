import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('/content/NFL Player Stats(1922 - 2022).csv')

# Drop rows with missing values in the specified column
df = df.dropna(subset=['Pos', 'Pts/G'])

# Use iloc to select rows and columns by integer position
df = df.iloc[1:]

# Convert columns with numeric values to float
numeric_columns = ['RshTD', 'RecTD', 'PR TD', 'KR TD', 'FblTD', 'IntTD', 'OthTD', '2PM', '2PA', 'D2P', 'XPM', 'XPA', 'FGM', 'FGA', 'Sfty', 'Pts', 'Pts/G']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Extract the primary position in case of multiple positions separated by '/'
df['Primary_Pos'] = df['Pos'].str.split('/').str[0]

# Map positions to broader categories
position_mapping = {
    'QB': 'QB',
    'RB': 'RB',
    'FB': 'RB',
    'WR': 'WR',
    'TE': 'TE',
    'OL': 'OL',
    'C': 'OL',
    'G': 'OL',
    'T': 'OL',
    'DL': 'DL',
    'DE': 'DL',
    'DT': 'DL',
    'LB': 'LB',
    'ILB': 'LB',
    'OLB': 'LB',
    'DB': 'DB',
    'CB': 'DB',
    'S': 'DB',
    'K': 'K',
    'P': 'P',
}

# Map primary positions to broader categories
df['Position_Group'] = df['Primary_Pos'].map(position_mapping)

# Calculate points per game for each player
df['Pts_Per_Game'] = df['Pts/G']


# Find the player with the most points per game for each position group
max_pts_per_game = df.groupby('Position_Group')['Pts_Per_Game'].idxmax()

# Display the players with the most points per game for each position group and their respective seasons
result = df.loc[max_pts_per_game, ['Position_Group', 'Player', 'Season', 'Pts_Per_Game']]


resultDF = pd.DataFrame(result)


# Plot the data with player names as annotations
plt.figure(figsize=(10, 6))
bars = plt.bar(resultDF['Position_Group'], resultDF['Pts_Per_Game'], color='skyblue')

# Add player names as text annotations above each bar
for bar, player in zip(bars, resultDF['Player']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), player, ha='center', va='bottom')

plt.title('Top Players in Each Position Group - Points per Game')
plt.xlabel('Position Group')
plt.ylabel('Points per Game')
plt.show()


# Group by season and position group, and calculate the average points per game
grouped_data = df.groupby(['Season', 'Position_Group'])['Pts_Per_Game'].mean().reset_index()

# Separate offense and defense data
offense_data = grouped_data[grouped_data['Position_Group'].isin(['QB', 'RB', 'WR', 'TE', 'OL'])]
defense_data = grouped_data[grouped_data['Position_Group'].isin(['DL', 'LB', 'DB', 'K', 'P'])]

# Plot the results for offense
plt.figure(figsize=(12, 8))
for position_group in offense_data['Position_Group'].unique():
    position_group_data = offense_data[offense_data['Position_Group'] == position_group]
    plt.plot(position_group_data['Season'], position_group_data['Pts_Per_Game'], label=position_group)

plt.title('Average Points per Game by Offensive Position Group Over the Years')
plt.xlabel('Season')
plt.ylabel('Average Points per Game')
plt.legend()
plt.grid(True)
plt.show()

# Plot the results for defense and special tems
plt.figure(figsize=(12, 8))
for position_group in defense_data['Position_Group'].unique():
    position_group_data = defense_data[defense_data['Position_Group'] == position_group]
    plt.plot(position_group_data['Season'], position_group_data['Pts_Per_Game'], label=position_group)

plt.title('Average Points per Game by Defensive Position Group and Special Teams Position Group Over the Years')
plt.xlabel('Season')
plt.ylabel('Average Points per Game')
plt.legend()
plt.grid(True)
plt.show()
