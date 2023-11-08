import pandas as pd

if __name__ == "__main__":
    # Read the CSV file into a DataFrame
    df = pd.read_csv('NFL Player Stats(1922 - 2022).csv')

    # Drop rows with missing values in the specified column
    df = df.dropna(subset=['Pos', 'Pts/G'])

    # Save the modified DataFrame back to a CSV file if needed
    df.to_csv('cleaned_file.csv', index=False)