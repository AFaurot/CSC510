import pandas as pd
import glob
import os
from datetime import date
import numpy as np
import matplotlib.pyplot as plt


# Function to classify size at which fish were stocked based on size in inches
# Used later in estimate_catchability() as part of heuristic calculation
def classify_stock_size(size):
    if size < 1.0:
        return "fry"
    elif size < 3.5:
        return "fingerling"
    elif size < 6.5:
        return "juvenile"
    elif size < 9.0:
        return "sub-adult"
    else:
        return "adult"


# Function to estimate catchability score. It takes into account the growth score, size class of the fish
# and the number of years since stocking. The function applies an exponential decay rate effect based on these factors.
# TODO: This is a heuristic function and may need adjustments based on real-world data and testing.
def estimate_catchability(growth_score, size_class, years_since_stocking):

    if pd.isnull(growth_score) or pd.isnull(years_since_stocking):
        return 0

    # Larger fish more likely to survive stocking, and suffer less mortality from predation
    size_class_weights = {
        'fry': 0.4,
        'fingerling': 0.6,
        'juvenile': 0.8,
        'sub-adult': 0.9,
        'adult': 1.0
    }

    # Creates a time decay mortality effect to based on years since stocking and size class
    # Simulates angler pressure and natural mortality over time
    # Accounts for time to initialize decay based on how old the stocked fish was
    if size_class == 'fry':
        time_effect = np.exp(-0.2 * (years_since_stocking - 5)) if years_since_stocking > 5 else 1
    elif size_class == 'fingerling' or size_class == 'juvenile':
        time_effect = np.exp(-0.2 * (years_since_stocking - 4)) if years_since_stocking > 4 else 1
    elif size_class == 'sub-adult':
        time_effect = np.exp(-0.3 * (years_since_stocking - 3)) if years_since_stocking > 3 else 1
    else:
        time_effect = np.exp(-0.35 * (years_since_stocking - 2)) if years_since_stocking > 3 else 1

    # If fish is stocked at catchable size, normalize growth score to 5
    if growth_score != 5 and size_class == 'adult':
        growth_score = 5

    # If fish is stocked at adult size, and only one year since stocking very high probability to be catchable
    if years_since_stocking <= 1 and size_class == 'adult':
        return round(15 * time_effect, 2)
    # If fish is stocked at adult size, and two years since stocking, high probability to be catchable
    elif years_since_stocking <= 2 and size_class == 'adult':
        return round(12 * time_effect, 2)
    # Boost score for sub-adult fish after two years by boost_rate (20 percent)
    elif years_since_stocking == 2 and size_class == 'sub-adult':
        boost_rate = 1.2
        return round(growth_score * years_since_stocking * size_class_weights.get(size_class, 0.5)
                     * time_effect * boost_rate, 2)
    # Calculate catchability score for other cases
    else:
        return round(growth_score * years_since_stocking * size_class_weights.get(size_class, 0.5) * time_effect, 2)


def main():

    # get the current year
    current_year = date.today().year

    # Name the columns based on the expected structure of the CSV files
    column_names = ['water_id', 'water_name', 'county_code', 'species_code', 'quantity', 'size']
    csv_files = glob.glob("fds_files_v2/*.csv")# Read all the CSV files into one data frame

    df_list = []

    # Ensure the path is correct and points to CSV files
    for file in csv_files:
        year = int(os.path.basename(file).split('_')[0])  # Extract year from filename
        df = pd.read_csv(file, header=None, names=column_names)
        # Append the year to the data frame
        df['year'] = year
        df_list.append(df)

    # Concatenate all data frames into one
    stocking_df = pd.concat(df_list, ignore_index=True)
    # Remove quote marks and convert to float
    stocking_df['size'] = stocking_df['size'].str.replace('"', '', regex=False)
    stocking_df['size_inches'] = pd.to_numeric(stocking_df['size'], errors='coerce')
    # Classify the size at which fish were stocked
    stocking_df['size_class'] = stocking_df['size_inches'].apply(classify_stock_size)
    stocking_df['years_since_stocking'] = current_year - stocking_df['year']


    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(stocking_df.head())
    print(stocking_df.tail())

    # Create growth rate heuristic, Growth scores reflect how quickly a species reaches catchable size
    growth_rates = {
        'ARC': ['Arctic Char', 3],
        'BCF': ['Blue Catfish', 4],
        'BCR': ['Black Crappie', 6],
        'BGL': ['Bluegill', 6],
        'BHD': ['Bullhead Catfish', 5],
        'BRK': ['Brook Trout', 8],
        'CCF': ['Channel Catfish', 5],
        'CHI': ['Chinook Salmon', 1],
        'CRA': ['Crappie', 6],
        'CUT': ['Cutthroat Trout', 5],
        'CAR': ['CAR Creek Cutthroat', 5],
        'CR1': ['Colorado River Cutthroat "A" strain', 5],
        'CRC': ['Colorado River Cutthroat', 5],
        'GBN': ['Greenback Cutthroat', 4],
        'PPN': ['Pikes Peak Cutthroat', 4],
        'RGN': ['Rio Grande Cutthroat', 4],
        'RXN': ['RXC Cross Cutthroat', 5],
        'SRN': ['Snake River Cutthroat', 5],
        'TRP': ['Trapper Creek Cutthroat', 5],
        'WEM': ['Weminuche Cutthroat', 4],
        'YSN': ['Yellowstone Cutthroat', 4],
        'NAN': ['Nanita Cutthroat', 4],
        'NAV': ['Navajo Cutthroat', 4],
        'FLC': ['Flathead Catfish', 3],
        'GOL': ['Golden Trout', 4],
        'GRA': ['Grayling', 3],
        'HBG': ['Hybrid Bluegill', 7],
        'KOK': ['Kokanee Salmon', 6],
        'LMB': ['Largemouth Bass', 5],
        'BRD': ['Largemouth Bass Brood', 10],
        'LOC': ['Brown Trout', 5],
        'LXB': ['Tiger Trout', 9],
        'MAC': ['Mackinaw Trout', 2],
        'MSK': ['Muskellunge', 2],
        'NPK': ['Northern Pike', 5],
        'RBT': ['Rainbow Trout', 5],
        'ARR': ['Rainbow Trout - Arlee', 5],
        'BEL': ['Rainbow Trout - Bellaire', 5],
        'RBF': ['Rainbow Trout - Brood', 5],
        'CRR': ['Rainbow Trout - Colorado River', 5],
        'ELR': ['Rainbow Trout - Eagle Lake', 5],
        'ERW': ['Rainbow Trout - Erwin', 5],
        'GRR': ['Rainbow Trout - Gunnison River', 5],
        'HOF': ['Rainbow Trout - Hofer', 6],
        'HXH': ['Rainbow Trout - Hofer x Harrison', 6],
        'HXC': ['Rainbow Trout - Hofer x Colorado River', 6],
        'HHC': ['Rainbow Trout - Hofer Harrison Cutthroat', 6],
        'HHN': ['Rainbow Trout - Hofer Harrison Snake River', 6],
        'FLR': ['Rainbow Trout - Fish Lake', 5],
        'RBM': ['Rainbow Trout - Federal Mitigation', 5],
        'STH': ['Rainbow Trout - Steelhead', 4],
        'HXG': ['Rainbow Trout - Hofer x Gunnison River', 6],
        'PRR': ['Rainbow Trout - Psychrophilum Resistant', 5],
        'RSF': ['Redeared Sunfish', 6],
        'SAG': ['Saugeye', 6],
        'SBS': ['Striped Bass', 2],
        'SGR': ['Sauger', 5],
        'SMB': ['Smallmouth Bass', 6],
        'LKS': ['Smallmouth Bass - Lake Strain', 6],
        'SNF': ['Sunfish - Green', 6],
        'SPB': ['Spotted Bass', 6],
        'SPE': ['Sacramento Perch', 4],
        'SPL': ['Splake', 4],
        'SXW': ['Wiper', 9],
        'TGM': ['Tiger Muskie', 6],
        'WAL': ['Walleye', 5],
        'WBA': ['White Bass', 8],
        'WCR': ['White Crappie', 6],
        'YPE': ['Yellow Perch', 5]
    }
    stocking_df['growth_score'] = stocking_df['species_code'].map(lambda x: growth_rates.get(x, [None, None])[1])
    numeric_columns = ['quantity', 'size_inches', 'years_since_stocking', 'growth_score']
    # Convert numeric columns to float, coercing errors to NaN
    for col in numeric_columns:
        if col in stocking_df.columns:
            stocking_df[col] = pd.to_numeric(stocking_df[col], errors='coerce')
    stocking_df['growth_score'] = pd.to_numeric(stocking_df['growth_score'], errors='coerce')
    stocking_df['catchability_score'] = stocking_df.apply(
        lambda row: estimate_catchability(
            row['growth_score'],
            row['size_class'],
            row['years_since_stocking']
        ), axis=1
    )
    lowest_catchability = stocking_df['catchability_score'].min()
    highest_catchability = stocking_df['catchability_score'].max()
    average_catchability = stocking_df['catchability_score'].mean()
    print(stocking_df.head())
    print(stocking_df.tail())
    print("Lowest catchability score:", lowest_catchability)
    print("Average catchability score:", average_catchability)
    print("Highest catchability score:", highest_catchability)
    # Identify rows where catchability_score is 0
    zero_catchability = stocking_df[stocking_df['catchability_score'] == 0]
    print("Rows with catchability_score == 0:")
    print(zero_catchability)

    num_nan_county_codes = stocking_df['county_code'].isna().sum()
    print("Number of NaN county_code values:", num_nan_county_codes)

    nan_county_code_rows = stocking_df[stocking_df['county_code'].isna()]
    print("Rows with NaN county_code:")
    print(nan_county_code_rows)
    nan_county_code_rows.to_csv('nan_county_code_rows.csv', index=False)

    invalid_species_code_rows = stocking_df[
        ~(
                stocking_df['species_code'].astype(str).str.match(r'^[A-Z]{3}$') |
                (stocking_df['species_code'] == 'CR1')
        )
    ]
    print("Rows with invalid species_code:")
    print(invalid_species_code_rows)
    invalid_species_code_rows.to_csv('invalid_species_code_rows.csv', index=False)

    # Plot histogram of catchability_score
    plt.figure(figsize=(8, 5))
    plt.hist(stocking_df['catchability_score'].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Catchability Score')
    plt.xlabel('Catchability Score')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == "__main__":
    main()
