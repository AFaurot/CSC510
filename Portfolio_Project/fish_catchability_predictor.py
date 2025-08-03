import pandas as pd
import numpy as np
import glob
import os
from datetime import date
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk


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
# and the number of years since stocking.
# The function applies an exponential decay rate effect based on these factors to model population decline.
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


# Build classification targets for the training data
def classify_catchability(score):
    if score < 2:
        return 'Improbable'
    elif score < 10:
        return 'Low'
    elif score < 12:
        return 'Medium'
    elif score < 15.0:
        return 'High'
    else:
        return 'Very High'


# Function to launch the GUI for selecting species and county, and displaying the top 100 locations
def launch_gui(growth_rates, full_df, model):
    name_to_code = {v[0]: k for k, v in growth_rates.items()}
    species_names = sorted(name_to_code.keys())

    county_names = {
        'ADA': 'ADAMS', 'ALA': 'ALAMOSA', 'ARA': 'ARAPAHOE',
        'ARC': 'ARCHULETA', 'BAC': 'BACA', 'BEN': 'BENT',
        'BOU': 'BOULDER', 'BRM': 'BROOMFIELD', 'CHA': 'CHAFFEE',
        'CHE': 'CHEYENNE', 'CLE': 'CLEAR CREEK', 'CON': 'CONEJOS',
        'COS': 'COSTILLA', 'CRO': 'CROWLEY', 'CUS': 'CUSTER',
        'DEL': 'DELTA', 'DEN': 'DENVER', 'DOL': 'DOLORES',
        'DOU': 'DOUGLAS', 'EAG': 'EAGLE', 'ELB': 'ELBERT',
        'ELP': 'EL PASO', 'FRE': 'FREMONT', 'GAR': 'GARFIELD',
        'GIL': 'GILPIN', 'GRA': 'GRAND', 'GUN': 'GUNNISON',
        'HIN': 'HINSDALE', 'HUE': 'HUERFANO', 'JAC': 'JACKSON',
        'JEF': 'JEFFERSON', 'KIO': 'KIOWA', 'KIT': 'KIT CARSON',
        'LAK': 'LAKE', 'LAP': 'LA PLATA', 'LAR': 'LARIMER',
        'LAS': 'LAS ANIMAS', 'LIN': 'LINCOLN', 'LOG': 'LOGAN',
        'MES': 'MESA', 'MIN': 'MINERAL', 'MOF': 'MOFFAT',
        'MON': 'MONTROSE', 'MOR': 'MORGAN', 'MTZ': 'MONTEZUMA',
        'NMC': 'a New Mexico County', 'OTE': 'OTERO', 'OUR': 'OURAY',
        'PAR': 'PARK', 'PHI': 'PHILLIPS', 'PIT': 'PITKIN',
        'PRO': 'PROWERS', 'PUE': 'PUEBLO', 'RBL': 'RIO BLANCO',
        'RGR': 'RIO GRANDE', 'ROU': 'ROUTT', 'SAG': 'SAGUACHE',
        'SED': 'SEDGWICK', 'SNJ': 'SAN JUAN', 'SNM': 'SAN MIGUEL',
        'SUM': 'SUMMIT', 'TEL': 'TELLER', 'WAS': 'WASHINGTON',
        'WEL': 'WELD', 'YUM': 'YUMA'
    }
    # Create a mapping from county code to county name
    code_to_name = {k: v for k, v in county_names.items()}
    # Create a reverse mapping from county name to county code
    name_to_code = {v: k for k, v in county_names.items()}
    county_display_names = sorted(name_to_code.keys())
    # Define the order for catchability classes
    catchability_order = {
        'Improbable': 0,
        'Low': 1,
        'Medium': 2,
        'High': 3,
        'Very High': 4
    }

    # Function to handle the submit button click
    def on_submit(species_name, county_name):
        species_code = {v[0]: k for k, v in growth_rates.items()}[species_name]
        filtered_df = full_df[full_df['species_code'] == species_code].copy()
        # If "All" county is selected, do not filter by county
        if county_name != "All":
            county_code = name_to_code.get(county_name)
            filtered_df = filtered_df[filtered_df['county_code'] == county_code]
        # If no data matches the selected species and county, display a message
        if filtered_df.empty:
            output.delete("1.0", tk.END)
            output.insert(tk.END, "No matching stocking data found.\n")
            return
        # Prepare the input features for the model
        features = ['quantity', 'size_inches', 'growth_score', 'years_since_stocking', 'size_class']
        input_X = pd.get_dummies(filtered_df[features], columns=['size_class'])

        # Ensure all model features are present in the input data fill in missing features with 0
        for col in model.feature_names_in_:
            if col not in input_X.columns:
                input_X[col] = 0
        input_X = input_X[model.feature_names_in_]

        # Predict the catchability class using the trained model
        filtered_df['predicted_class'] = model.predict(input_X)
        # Map the predicted class to catchability rank for sorted display
        filtered_df['catchability_rank'] = filtered_df['predicted_class'].map(catchability_order)
        # Sort the filtered DataFrame by catchability rank and select the top 100 locations
        top_locations = filtered_df.sort_values(by='catchability_rank', ascending=False).head(100)
        # Clear the output text area
        output.delete("1.0", tk.END)
        # Insert the top locations into the output text area
        for _, row in top_locations.iterrows():
            county_display = code_to_name.get(row['county_code'], row['county_code'])
            output.insert(tk.END, f"{row['water_name']} ({county_display}), Year: {row['year']} -> {row['predicted_class']}\n")
    # Create the main GUI window
    root = tk.Tk()
    root.title("Fish Catchability Predictor")

    tk.Label(root, text="Select Species:").grid(row=0, column=0)
    species_cb = ttk.Combobox(root, values=species_names, width=30)
    species_cb.grid(row=0, column=1)
    species_cb.set(species_names[0])

    tk.Label(root, text="Select County:").grid(row=1, column=0)
    county_cb = ttk.Combobox(root, values=["All"] + county_display_names, width=30)
    county_cb.grid(row=1, column=1)
    county_cb.set("All")

    submit_btn = tk.Button(root, text="Find Best Locations", command=lambda: on_submit(species_cb.get(), county_cb.get()))
    submit_btn.grid(row=2, column=0, columnspan=2)

    output = tk.Text(root, height=15, width=60)
    output.grid(row=3, column=0, columnspan=2)

    root.mainloop()


# Main function to read CSV files, build scores for training data, train the model, and launch the GUI
def main():
    print("Starting Fish Catchability Predictor...\n")
    print("Creating labeled classification targets for training data...")
    current_year = date.today().year
    column_names = ['water_id', 'water_name', 'county_code', 'species_code', 'quantity', 'size']
    csv_files = glob.glob("fds_files_v2/*.csv")
    df_list = []
    # Ensure the path is correct and points to CSV files
    for file in csv_files:
        year = int(os.path.basename(file).split('_')[0])
        df = pd.read_csv(file, header=None, names=column_names)
        df['year'] = year
        df_list.append(df)
    # Concatenate all data frames into one
    stocking_df = pd.concat(df_list, ignore_index=True)

    # Data cleaning and preprocessing
    stocking_df['size'] = stocking_df['size'].str.replace('"', '', regex=False)
    stocking_df['size_inches'] = pd.to_numeric(stocking_df['size'], errors='coerce')
    stocking_df['size_class'] = stocking_df['size_inches'].apply(classify_stock_size)
    stocking_df['years_since_stocking'] = current_year - stocking_df['year']

    # Generate growth score and common name based on species code
    growth_rates = {
        'ARC': ['Arctic Char', 3],
        'BCF': ['Blue Catfish', 4],
        'BCR': ['Black Crappie', 6],
        'BGL': ['Bluegill', 6],
        'BHD': ['Bullhead Catfish', 5],
        'BRK': ['Brook Trout', 7],
        'CCF': ['Channel Catfish', 5],
        'CHI': ['Chinook Salmon', 1],
        'CRA': ['Crappie', 6],
        'CUT': ['Cutthroat Trout', 5],
        'CAR': ['CAR Creek Cutthroat', 5],
        'CR1': ['Colorado River Cutthroat "A" strain', 5],
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
        'BRD': ['Largemouth Bass Brood', 5],
        'LOC': ['Brown Trout', 5],
        'LXB': ['Tiger Trout', 8],
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
        'SPL': ['Splake', 5],
        'SXW': ['Wiper', 8],
        'TGM': ['Tiger Muskie', 6],
        'WAL': ['Walleye', 5],
        'WBA': ['White Bass', 7],
        'WCR': ['White Crappie', 6],
        'YPE': ['Yellow Perch', 5]
    }

    # Map species code to growth score for each row in the DataFrame
    stocking_df['growth_score'] = stocking_df['species_code'].map(lambda x: growth_rates.get(x, [None, None])[1])
    # Specify numeric columns to convert to numeric types
    numeric_columns = ['quantity', 'size_inches', 'years_since_stocking', 'growth_score']
    for col in numeric_columns:
        stocking_df[col] = pd.to_numeric(stocking_df[col], errors='coerce')

    # For each row, apply estimate_catchability function to calculate catchability score
    stocking_df['catchability_score'] = stocking_df.apply(
        lambda row: estimate_catchability(row['growth_score'], row['size_class'], row['years_since_stocking']), axis=1
    )
    # Apply catchability score labels to training data
    stocking_df['catchability_class'] = stocking_df['catchability_score'].apply(classify_catchability)
    print("Labeled classification targets created successfully!\n")
    # Now we have a DataFrame with all necessary columns for training the model!!
    # Next, we will train a Random Forest Classifier to predict catchability class based on the features
    print("Training the model...")
    features = ['quantity', 'size_inches', 'growth_score', 'years_since_stocking', 'size_class']
    X = pd.get_dummies(stocking_df[features], columns=['size_class'])
    y = stocking_df['catchability_class']

    # Random Forest Classifier model training random state set to 510 for reproducibility and CSC510!!!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=510)
    model = RandomForestClassifier(n_estimators=100, random_state=510)
    model.fit(X_train, y_train)

    print("Model training complete!\n")
    print("Model features:", model.feature_names_in_,"\n")
    print("Model classes:", model.classes_,"\n")
    # Evaluate the model feature importance rankings
    feature_importances = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    print("Feature importance:\n", feature_importances, "\n")
    print("Total Rows in full dataset:", len(stocking_df))

    # Launch the GUI to select species and county, and display top locations
    # Note that model.predict is in GUI function on_submit() see lines 134-167
    launch_gui(growth_rates, stocking_df, model)


if __name__ == "__main__":
    main()
