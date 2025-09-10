import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

json_file_path = "results.json"

with open(json_file_path, "r") as f:
    data = json.load(f)

print(f"Total samples in dataset: {len(data)}")

train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")

train_file_path = "train_data.json"
test_file_path = "test_data.json"

with open(train_file_path, "w") as f:
    json.dump(train_data, f, indent=2)

with open(test_file_path, "w") as f:
    json.dump(test_data, f, indent=2)

print(f"Training data saved to {train_file_path}")
print(f"Testing data saved to {test_file_path}")

# Load test data
print("Loading test data...")
with open("test_data.json", "r") as f:
    test_data = json.load(f)

# Load IEMOCAP dataset
print("Loading IEMOCAP dataset...")
ds = load_dataset("AbstractTTS/IEMOCAP")
iemocap_df = pd.DataFrame(ds['train'])

# Filter for frustrated and angry emotions
print("Filtering emotions...")
filtered_df = iemocap_df[
    (iemocap_df['emotion'].str.lower() == 'frustrated') | 
    (iemocap_df['emotion'].str.lower() == 'angry')
]
print(f"Filtered to {len(filtered_df)} rows with angry/frustrated emotions")

# Function to find matches between IEMOCAP and our test data
def find_matches(test_data, filtered_df):
    matches = []
    metrics = [
        "gender", "EmoAct", "EmoVal", "EmoDom", "speaking_rate", 
        "pitch_mean", "pitch_std", "rms", "relative_db"
    ]
    
    # Map IEMOCAP columns to our test data columns if needed
    column_mapping = {
        "gender": "gender",  # Assuming gender is stored the same way
        # Add mappings for other columns if they have different names
    }
    
    # Convert test_data to DataFrame for easier manipulation
    test_df = pd.DataFrame(test_data)
    
    # Define tolerance for floating point comparisons
    tolerance = 0.01
    
    print("Finding matches...")
    for _, test_row in tqdm(test_df.iterrows(), total=len(test_df)):
        # Check if test_row has frustrated or angry emotion
        if test_row.get("major_emotion", "").lower() not in ["frustrated", "angry"]:
            continue
            
        for _, iemocap_row in filtered_df.iterrows():
            match = True
            
            # Check gender match (special case, might need different handling)
            if "gender" in test_row and "gender" in iemocap_row:
                # Handle potential different formats (Male/Female vs M/F)
                test_gender = test_row["gender"].lower()
                iemocap_gender = iemocap_row["gender"].lower()
                
                test_gender_norm = "male" if "male" in test_gender else "female"
                iemocap_gender_norm = "male" if "male" in iemocap_gender else "female"
                
                if test_gender_norm != iemocap_gender_norm:
                    match = False
                    continue
            
            # Check numerical metrics with tolerance
            for metric in metrics:
                if metric == "gender":
                    continue  # Already handled
                
                if metric in test_row and metric in iemocap_row:
                    test_val = test_row[metric]
                    iemocap_val = iemocap_row[metric]
                    
                    if isinstance(test_val, (int, float)) and isinstance(iemocap_val, (int, float)):
                        if abs(test_val - iemocap_val) > tolerance:
                            match = False
                            break
            
            if match:
                # Combine data from both sources
                combined = {**iemocap_row.to_dict(), **test_row.to_dict()}
                matches.append(combined)
    
    return matches

# Find matches
matches = find_matches(test_data, filtered_df)
print(f"Found {len(matches)} matches between test data and IEMOCAP dataset")

# Convert matches to DataFrame and save to CSV
if matches:
    matches_df = pd.DataFrame(matches)
    output_file = "iemocap_test_matches.csv"
    matches_df.to_csv(output_file, index=False)
    print(f"Matches saved to {output_file}")
else:
    print("No matches found. Trying with relaxed criteria...")
    
    # If no exact matches, we can create a version with just the filtered data
    output_file = "iemocap_angry_frustrated.csv"
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered IEMOCAP data (angry/frustrated only) saved to {output_file}")