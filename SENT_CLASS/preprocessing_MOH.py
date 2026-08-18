import re

DATA_FILE = r"data/MOH-X/Data-metaphoric-or-literal.txt"
CLEANED_FILE = r"data/MOH-X/data.txt"

# Empty data.txt file if it exists already to ensure a clean file
open(CLEANED_FILE, "w").close()

# Loop through data and assign labelid for each sentence based on the label in the data
with open(DATA_FILE, 'r', encoding="utf-8") as data_file, open(CLEANED_FILE, 'w', encoding="utf-8") as cleaned_file:
    data = data_file.readlines()
    
    for line in data[1:-2]: # Skip heading and last 2 lines as it contains statistics on the data
        columns = line.strip().split('\t')
        sentence = columns[2] 
        label = columns[3]

        # Assign label id based on the given label and continue to next sentence if label is neither literal or metaphorical
        if label == "literal": 
            labelid = 0
        elif label == "metaphorical":
            labelid = 1
        else: 
            print(f"Unknown label {label} in: {sentence}")
            continue

        # Remove <b> and </b> from sentences which denote metaphors in the original data
        clean_sentence = re.sub(r'<.*?>', '', sentence)

        # Write data to data.txt
        cleaned_file.write(f'{clean_sentence}\t{label}\t{labelid}\n')

# Remove last empty line in file
with open(CLEANED_FILE, 'rb+') as cleaned_file:
    cleaned_file.seek(-2, 2)
    cleaned_file.truncate()

print(f"Saved {CLEANED_FILE}")
