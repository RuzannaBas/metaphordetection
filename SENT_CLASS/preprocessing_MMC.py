import pandas as pd

DATA_FILE = r"data/MMC/annotated_mmc(2).csv"
CLEANED_FILE = r"data/MMC/data.txt"

df = pd.read_csv(DATA_FILE)

# Empty data.txt file if it exists already to ensure a clean file 
open("data/MMC/data.txt", "w").close()

# Loop through the dataframe and save the sentence, metaphor label and labelid for each sentence based on the label and metaphor in the data
with open(CLEANED_FILE, 'w', encoding="utf-8") as f:
    for i in range(len(df)):
        sentence = df['Sentence'][i] # Get sentence
        goldlabel = df['GoldLabel'][i] # Get given goldlabel, this can be either 'yes', 'no' or 'tie'
        metaphor = df['Metaphor'][i] # Get the underlying metaphor assigned to the sentence, e.g. CELLS_ARE_HUMAN_BEINGS

        if goldlabel == 'no': # Sentence is literal 
            label = 'literal'
        elif goldlabel == 'yes': # Sentence is a metaphor
            label = 'metaphorical'
        elif goldlabel == 'Tie': # Annotators did not reach a definitive conclusion on whether the sentence contains a metaphor or not
            if pd.isna(metaphor): # If no underlying metaphor is given, we assume the sentence does not contain a metaphor and rewrite the goldlabel to literal
                label = 'literal'
            else: # If an underlying metaphor is given, we assume the sentence contains a metaphor, and rewrite the goldlabel to metaphorical
                label = 'metaphorical'
        
        # Assign labelid based on the given label
        if label == 'metaphorical':
            labelid = 1
        else:
            labelid = 0
        
        # Write data to data.txt
        f.write(f"{sentence}\t{label}\t{labelid}\n")

print(f"Saved {CLEANED_FILE}")