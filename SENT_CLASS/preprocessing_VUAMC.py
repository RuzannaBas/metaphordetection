import re
from datasets import load_dataset


ds = load_dataset("matejklemen/vuamc")
CLEANED_FILE = r"data/VUAMC/data.txt"
# Empty data.txt file if it exists already to ensure a clean file
open(CLEANED_FILE, "w").close()
for data in ds['train']:
    sentence = " ".join(data['words']) # Form sentences by joining the seperate words
    clean = re.sub(r"([‘“\"(])\s+", r"\1", sentence) # Clean sentence by removing spaces after certain punctuation marks
    clean_sentence = re.sub(r'\s+([^\w\s])', r'\1', clean) # Clean further by removing spaces before punctuation marks
    
    meta_data = data["meta"]
    vocal = [['gap/unclear'], ['vocal/laugh'], ['gap/unclear', 'vocal/laugh'], ['gap/unclear', 'pause'], ['gap/unclear', 'pause', 'gap/unclear'], ['vocal/cough'], ['gap/unclear', 'gap/unclear'], ['vocal/laugh', 'gap/unclear'], ['vocal/laugh', 'pause'], ['pause'], ['vocal/clears throat'], ['incident/baby talk', 'pause'], ['incident/playing with baby', 'incident/noise'], ['vocal/laugh', 'pause', 'gap/unclear'], ['vocal/sneeze']]
    
    if meta_data in vocal: # Remove sentences that are empty based on their meta data
        continue
        
    # Check met_type and assign the according label
    if data['met_type'] == []: # Sentence is literal if met_type is empty
        label = "literal"
        labelid = 0
    elif any('met' in item for item in data): # Sentence is metaphorical if any type of metaphor is mentioned in met_type
        label = "metaphorical"
        labelid = 1
    else: 
        print(f"Unknown label in {data}")

    with open(CLEANED_FILE, "a", encoding="utf-8") as f:
        f.write(f"{clean_sentence}\t{label}\t{labelid}\n")

print(f"Saved {CLEANED_FILE}")



