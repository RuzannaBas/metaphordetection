import re
import csv
from datasets import load_dataset

test_file = r'data\VUAMC\test.txt'
output_file = r'\data\VUAMC.csv'

dataset = load_dataset("matejklemen/vuamc")
output_rows = []
with open(test_file, 'r', encoding='utf-8', newline="") as test:
    sentences = []
    for line in test:
        columns = line.strip().split('\t')
        sentence = columns[0]

        # sentence_clean = sentence.replace('"','')
        sentences.append(sentence.strip())

dataset_lookup = {}
for data in dataset['train']:
    sent = " ".join(data['words']) # Form sentences by joining the seperate words
    clean = re.sub(r"([‘“\"(])\s+", r"\1", sent) # Clean sentence by removing spaces after certain punctuation marks
    clean_sentence = re.sub(r'\s+([^\w\s])', r'\1', clean).strip() # Clean further by removing spaces before punctuation marks
    if clean_sentence not in dataset_lookup:
        dataset_lookup[clean_sentence] = data

# Annotate all words that contain a metaphor annotation with B-MET and others with O
for sentence in sentences:
    label = 0
    data = dataset_lookup.get(sentence)
    if data is None:
        print("NONE", sentence)
        continue
    tokens = data["words"]
    labels = ['O'] * len(tokens)
    
    for annotation in data["met_type"]:
        if '/met' in annotation["type"]:
            label = 1
            for index in annotation["word_indices"]:
                labels[index] = "B-MET"

    output_rows.append({
        "sentence": sentence,
        "tokens": tokens,
        "labels": labels,
        "label": label
        })
   
# Save as csv
with open(output_file, 'w', encoding='utf-8', newline="") as out:
    writer = csv.DictWriter(out, fieldnames=['sentence', 'tokens', 'labels', 'label'])
    writer.writeheader()
    writer.writerows(output_rows)