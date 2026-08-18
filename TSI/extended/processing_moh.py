import re
import csv

test_file = r'data\MOH-X\test.txt'
original_file = r'data\MOH-X\Data-metaphoric-or-literal.txt'
output_file = r'data\MOH-X.csv'

# Split sentence into tokens, annotate all words within <b> </b> with B-MET and others with O
def annotate_sentence(row,total):
    sentence = row['sentence'].strip()
    print(sentence)
    tokens = []
    labels = []
    parts = re.split(r"(<b>.*?</b>)", sentence)
    for part in parts:
        if part.startswith("<b>") and part.endswith("</b>"):
            met = part[3:-4]
            part_tokens = re.findall(r"\w+|[^\w\s]", met)
            for token in part_tokens:
                tokens.append(token)
                if re.match(r"^\w+$", token) and total ==1:
                    labels.append('B-MET')
                else: 
                    labels.append('O')
        else:
            part_tokens = re.findall(r"\w+|[^\w\s]", part)
            tokens.extend(part_tokens)
            labels.extend(['O'] * len(part_tokens))
    return tokens, labels

output_rows = []
with open(test_file, 'r', encoding='utf-8', newline="") as test:
    sentences = []
    for line in test:
        columns = line.strip().split('\t')
        sentence = columns[0]

        sentence_clean = sentence.replace('"','')
        sentences.append(sentence_clean.strip())

with open(original_file, 'r', encoding='utf-8', newline="") as original:
    reader = csv.DictReader(original, delimiter="\t")
    next(reader)
    rows = list(reader)
    rows = rows[:-2]

    for sentence in sentences:
        for row in rows:
            original_sentence = row["sentence"].strip()
            if row['class'] == 'metaphorical': 
                label = 1
            else:
                label = 0

            # Remove <b> </b> tags   
            original_sentence = re.sub(r"</?b>", "", original_sentence)

            # Save cleaned sentence, tokens, token-level labels and sentence-level label
            if sentence.strip() == original_sentence:
                tokens, labels = annotate_sentence(row,label)
                output_rows.append({
                    "sentence": sentence,
                    "tokens": tokens,
                    "labels": labels,
                    "label": label
                    })
                break
# Save as csv
with open(output_file, 'w', encoding='utf-8', newline="") as out:
    writer = csv.DictWriter(out, fieldnames=['sentence', 'tokens', 'labels', 'label'])
    writer.writeheader()
    writer.writerows(output_rows)