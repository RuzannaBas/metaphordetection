import csv

test_file = r'data\MMC\test.txt'
output_file = r'data\MMC.csv'
output_rows = []

with open(test_file, 'r', encoding='utf-8') as test:
    for line in test:
        print(repr(line))
        columns = line.split('\t')
        sentence = columns[0]
        label = columns[2]
        output_rows.append({
            "sentence": sentence,
            "label": int(label)
        })

# Save MMC test file as csv
with open(output_file, 'w', encoding='utf-8', newline="") as out:
        writer = csv.DictWriter(out, fieldnames=['sentence', 'label'])
        writer.writeheader()
        writer.writerows(output_rows)