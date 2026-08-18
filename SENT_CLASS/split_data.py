import sys
import random

def split_data():
    with open(CLEANED_FILE, 'r', encoding= "utf-8") as file:
        data = file.read()
    
    samples = data.split("\n")

    random.seed(42)
    random.shuffle(samples)

    train_split = int(len(samples) * 0.8)
    train_samples = samples[:train_split]
    test_samples = samples[train_split:]
    
    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    with open(TRAIN_FILE, 'w', encoding="utf-8") as train_file:
        train_file.write("\n".join(train_samples))
    with open(TEST_FILE, "w", encoding="utf-8") as test_file:
        test_file.write("\n".join(test_samples))

if __name__ == "__main__":
    dataset = sys.argv[1] 
    CLEANED_FILE = f'data/{dataset}/data.txt'
    TRAIN_FILE = f'data/{dataset}/train.txt'
    TEST_FILE = f'data/{dataset}/test.txt'
    split_data()