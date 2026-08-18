from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import matplotlib.pyplot as plt
import torch
import numpy as np

def create_dataset():
    df = pd.read_csv(r"data\kankernl\kankernl_test.csv")
    df = df.drop(columns=["labels"])
    df = df.rename(columns={"sentence": "text", "metaphorical": "labels"})
    df = df[["text", "labels"]]
    dataset = Dataset.from_pandas(df)
    return dataset

def preprocess_function(data, tokenizer):
    return tokenizer(data["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    predictions = np.argmax(predictions, axis=1)

    acc = accuracy_score(labels, predictions)
    precision_class, recall_class, f1_class, _ = precision_recall_fscore_support(labels, predictions, average=None)
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(labels, predictions, average='macro')

    return {
        "accuracy": acc,
        "precision_lit": precision_class[0],
        "precision_met": precision_class[1],
        "precision_m": precision_m,
        "recall_lit": recall_class[0],
        "recall_met": recall_class[1],
        "recall_m": recall_m,
        "f1_lit": f1_class[0],
        "f1_met": f1_class[1],
        "f1_m": f1_m
    }

def confidence_boxplot(trainer, tokenized_dataset, data_name, model, seed, eval=False):
    predictions = trainer.predict(tokenized_dataset)

    logits = torch.tensor(predictions.predictions)
    probs = torch.softmax(logits, dim=1).numpy()
    labels = predictions.label_ids

    predicted_classes = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    data = {
        "Literal (correct)": confidence[(labels == 0) &  (predicted_classes == labels)],
        "Literal (wrong)" : confidence[(labels == 0) & (predicted_classes != labels)],
        "Metaphor (correct)": confidence[(labels == 1) &  (predicted_classes == labels)],
        "Metaphor (wrong)" : confidence[(labels == 1) & (predicted_classes != labels)]
    }

    fig, ax = plt.subplots(figsize=(10,6))
    ax.boxplot(data.values(), tick_labels=data.keys(), patch_artist=True)
    ax.set_ylabel("Confidence")
    ax.set_title(f"{model} model confidence per label and correctness ({data_name} seed {seed})")
    ax.set_ylim(0,1)
    plt.tight_layout()
    if eval:
        plt.savefig(f"eval/confidence_boxplot_{model}_{data_name}_{seed}")
    else:
        plt.savefig(f"{data_name}/confidence_boxplot_{data_name}_{seed}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    models = ['MOH-X', 'VUAMC', 'MMC', 'combi']
    seeds = [43, 44, 45, 46, 47]
    data = "kankernl"
    for model_name in models:
        
            print(f"Evaluating {model_name} model on {data} dataset")
            with open(f"eval/test_metrics_{model_name}_{data}.txt", "w", encoding="utf-8") as f:
                f.write(f"Evaluation {model_name} model on {data} dataset")
            dataset = create_dataset()
                    
            
            results = []
            for seed in seeds:
                path = f"Models/{model_name}/final_model_{model_name}_{seed}"
                model = AutoModelForSequenceClassification.from_pretrained(path)
                tokenizer = AutoTokenizer.from_pretrained(path)

                args = TrainingArguments(
                    per_device_eval_batch_size=8
                )

                trainer = Trainer(
                    model=model,
                    args=args,
                    processing_class=tokenizer,
                    compute_metrics=compute_metrics
                )

                tokenized_test = dataset.map(
                    lambda x:preprocess_function(x, tokenizer),
                    batched=True
                )

                result = trainer.evaluate(tokenized_test)

                result["seed"] = seed
                results.append(result)
                print(result)
                with open(f"eval/test_metrics_{model_name}_{data}.txt", "a", encoding="utf-8") as f:
                    f.write(f"Seed {seed}")
                    for metric, value in result.items():
                        f.write(f"{metric}: {value:.3f}\n")
                        print(f"{metric}: {value:.3f}")
                
                confidence_boxplot(trainer, tokenized_test, data, model_name, seed, eval=True)
                torch.cuda.empty_cache()

            with open(f"eval/test_metrics_{model_name}_{data}.txt", "a", encoding="utf-8") as f:
                f.write(f"Mean + Std")
                metrics = [k for k in results[0].keys() if k != "seed"]
                for metric in metrics:
                    values = np.array([r[metric] for r in results])
                    mean = np.mean(values)
                    std = np.std(values)
                    f.write(f"{metric}: {mean:.3f} ± {std:.3f}\n")
                    print(f"Mean {metric}: {mean:.3f} ± {std:.3f}")