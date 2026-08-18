import sys
import torch
import torch.nn as nn
import random
import optuna
import argparse
import numpy as np
import matplotlib.pyplot as plt
from optuna.trial import TrialState
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from datasets import Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Get data and return labels and sentence
def read_data_file(path):
    sentences, labels, = [], []
    with open(path, encoding="utf-8") as file:
        for line in file:
            columns = line.strip().split('\t')
            sentences.append(columns[0])
            labels.append(int(columns[2]))

        return {"labels": labels, "text": sentences}

# Create a dataset using the train and test split data
def create_dataset(data):
    train_data = read_data_file(f"{data}/train.txt")
    test_data = read_data_file(f"{data}/test.txt")

    dataset = DatasetDict({
        "train": Dataset.from_dict(train_data),
        "test": Dataset.from_dict(test_data)
    })

    return dataset

# Tokenize text
def preprocess_function(data, tokenizer):
    return tokenizer(data["text"], truncation=True)

# Compute and return evaluation metrics
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

# Weighted trainer to handle imbalanced data
class WeightedTrainer(Trainer): 
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        weight = self.class_weights.to(logits.device)

        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# Creat folds in training data to enable cross-validation
def make_folds(dataset):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = np.array(dataset["labels"])
    folds = []

    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        train_fold = dataset.select(train_idx)
        val_fold = dataset.select(val_idx)
        folds.append((train_fold, val_fold))
     
    return folds

# Train function used for HPO
def objective(folds, data, tokenizer, seed, trial=None):
    set_seed(seed)
    id2label = {0: 'literal', 1: 'metaphorical'}
    label2id = {'literal': 0, 'metaphorical': 1}

    # Search space for HPO
    lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)

    scores = []

    # Train for each fold and save results
    for i, (train_fold, val_fold) in enumerate(folds):
        # Set class_weights to handle imbalanced data
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_fold['labels']),
            y=train_fold['labels']
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-multilingual-uncased', num_labels=2, id2label=id2label, label2id=label2id,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )

        args = TrainingArguments(
            num_train_epochs=5,
            learning_rate = lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=weight_decay,
            eval_strategy="epoch",
            save_strategy="no",
            logging_strategy="epoch",
            seed=seed,
            data_seed=seed
        )

        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_fold,
            eval_dataset=val_fold,
            processing_class=tokenizer,
            class_weights=class_weights,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        result = trainer.evaluate()
        with open(f"{data}/log.txt", 'a', encoding="utf-8") as f:
            f.write(f"Trial:{trial.number}, Fold:{i}, result: {result}")
        scores.append(result["eval_f1_m"])
        del trainer
        del model
        torch.cuda.empty_cache()

    return np.mean(scores)

# Train best configuration from HPO on the full training set using given seed and save model
def train_final(train, best_params, data, tokenizer, seed):
    set_seed(seed)

    id2label = {0: 'literal', 1: 'metaphorical'}
    label2id = {'literal': 0, 'metaphorical': 1}

    class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train['labels']),
            y=train['labels']
    )

    model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-multilingual-uncased', num_labels=2, id2label=id2label, label2id=label2id,
            hidden_dropout_prob=best_params['dropout'],
            attention_probs_dropout_prob=best_params['dropout']
        )
    
    args = TrainingArguments(
            num_train_epochs=4,
            learning_rate = best_params['learning_rate'],
            per_device_train_batch_size=best_params['per_device_train_batch_size'],
            weight_decay=best_params['weight_decay'],
            eval_strategy="no",
            logging_strategy="epoch",
            seed=seed,
            data_seed=seed
        )
    
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train,
        processing_class=tokenizer,
        class_weights=class_weights,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(f"./{data}/final_model_{data}_{seed}")

    return trainer
    
# Plot boxplot of model confidence per label and correctness
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
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action='store_true')
    parser.add_argument("--final", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--data", type=str, help='Dataset that is used for training, MOH-X, MMC, VUAMC, combi')
    args = parser.parse_args()


    if args.worker or args.final:
        data= args.data
        dataset = create_dataset(data)
        train_dataset = dataset["train"].shuffle(seed=42)
        test_dataset = dataset["test"]

        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')

        tokenized_train = train_dataset.map(
            lambda x:preprocess_function(x, tokenizer),
            batched=True
        )
        tokenized_test = test_dataset.map(
            lambda x:preprocess_function(x, tokenizer),
            batched=True
        )

        folds = make_folds(tokenized_train) # create folds for cross-validation

        storage = JournalStorage(JournalFileBackend(f"./{data}/journal.log")) # store hpo results 
        
        nr_trials = 100 # number of trials for HPO

    # run worker for hpo
    if args.worker:
        seed = 42
        while True:
            study = optuna.create_study(study_name=f"{data}_study", storage=storage, load_if_exists=True, direction="maximize")
            completed = len(study.get_trials(states=[TrialState.COMPLETE])) + len(study.get_trials(states=[TrialState.RUNNING])) + len(study.get_trials(states=[TrialState.WAITING]))
            if completed >= nr_trials:
                break
        
            study.optimize(lambda trial: objective(folds, data, tokenizer, seed, trial), n_trials=1)
        sys.exit()

    # Train final configuration on 5 different seeds and save model
    if args.final:
        study = optuna.load_study(
            study_name=f"{data}_study",
            storage=storage,
        )
        print("Best score:", study.best_value)
        print("Best parameters", study.best_params)

        seeds = [43, 44, 45, 46, 47]
        results = []
        with open(f"{data}/test_metrics.txt", "w", encoding="utf-8") as f:
            f.write(f"Best params:{study.best_params}")

        for seed in seeds:
            print(f"Final training with seed {seed}")
            trainer = train_final(tokenized_train, study.best_params, data, tokenizer, seed)
            result = trainer.evaluate(tokenized_test)
            result["seed"] = seed
            results.append(result)

            with open(f"{data}/test_metrics.txt", "a", encoding="utf-8") as f:
                f.write(f"Seed {seed}")
                for metric, value in result.items():
                    f.write(f"{metric}: {value}\n")
            
            confidence_boxplot(trainer, tokenized_test, data, data, seed)
            torch.cuda.empty_cache()


        with open(f"{data}/test_metrics.txt", "a", encoding="utf-8") as f:
                f.write(f"Mean + Std")
                metrics = [k for k in results[0].keys() if k != "seed"]
                for metric in metrics:
                    values = np.array([r[metric] for r in results])
                    mean = np.mean(values)
                    std = np.std(values)
                    f.write(f"{metric}: {mean} ± {std}\n")
        
    # Run evaluation on all models and the three English test sets
    if args.eval:
        models = ['MOH-X', 'VUAMC', 'MMC', 'combi']
        datasets = ['MOH-X', 'VUAMC', 'MMC']
        seeds = [43, 44, 45, 46, 47]

        for model_name in models:
            for data in datasets:
                print(f"Evaluating {model_name} model on {data} dataset")
                with open(f"eval/test_metrics_{model_name}_{data}.txt", "w", encoding="utf-8") as f:
                    f.write(f"Evaluation {model_name} model on {data} dataset")
                dataset = create_dataset(data)
                test_dataset = dataset["test"]
                 
                tokenized_test = test_dataset.map(
                    lambda x:preprocess_function(x, tokenizer),
                    batched=True
                )
                results = []
                for seed in seeds:
                    path = f"{model_name}/final_model_{model_name}_{seed}"
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

                    result = trainer.evaluate(tokenized_test)

                    result["seed"] = seed
                    results.append(result)

                    with open(f"eval/test_metrics_{model_name}_{data}.txt", "a", encoding="utf-8") as f:
                        f.write(f"Seed {seed}")
                        for metric, value in result.items():
                            f.write(f"{metric}: {value:.3f}\n")
                    
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