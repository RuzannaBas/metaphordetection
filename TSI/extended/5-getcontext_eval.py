import argparse
import os
import csv
import numpy as np
import random
import string
import itertools
import pandas as pd
from ollama import chat
from datetime import datetime
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    accuracy_score
)
from tqdm import tqdm 

class Data_Processor:
    def __init__(self, args):
        self.data_root_path = os.path.join(args.data_dir, args.dataset)
        self.output_root_path = os.path.join(args.output_path, args.dataset, args.repetition)
    
    def get_dataset(self, file):
        examples = []
        with open(file, 'r', encoding='utf-8') as f:
            lines = csv.reader(f)
            next(lines)  # skip the headline
            for i, line in enumerate(lines):
                examples.append(line)
        return examples

    def write_dataset(self, file, head, examples):
        if not os.path.exists(self.output_root_path):
            os.makedirs(self.output_root_path)
        with open(file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(head)
            for line in examples:
                writer.writerow(line)

    def get_dataset_output(self, file):
        if not os.path.exists(os.path.join(self.output_root_path, file)):
            return []
        else:
            with open(os.path.join(self.output_root_path, file), 'r', encoding='utf-8') as f:
                lines = csv.reader(f)
                next(lines)  # skip the headline
                examples = []
                for i, line in enumerate(lines):
                    examples.append(line)
            return examples
        



def parse_option():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--vote_number', default=5, type=int)
    parser.add_argument('--dataset', default='MOH-X', type=str, help='dataset name')
    parser.add_argument('--output_path', default='results', type=str)
    parser.add_argument('--think', action="store_true", default=False) 
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--_model', default='qwen3', type=str, help='''qwen3''')
    parser.add_argument('--repetition', default='0', type=str)
    parser.add_argument('--vote_scaffolding', default=3, type=int)
    parser.add_argument('--nocontext', action="store_true", default=False)
    args, unparsed = parser.parse_known_args()
    return args

# Count number of ground truth metaphors that are detected by at least 1 token
def count_metaphors(ground_truth, prediction, output_dir):
    t_count = 0
    p_count = 0
    seen = False
    for i, label in enumerate(ground_truth):
        if label == 'B-MET':
            t_count += 1
            seen = False
            if prediction[i] == 'B-MET' or prediction[i] == 'I-MET':
                p_count += 1
                seen = True
        elif label == 'I-MET' and seen != True:
            if prediction[i] == 'B-MET' or prediction[i] == 'I-MET':
                p_count += 1
                seen = True
    with open(f'{output_dir}/results5.txt', 'a') as f:
        f.write(f"True metaphor count: {t_count}, Predicted metaphor count: {p_count}")
        

def ask(args, sentence, tokens, words):
    _model = args._model
    sentence_add_punch = sentence.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' "', '"').replace('  .', '.')

    system_prompt = '''You are a helpful AI assistant. Answer as concisely as possible.'''

    prompt = '''The sentence '%s' contains a metaphor or another type of figurative language. 
        Given the metaphorical tokens '%s', identify the complete metaphorical phrase.
        For each metaphorical expression:
        1. Identify the word or lexical unit that triggers the metaphor.
        2. Return only the tokens, do not give an explanation.
        3. Extend the annotation to the smallest phrase that fully expresses the metaphorical meaning.
        4. If the words that form the metaphorical phrase are not adjacent, include all words between them.
        5. Include articles, personal pronouns, and possessive pronouns when they are grammatically part of the metaphorical phrase.
        6. Do not include words that are not necessary for the metaphorical interpretation.
        Return the tokens belonging to the complete metaphorical phrase. Return only the tokens, without an explanation.
        Sentence: '%s'
        Tokens: '%s'
        ''' %(sentence_add_punch, words, sentence_add_punch, tokens)
        
    dialogs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
        ]

    answers_res = []
    answers = []
    classification = []
    for _ in range(args.vote_number):
        response = chat(
            model=_model,
            messages=dialogs,
            think=args.think,
            options={
                "temperature": args.temp,
                "num_predict": 256
            }
        )
        answer = response['message']['content'].replace('*', '')

        # Answer returned by prompt is often in list format, clean it if it's not
        try:
            answer = eval(answer)
        except:
            answer = answer.replace('\n', '').strip()

            answer = answer.replace('[', '')
            answer = answer.replace(']', '')
            answer = answer.replace('"', '')
            answer = answer.replace('\n', '')
   
            if ',' in answer:
                answer = [x.strip() for x in answer.split(",")]
            else:
                answer = [x.strip() for x in answer.split()]


        if isinstance(answer, tuple):
            answer = list(answer)
        if isinstance(answer, str):
            answer = [x.strip() for x in answer.split()]

        if any(isinstance(x, list) for x in answer):
            answer =  list(itertools.chain(*answer))
        answer.sort()
        answers_res.append(answer)
        answers.extend(answer)

    # Assign classification if all reponses are the same
    if all(answer == answers_res[0] for answer in answers_res):
        classification = answers_res[0]
        
    else:  # Get the tokens that are returned by all reponses
        set_answers = list(set(answers))
        count_answers = [answers.count(item) for item in set_answers]

        classification = [item for item, count in zip(set_answers, count_answers) if count >= 3]
    words = eval(words)
    classification.extend(words)

    # Label all tokens with O and relabel to MET if token is available in classification
    labels = ['O'] * len(eval(tokens))
    for i, token in enumerate(eval(tokens)):
        if any(token == word for word in classification):
            labels[i] = 'MET'

    return dialogs, answers_res, labels, classification

# Turn binary MET labels to BIO labelling 
def clean_labels(labels):
    clean = []
    binary = []
    prev = ''
    for label in labels:
        if label == 'MET':
            if prev == 'B-MET' or prev == 'I-MET':
                clean.append('I-MET')
                prev = 'I-MET'
            else:
                clean.append('B-MET')
                prev = 'B-MET'
            binary.append('MET')
        else:
            clean.append('O')
            binary.append('O')
            prev = 'O'

    return clean, binary
        
        
            
def evaluate(predict_list, ground_truth_list, output_dir):
    report = classification_report(np.array(ground_truth_list), np.array(predict_list), digits=4, zero_division=np.nan)
    precision = precision_score(np.array(ground_truth_list), np.array(predict_list), average='macro', zero_division=np.nan)
    recall = recall_score(np.array(ground_truth_list), np.array(predict_list), average='macro', zero_division=np.nan)
    f1 = f1_score(np.array(ground_truth_list), np.array(predict_list), average='macro', zero_division=np.nan)
    accuracy = accuracy_score(np.array(ground_truth_list), np.array(predict_list))

    print(f"Report: {report}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"Accuracy: {accuracy}")
    print()

    with open(f'{output_dir}/results5.txt', 'a') as f:
        f.write(f"Results Step 5 English Prompts dataset {args.dataset} Repetition {args.repetition}\n")
    with open(f'{output_dir}/results5.txt', 'a') as f:
        f.write(report)
        f.write('\n')
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")

    # return report, precision, recall, f1, accuracy

def main(args):
    now = datetime.now()
    print(now)
   
    dataloader = Data_Processor(args)
    if 'kankernl' in args.dataset:
        file = f'results/context/{args.dataset}/{args.repetition}/{args.dataset}4.csv'
        output_dir = f'{args.output_path}/context/{args.dataset}/{args.repetition}'
    else: 
        file = f'results/{args.dataset}/{args.repetition}/{args.dataset}4.csv'
        output_dir = f'{args.output_path}/{args.dataset}/{args.repetition}'

    output_file = f'{output_dir}/{args.dataset}5.csv'
    samples = dataloader.get_dataset(file)
    pred_detailed = []
    pred_binary = []
    true_detailed = []
    true_binary = []
    new_samples = []
    for sample_id in tqdm(range(len(samples))):
        full_blog = None
        dialogs = []
        answers_res= []
        sample  = samples[sample_id]

        if 'kankernl' in args.dataset:
            blog_id, sentence_id, full_blog, sentence, tokens, labels, true_label, final, prompt3, classification3, chosen_answer3, vote, words3, prompt4, classification4, chosen_answer4, words4 = sample
        else:
            sentence, tokens, labels, true_label, final, prompt3, classification3, chosen_answer3, vote, words3,  prompt4, classification4, chosen_answer4, words4 = sample 

        labels = eval(labels)
        true_detailed.extend(labels)
        binary_labels = ['MET' if 'MET' in label else 'O' for label in labels]  
        true_binary.extend(binary_labels)
        
        dialogs, answers_res, classification, words = ask(args, sentence, tokens, words4)   
        detailed_class, binary_class = clean_labels(classification)
        pred_detailed.extend(detailed_class)
        pred_binary.extend(binary_class)
       
        
        if 'kankernl' in args.dataset:
            new_samples.append([blog_id, sentence_id, full_blog, sentence, tokens, labels, true_label, final, prompt3, classification3, chosen_answer3, vote, words3, prompt4, classification4, chosen_answer4, words4, dialogs, answers_res, classification, words])
            head = ['blog_id', 'sentence_id', 'full_blog', 'sentence', 'tokens', 'labels', 'true_label', 'final_label', 'prompt3', 'classification3', 'chosen_answer', 'voting', 'metaphorical_words3', 'prompt4', 'classification4', 'chosenanswer4', 'metaphorical_words4', 'prompt5', 'classification5', 'chosenanswer5', 'metaphorical_words5']
            dataloader.write_dataset(output_file, head, new_samples)
        else:
            new_samples.append([sentence, tokens, labels, true_label, final, prompt3, classification3, chosen_answer3, vote, words3, prompt4, classification4, chosen_answer4, words4, dialogs, answers_res, classification, words ])
            head = ['sentence', 'tokens', 'labels', 'true_label','final_label', 'prompt3', 'classification3', 'chosen_answer', 'voting', 'metaphorical_words3', 'prompt4', 'classification4', 'chosenanswer4', 'metaphorical_words4', 'prompt5', 'classification5', 'chosenanswer5', 'metaphorical_words5']
            dataloader.write_dataset(output_file, head, new_samples)

    print("Evaluate Detailed")
    evaluate(pred_detailed, true_detailed, output_dir)
    print("Evaluate Binary")
    evaluate(pred_binary, true_binary, output_dir)
    print("Count seen metaphors")
    count_metaphors(true_detailed, pred_detailed, output_dir)
  

if __name__ == '__main__':
    args = parse_option()

    main(args)



