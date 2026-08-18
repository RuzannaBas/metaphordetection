import argparse
import os
import csv
import numpy as np
import random
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
    print(f"True metaphor count: {t_count}, Predicted metaphor count: {p_count}")
    with open(f'{output_dir}/results4.txt', 'a') as f:
        f.write(f"True metaphor count: {t_count}, Predicted metaphor count: {p_count}")
        

def ask_with_knowledge(args, sentence, tokens):
    _model = args._model
    sentence_add_punch = sentence.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' "', '"').replace('  .', '.')

    system_prompt = '''You are a helpful AI assistant. Answer as concisely as possible.'''

    knowledge = '''- The source domain is the conceptual domain from which the metaphor is drawn. It is typically more concrete, tangible, and familiar.
                    - The target domain is used for the conceptual domain to which the metaphor is applied. It is typically more vague and abstract.
                    - Conceptual domains are sets of value meanings (presented using a list of concepts or a description of the members of the set) and are used to describe the set of concepts that can be represented within a data element. 
                    - In a conceptual metaphor, elements from the concrete source domain are used to understand, describe, or conceptualiz the target domain.
                    - For example, in "I've invested a lot of time in her", MONEY is the source domain and TIME is the target domain. The word "invested" uses a concept from the concrete domain of money to describe the more abstract domain of time.
                    - A word should be considered metaphorical when its meaning in the sentence invokes a concept or property from a source domain to describe or conceptualize something belonging to another domain.'''
    
    question = "Use the knowledge above to identify metaphorical tokens. The sentence '%s' contains a metaphor or other types of figurative language, such as similes, that are considered metaphorical here. Given the tokens '%s' return a list of the token or tokens that are used metaphorically. Return the tokens that are used metaphorically and other tokens that are connected to the metaphorically used token. Return only the tokens, do not give an explanation." %(sentence_add_punch, tokens)
        
    prompt = knowledge + question
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
            answer = answer.strip("[]")
            answer = [x.strip() for x in answer.split(",")]

        if isinstance(answer, tuple):
                    answer = list(answer)
        if isinstance(answer, str):
            answer = [x.strip() for x in answer.split()]
        answer.sort()
        answers_res.append(answer)
        answers.extend(answer)

    # Assign classification if all reponses are the same
    if all(answer == answers_res[0] for answer in answers_res):
        classification = answers_res[0]
        
    else: # Otherwise choose a random response
        classification = random.choice(answers_res)

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

    with open(f'{output_dir}/results4.txt', 'a') as f:
        f.write(report)
        f.write('\n')
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")


def main(args):
    now = datetime.now()
    print(now)
   
    dataloader = Data_Processor(args)
    if 'kankernl' in args.dataset:
        file = f'results/context/{args.dataset}/{args.repetition}/{args.dataset}3.csv'
        output_dir = f'{args.output_path}/context/{args.dataset}/{args.repetition}'
    else: 
        file = f'results/{args.dataset}/{args.repetition}/{args.dataset}3.csv'
        output_dir = f'{args.output_path}/{args.dataset}/{args.repetition}'

    output_file = f'{output_dir}/{args.dataset}4.csv'
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
            blog_id, sentence_id, full_blog, sentence, tokens, labels, true_label, final, prompt3, classification3, chosen_answer3, vote, words3 = sample
        else:
            sentence, tokens, labels, true_label, final, prompt3, classification3, chosen_answer3, vote, words3 = sample 

        labels = eval(labels)
        true_detailed.extend(labels)
        binary_labels = ['MET' if 'MET' in label else 'O' for label in labels]  
        true_binary.extend(binary_labels)
        # Only run knowledge based prompting if voting was needed in Step 3 (no unanimous answer)
        if vote == 'True':
            dialogs, answers_res, classification, words = ask_with_knowledge(args, sentence, tokens)   
            detailed_class, binary_class = clean_labels(classification)
            pred_detailed.extend(detailed_class)
            pred_binary.extend(binary_class)
        else:
            words = words3
            classification = eval(chosen_answer3)
            pred_detailed.extend(classification)
            binary_class = ['MET' if 'MET' in label else 'O' for label in classification]  
            pred_binary.extend(binary_class)
        
        if 'kankernl' in args.dataset:
            new_samples.append([blog_id, sentence_id, full_blog, sentence, tokens, labels, true_label, final, prompt3, classification3, chosen_answer3, vote, words3, dialogs, answers_res, classification, words])
            head = ['blog_id', 'sentence_id', 'full_blog', 'sentence', 'tokens', 'labels', 'true_label', 'final_label', 'prompt3', 'classification3', 'chosen_answer', 'voting', 'metaphorical_words3', 'prompt4', 'classification4', 'chosenanswer4', 'metaphorical_words_4']
            dataloader.write_dataset(output_file, head, new_samples)
        else:
            new_samples.append([sentence, tokens, labels, true_label, final, prompt3, classification3, chosen_answer3, vote, words3, dialogs, answers_res, classification, words])
            head = ['sentence', 'tokens', 'labels', 'true_label','final_label', 'prompt3', 'classification3', 'chosen_answer', 'voting', 'metaphorical_words3', 'prompt4', 'classification4', 'chosenanswer4', 'metaphorical_words4']
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



