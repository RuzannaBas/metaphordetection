import argparse
import os
import csv
import numpy as np
import random
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
    parser.add_argument('--nocontext', action="store_true", default=False) 
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--_model', default='qwen3', type=str, help='''qwen3''')
    parser.add_argument('--repetition', default='0', type=str)
    parser.add_argument('--vote_scaffolding', default=3, type=int)
    args, unparsed = parser.parse_known_args()
    return args

def ask(args, sentence, nocontext, full_blog=None):
    _model = args._model
    
    classification = -1 # final metaphor classification
    counter = 0 

    no = open(f'{args.data_dir}/no.txt').read().splitlines() # rules for mapping LLM response to literal
    yes = open(f'{args.data_dir}/yes.txt').read().splitlines() # rules for mapping LLM reponse to metaphor

    sentence_add_punch = sentence.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' "', '"').replace('  .', '.')

    system_prompt = '''You are a helpful AI assistant. Answer as concisely as possible.'''

    knowledge = '''1. Metaphors reflect underlying conceptual mappings between a source domain and a target domain. These mappings allow people to understand abstract concepts in terms of more concrete and familiar experiences. \n
            2. There are multiple types of metaphors: structural metaphors, orientational metaphors, ontological metaphors, container metaphors, personification. \n
            3. Structural metaphors: An abstract concept is understood in terms of a more concrete or familiar domain. For example, “I have come a long way” conceptualizes LIFE as a JOURNEY. \n
            4. Orientational metaphors: An abstract state or concept is described using spatial direction or movement, such as up/down, in/out, or rising/falling. For example, “My mood went up” conceptualizes positive emotion as UP. \n
            5. Ontological metaphors: An abstract concept, feeling, event, or state is treated as a concrete object or substance that can be carried, increased, lost, or otherwise interacted with. For example, “I carry a lot of sadness” treats SADNESS as a physical object. \n
            6. Container metaphors: An abstract state, situation, or experience is conceptualized as a bounded space or container that someone can be in, out of, full of, or come out of. For example, “I am in a difficult period.” \n
            7. Personification: A non-human or abstract concept is described as having human abilities, intentions, or actions. For example, “The disease is chasing me.” \n'''

    if nocontext:
        question = 'Taking the knowledge provided above into account, please answer the following question:\n Given the sentence "%s", does the sentence contain a metaphor?' %(sentence_add_punch)
    else:
        question = 'Taking the knowledge provided above into account, please answer the following question:\n Given the blog "%s", does the sentence "%s" contain a metaphor?' %(full_blog, sentence_add_punch)

    prompt = knowledge + question

    dialogs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
        ]

    answers_res = []
    for _ in range(args.vote_scaffolding):
        response = chat(
            model=_model,
            messages=dialogs,
            think=args.think,
            options={
                "temperature": args.temp
            }
        )
        answer = response['message']['content'].replace('*', '')
        answers_res.append(answer)
    # Randomly choose an answer given by the LLM, repeat if unsure
    while classification == -1 and counter < args.vote_scaffolding:
        final_answer = random.choice(answers_res)
        if any(x in final_answer for x in yes):
            classification = 1
        elif any(x in final_answer for x in no):
            classification = 0
        else:
            counter += 1
            print("Unsure,", final_answer)
            with open('unsure.txt', 'a') as f:
                f.write(f"{answer} \n")

    return dialogs, answers_res, final_answer, classification

def main(args):
    now = datetime.now()
    print(now)
   
    dataloader = Data_Processor(args)
    if 'kankernl' in args.dataset:
        nocontext = args.nocontext
        file = f'results/context/{args.dataset}/{args.repetition}/{args.dataset}.csv'
        output_dir = f'{args.output_path}/context/{args.dataset}/{args.repetition}'
    else: 
        nocontext = True
        file = f'results/{args.dataset}/{args.repetition}/{args.dataset}.csv'
        output_dir = f'{args.output_path}/{args.dataset}/{args.repetition}'

    output_file = f'{output_dir}/{args.dataset}2.csv'
    samples = dataloader.get_dataset(file)
    new_samples = []
    for sample_id in tqdm(range(len(samples))):
        classification = -1
        full_blog = None
        dialogs = []
        answers_res= []
        final_answer = []
        sample  = samples[sample_id]

        if 'kankernl' in args.dataset:
            blog_id, sentence_id, full_blog, sentence, tokens, labels, label, dialogs1, answers_res1, metaphor_count, literal_count, final1, final = sample
        elif 'MMC' in args.dataset:
            sentence, label, dialogs1, answers_res1, metaphor_count, literal_count, final1, final = sample
        else:
            sentence, tokens, labels, label, dialogs1, answers_res1, metaphor_count, literal_count, final1, final = sample 

        # Prompt LLM with additional knowledge if LLMs did not agree unanimously in Step 1
        if int(metaphor_count) != args.vote_number and int(literal_count) != args.vote_number:
            dialogs, answers_res, final_answer, classification = ask(args, sentence, nocontext, full_blog)

        # Use assigned label from Step 1 for unanimous cases and if the additional knowledge prompt did not result in a label
        if classification == -1:
            final = final1
        else:
            final = classification

    
        # Save data
        if 'kankernl' in args.dataset:
            new_samples.append([blog_id, sentence_id, full_blog, sentence, tokens, labels, label, dialogs1, answers_res1, metaphor_count, literal_count, final1, dialogs, answers_res, final_answer, classification, final])
            head = ['blog_id', 'sentence_id', 'full_blog', 'sentence', 'tokens', 'labels', 'true_label', 'prompt1', 'classification', 'metaphor_count', 'literal_count', 'label1', 'prompt2', 'classification2', 'chosen_answer', 'label2', 'final_label']
            dataloader.write_dataset(output_file, head, new_samples)
        elif 'MMC' in args.dataset:
            new_samples.append([sentence, label, dialogs1, answers_res1, metaphor_count, literal_count, final1, dialogs, answers_res, final_answer, classification, final])
            head = ['sentence', 'true_label', 'prompt1', 'classification', 'metaphor_count', 'literal_count', 'label1', 'prompt2', 'classification2', 'chosen_answer', 'label2', 'final_label']
            dataloader.write_dataset(output_file, head, new_samples)
        else:
            new_samples.append([sentence, tokens, labels, label, dialogs1, answers_res1, metaphor_count, literal_count, final1, dialogs, answers_res, final_answer, classification, final])
            head = ['sentence', 'tokens', 'labels', 'true_label', 'prompt1', 'classification', 'metaphor_count', 'literal_count', 'label1', 'prompt2', 'classification2', 'chosen_answer', 'label2', 'final_label']
            dataloader.write_dataset(output_file, head, new_samples)

    # Evaluate sentence classification
    data = pd.read_csv(output_file)
    ground_truth_list = data['true_label'].tolist()
    predict_list = data['final_label'].tolist()
    report = classification_report(np.array(ground_truth_list), np.array(predict_list), digits=4)
    precision = precision_score(np.array(ground_truth_list), np.array(predict_list), average='macro')
    recall = recall_score(np.array(ground_truth_list), np.array(predict_list), average='macro')
    f1 = f1_score(np.array(ground_truth_list), np.array(predict_list), average='macro')
    accuracy = accuracy_score(np.array(ground_truth_list), np.array(predict_list))
    with open(f'{output_dir}/results.txt', 'a') as f:
        f.write(f"Results Step 2 English Prompts dataset {args.dataset} Think {args.think} No context {args.nocontext} Repetition {args.repetition}")
        f.write(f"Start 2: {now}")
        f.write(f"End 2: {datetime.now()}\n") 
    with open(f'{output_dir}/results.txt', 'a') as f:
        f.write(report)
        f.write('\n')
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")

if __name__ == '__main__':
    args = parse_option()

    main(args)



