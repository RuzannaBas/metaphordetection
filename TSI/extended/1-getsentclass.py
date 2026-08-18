import argparse
import os
import csv
import numpy as np
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
        with open(os.path.join(file), 'w', encoding='utf-8', newline='') as f:
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
        

system_prompt = '''You are a helpful AI assistant. Answer as concisely as possible.'''


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
    parser.add_argument('--dutch', action="store_true", default=False)
    args, unparsed = parser.parse_known_args()
    return args

def main(args):
    now = datetime.now()
    print(now)
    _model = args._model
    if 'kankernl' in args.dataset:
        nocontext = args.nocontext
        if nocontext:
            context = 'nocontext'
        else: 
            context = 'context'
        output_dir = f'{args.output_path}/{context}/{args.dataset}/{args.repetition}'
    else: 
        nocontext = True
        output_dir = f'{args.output_path}/{args.dataset}/{args.repetition}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f'{output_dir}/{args.dataset}1.csv'
    dataloader = Data_Processor(args)
    file = f'{args.data_dir}/{args.dataset}.csv'
    samples = dataloader.get_dataset(file)
    new_samples = []
    # Load files with rules for mapping LLM reponses to metaphor/literal
    if args.dutch:
        no = open(f'{args.data_dir}/nee.txt').read().splitlines()
        yes = open(f'{args.data_dir}/ja.txt').read().splitlines()
    else: 
        no = open(f'{args.data_dir}/no.txt').read().splitlines()
        yes = open(f'{args.data_dir}/yes.txt').read().splitlines()

    # Determine metaphoricity label for each label   
    for sample_id in tqdm(range(len(samples))):
        metaphor_count = 0 # Number of LLM responses that can be mapped to metaphor
        literal_count = 0 # Number of LLM responses that can be mapped to literal
        final = -1 # Final metaphor label based on LLM responses
        sample  = samples[sample_id]

        # Load data depending on dataset
        if 'kankernl' in args.dataset:
            blog_id,sentence_id,full_blog,sentence,tokens,labels,label = sample
        elif 'MMC' in args.dataset:
            sentence,label = sample
        else:
            sentence,tokens,labels,label = sample                
        
        sentence_add_punch = sentence.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' "', '"').replace('  .', '.')

        # Define prompts depending on prompt language and context
        if args.dutch:
            if nocontext:
                prompt_sent = 'Een metafoor beschrijft een concept in termen van een ander, waardoor een woord of zinsdeel buiten de meest basale betekenis gebruikt kan worden. Voor deze opdracht moeten andere vormen van beeldspraak, zoals vergelijkingen, ook als metafoor worden beschouwd. Bepaal aan de hand van de zin "%s" of deze een metafoor bevat. Controleer of een van de woorden in deze context een andere betekenis heeft dan de letterlijke betekenis. Bevat de zin een metafoor?' %(sentence_add_punch)
            else:
                prompt_sent = 'Een metafoor beschrijft een concept in termen van een ander, waardoor een woord of zinsdeel buiten de meest basale betekenis gebruikt kan worden. Voor deze opdracht moeten andere vormen van beeldspraak, zoals vergelijkingen, ook als metafoor worden beschouwd. Bepaal aan de hand van de blog "%s" en de zin "%s" of deze een metafoor bevat. Controleer of een van de woorden in deze context een andere betekenis heeft dan de letterlijke betekenis. Bevat de zin een metafoor?' %(full_blog, sentence_add_punch)      

        else:
            if nocontext:
                prompt_sent = 'A metaphor describes one concept in terms of another, allowing a word or phrase to be used beyond its most basic meaning. For this task, other types of figurative language, such as similes, should also be considered metaphorical. Given the sentence "%s" determine whether the sentence contains a metaphor. Check whether any of the words are used with a meaning that differs from their basic or literal meaning in this context. Does the sentence contain a metaphor?' %(sentence_add_punch)   
            else:
                prompt_sent = 'A metaphor describes one concept in terms of another, allowing a word or phrase to be used beyond its most basic meaning. For this task, other types of figurative language, such as similes, should also be considered metaphorical. Given the blog "%s" and sentence "%s" determine whether the sentence contains a metaphor. Check whether any of the words are used with a meaning that differs from their basic or literal meaning in this context. Does the sentence contain a metaphor?' %(full_blog, sentence_add_punch)      

        dialogs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_sent}
            ]
    
        answers_res = []
        # Prompt the LLM 
        for _ in range(args.vote_number):
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
            # Map LLM response to metaphor/literal
            if any(x in answer for x in yes):
                metaphor_count += 1
            elif any(x in answer for x in no):
                literal_count += 1
            else:
                print("Unsure,", answer)
                with open('unsure.txt', 'a') as f:
                    f.write(f"{answer} \n")

        # Define final metaphor label for sentence using majority voting
        if metaphor_count > literal_count:
            final = 1
        else:
            final = 0

        # Save new data including prompt, LLM responses, counts and final label
        if 'kankernl' in args.dataset:
            new_samples.append([blog_id, sentence_id, full_blog, sentence, tokens, labels, label, dialogs, answers_res, metaphor_count, literal_count, final, final])
            head = ['blog_id', 'sentence_id', 'full_blog', 'sentence', 'tokens', 'labels', 'true_label', 'prompt', 'classification', 'metaphor_count', 'literal_count', 'label1', 'final_label']
            dataloader.write_dataset(output_file, head, new_samples)
        elif 'MMC' in args.dataset:
            new_samples.append([sentence, label, dialogs, answers_res, metaphor_count, literal_count, final, final])
            head = ['sentence', 'true_label', 'prompt', 'classification', 'metaphor_count', 'literal_count', 'label1', 'final_label']
            dataloader.write_dataset(output_file, head, new_samples)
        else:
            new_samples.append([sentence, tokens, labels, label, dialogs, answers_res, metaphor_count, literal_count, final, final])
            head = ['sentence', 'tokens', 'labels', 'true_label', 'prompt', 'classification', 'metaphor_count', 'literal_count', 'label1', 'final_label']
            dataloader.write_dataset(output_file, head, new_samples)

    # Evaluate sentence classification by LLM 
    data = pd.read_csv(output_file)
    ground_truth_list = data['true_label'].tolist()
    predict_list = data['final_label'].tolist()
    report = classification_report(np.array(ground_truth_list), np.array(predict_list), digits=4)
    precision = precision_score(np.array(ground_truth_list), np.array(predict_list), average='macro')
    recall = recall_score(np.array(ground_truth_list), np.array(predict_list), average='macro')
    f1 = f1_score(np.array(ground_truth_list), np.array(predict_list), average='macro')
    accuracy = accuracy_score(np.array(ground_truth_list), np.array(predict_list))
    with open(f'{output_dir}/results.txt', 'a') as f:
        f.write(f"Results Step 1 English Prompts dataset {args.dataset} Think {args.think} No context {args.nocontext} Repetition {args.repetition}\n")
        f.write(f"Start 1: {now}")
        f.write(f"End 1: {datetime.now()}\n") 
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



