#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import argparse
import os
import csv
from ollama import chat, AsyncClient
import asyncio
from datetime import datetime

client = AsyncClient()
class Data_Processor:
    def __init__(self, args):
        self.data_root_path = os.path.join(args.data_dir, args.dataset)
        self.output_root_path = os.path.join(args.output_path, args.method, args._model, args.dataset)
    
    def get_dataset(self, file):
        file_dir = os.path.join(self.data_root_path, file)
        examples = []
        with open(file_dir, 'r', encoding='utf-8') as f:
            lines = csv.reader(f)
            next(lines)  # skip the headline
            for i, line in enumerate(lines):
                # sentence,label,target_position,target_word,pos_tag,gloss,eg_sent
                examples.append(line)
        return examples

    def write_dataset(self, file, head, examples):
        if not os.path.exists(self.output_root_path):
            os.makedirs(self.output_root_path)
        with open(os.path.join(self.output_root_path, file), 'w', encoding='utf-8', newline='') as f:
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
                    # sentence,label,target_position,target_word,pos_tag,gloss,eg_sent
                    examples.append(line)
            return examples
dataset_to_fileList = {
    'MOH-X':['MOH-X.csv'],
    'TroFi':['TroFi.csv'],
}


pos_to_word = {
    'verb':'verb',
}
system_prompt = '''You are a helpful AI assistant. Answer as concisely as possible.'''

source_domain_prompt = '''The source domain is a conceptual domain containing concepts that are typically concrete, tangible, and familiar to us. What is the source domain implied by the [pos] "[target_word]"?'''

target_domain_prompt = '''The target domain is a conceptual domain containing concepts that are typically vague and abstract. In the sentence "[sentence]", what is the target domain that the word "[target_word]" tries to describe?'''


def parse_option():
    parser = argparse.ArgumentParser(description='Metaphor reasoning')
    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--method', default='CMT-1que-2que', type=str, help='')
    parser.add_argument('--dataset', default='MOH-X', type=str, help='dataset name, MOH-X, TroFi')
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--think', action="store_true", default=False)
    parser.add_argument('--output_path', default='results/', type=str)
    parser.add_argument('--_model', default='qwen3', type=str, help='''qwen3''')
    parser.add_argument('--N_o', default=5, type=int)
    args, unparsed = parser.parse_known_args()
    return args


async def ask(dialogs, model, temp):
    response = await client.chat(
        model=model,
        messages=dialogs,
        think=args.think,
        options={"temperature": temp},
    )
    return response["message"]["content"]

async def answer_question(question, model, temp, n_outputs):
    dialogs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    tasks = [
        ask(dialogs, model, temp)
        for _ in range(n_outputs)
    ]

    return await asyncio.gather(*tasks)

async def main(args):
    now = datetime.now()
    print(now)
    args.method = args.method + '-' + str(args.temp) 
    _model = args._model
    dataloader = Data_Processor(args)

    for file in dataset_to_fileList[args.dataset]:
        new_samples = []
        samples = dataloader.get_dataset(file)
        new_samples = dataloader.get_dataset_output(file)
        for sample_id in range(len(new_samples), len(samples)):
            print(sample_id)
            sample  = samples[sample_id]
            sentence,label,target_position,target_word,pos_tag,gloss,eg_sent = sample
            sentence_add_punch = sentence.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' "', '"')
            answers = [[], []]
            answers_kdg = [[], []]
            questions = [
                source_domain_prompt.replace('[pos]', pos_to_word[pos_tag.lower()]).replace('[target_word]', target_word),
                target_domain_prompt.replace('[sentence]', sentence_add_punch).replace('[target_word]', target_word),
            ]

            question_tasks = [
                answer_question(
                    question,
                    _model,
                    args.temp,
                    args.N_o
                )
                for question in questions
            ]
            question_answers = await asyncio.gather(*question_tasks)
            answers = question_answers

            new_samples.append([sentence,label,target_position,target_word,pos_tag,gloss,eg_sent,str(questions),str(answers),str(answers_kdg)])
            head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent','questions', 'answers', 'answers_kdg']
            dataloader.write_dataset(file, head, new_samples)
        with open('results/results.txt', 'a') as f:
            f.write(f"Start 1: {now}\n")
            f.write(f"End 1: {datetime.now()}\n")


if __name__ == '__main__':
    args = parse_option()
    asyncio.run(main(args))
