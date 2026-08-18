# From war to wellness: Understanding patient experience through automatic metaphor detection
This repository contains the code for the thesis From war to wellness: Understanding patient experience through automatic metaphor detection.
Two approaches are tested and the code is available in the following folders
## Sentence classification using fine-tuned multilingual BERT models
All code for the fine-tuning of the multilingual BERT models can be found under <code>SENT_CLASS</code>.
The fine-tuned models used for evaluation in the thesis are available upon request.
The following folders and files can be found in SENT_CLASS
### <code>Data</code>
All data needed for the sentence classification is available under this folder. Each dataset is available under its own subfolder and contains the original data, cleaned data as returned by  <code>preprocessing_{dataset}.py</code>, and the train and test splits as returned by <code>split_data.py</code>.
### <code>split_data.py</code> 
Splits the data in a train and test set. Run using:
```python split_data.py {dataset}```
### <code>train.py</code>
Main file to run hpo, train the final configuration and to evaluate all models on all English datasets.\
Run HPO with \
```python train.py --data {dataset} --worker``` \
Train the best configuration of the HPO on the full train set on 5 different seeds with \
```python train.py --data {dataset} --final``` \
Evaluate all models on the English datasets with \
```python train.py --eval```
### <code>evaluate_kankernl.py</code>
Evaluate all models on the kankernl dataset. \
Run with ```python eval_kankernl.py```
### <code>run.py</code>
This file can be used to speed up the process of running HPO by dividing workers over multiple GPUs. The trials will be divided over the GPUs untill the total number of trials has been reached.\
Run with ```python run.py```

## Theory-guided Scaffolding Instruction
All code regarding the TSI framework can be found under <code>TSI</code>\
All code regarding the reproduction of the original TSI framework can be found under <code>TSI/reproduction</code>\
All code regarding our proposed extended TSI framework can be found under <code>TSI/extended</code>\
### <code>reproduction</code>
This folder contains all files for the reproduction of the original TSI framework. The same code is used as in https://github.com/TIAN-viola/TSI/tree/main. The only thing that is changed is the model used (Qwen3:8B), the system prompts and the prompt in <code>1-1que_2que.py</code>.\
The framework can be run by running the following files in order:
```
python 1-1que_2que.py
python 2-extract_1ans_2ans.py
python 3-1ans_2ans_sim_score_for_CMT_and_MIP.py
python 4-1que_2que_kdg.py
python 5-3que.py
python 6-calcu.py
```
<code>--think</code> can be added for scripts 1, 4 and 5 to enable the thinking mode of Qwen3. 

Alternatively, <code>total.py</code> can be used. This script is used to run experiments on the different think modes of Qwen3 but can also be used to run the framework in total by putting the preferred think configuration in the <code>parameters</code> variable in <code>total.py</code>

### <code>extended</code>
This folder contains all files for our proposed extended TSI framework. Data can be found under <code>data</code>.
Run sentence classification using <code>1-getsentclass.py --dataset {dataset}</code>\
{dataset} can be <code>MOH-X</code>, <code>MMC</code>, <code>VUAMC</code> or <code>kankernl_test</code>
<code>--dutch</code> and <code>--nocontext</code> can be used to run with dutch prompts and without context on the kankernl_test dataset.

Run sentence classification with knowledge using <code>2-getknowledgebased_class.py --dataset {dataset}</code>
 
To run the full framework, run the following scripts in order after running <code>1-getsentclass.py</code> and <code>2-knowledgebased_class</code>
```
python 3-getspan.py --dataset {dataset}
python 4-knowledgebased_span.py --dataset {dataset}
python 5-getcontext.py --dataset {dataset}
```

To run evaluation on only the token-level classification using ground truth metaphors only, run the following scripts in order:
```
python 3-getspan_eval.py --dataset {dataset}
python 4-knowledgebased_span.py --dataset {dataset}
python 5-getcontext_eval.py --dataset {dataset}
```


## Citations
MOH-X Dataset - Mohammad, S., Shutova, E., & Turney, P. (2016, August). Metaphor as a medium for emotion: An empirical study. In Proceedings of the fifth joint conference on lexical and computational semantics (pp. 23-33).

VUAMC Dataset - Steen, G. (2010). A method for linguistic metaphor identification. Converging evidence in language and communication research.

MMC Dataset  - Lippolis, A. S., Nuzzolese, A. G., & Gangemi, A. (2025). The Medical Metaphors Corpus (MCC). arXiv preprint arXiv:2508.07993.
