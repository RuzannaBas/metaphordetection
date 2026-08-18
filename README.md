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
