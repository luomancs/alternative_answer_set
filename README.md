# Evaluation of Open-Ended VQA tasks
This repository contains the implementations of the evaluation metric for [GQA](https://arxiv.org/pdf/1902.09506.pdf) and [VQA](https://arxiv.org/pdf/1505.00468.pdf) introduced in the paper [‘Just because you are right, doesn’t mean I am wrong’: Overcoming a bottleneck in development and evaluation of Open-Ended VQA tasks](https://arxiv.org/pdf/2103.15022.pdf) at the [EACL 2021 conference](https://2021.eacl.org/).

## Repository Structure
```
alternative_answer_set
├── aas_generation
│   └── bert_embedding
│       ├── embedding.py # obtain the bert-based embeddings of label.
│       ├── run.py # obtain the aas using the similarity between embeddings.
│       ├── gqa # folder to save the files generated for gqa. 
│       └── vqa # folder to save the files generated for vqa.  
│   └──conceptNet
│       ├── run.py # obtain the aas using the conceptNet.
│       ├── gqa # folder to save the files generated for gqa. 
│       └── vqa # folder to save the files generated for vqa.  
│   └──counter_fit_synonyms
│       ├── run.py # obtain the aas using the counter_fit_synonyms.
│       ├── counter-fitted-vectors.txt # vectos for words.
│       ├── run.py # obtain the aas using the counter_fit_synonyms.
│       ├── gqa # folder to save the files generated for gqa. 
│       └── vqa # folder to save the files generated for vqa. 
│   └──wordNet
│       ├── run.py # obtain the aas using the wordNet.
│       ├── gqa # folder to save the files generated for gqa. 
│       └── vqa # folder to save the files generated for vqa.  
│   └──data
│       ├── gqa # folder for gqa orginal data downloaded from (https://cs.stanford.edu/people/dorarad/gqa/download.html). 
│       └── vqa # folder for vqa orginal data downloaded from (https://visualqa.org/download.html). 
│   └──total_union
│       ├── gqa # folder to save the file of the union of 4 different aas for GQA. 
│       └── vqa # folder to save the file of the union of 4 different aas for VQA. 
├── entailment
│   └── entailment_score.py # obtain the entailment score.  
├── evaluation
│   └── aas_gqa_files
│       ├── bert_aas.json  
│       ├── conceptNet_aas.json  
│       ├── counterfit_aas.json  
│       ├──  union_5_aas.json  
│       ├──  wordNet_aas.json  
│   └── gqa_prediction
│       ├── testdev_predict_aas.json # the prediction of lxmert model trained on the aas gqa labels. 
│       ├── testdev_predict_lxmert.json # the prediction of lxmert model trained on the original gqa labels.
│       ├── testdev_predict_vilbert.json # the prediction of vilbert model trained on the original gqa labels.
│   └── evaluation.py # run this script to get the performance.
│   └── gqa_testdev.json # the golden testdev file of gqa dataset, used in the evaluation script. 
├── grounding
│   └──  GQA_grounded_questions50.json # the templates for each label in GQA dataset 
│   └── VQA_v2_grounded_questions50.json # the templates for each label in VQA dataset 
```
## Install 
```
conda create -n aas python=3.8
conda activate aas
pip install -r requirements.txt
```
## Run Evaluation
An example of running evaluation is given belows, and change each parameters correspondingly. The prediction file is a json file and the format is given in gqa_prediction folder. 
```
cd evaluation
python evaluation.py \
--prediction_file gqa_prediction/testdev_predict_lxmert.json \
--golden_testing_file gqa_testdev.json \
--dataset_type gqa
```

## Generate the AAS 

### Generate AAS using BERT 
```
cd aas_generation
python bert_embedding/embedding.py \
--label_file data/gqa/trainval_label2ans.json \
--save_path bert_embedding/gqa/trainval_ans2bertemb.pth \

python bert_embedding/run.py \
--embedding_file bert_embedding/gqa/trainval_ans2bertemb.pth \
--save_file bert_embedding/gqa/bert_aas.json \
```
### Generate AAS using ConceptNet 
```
cd aas_generation
python conceptNet/run.py \
--label_file data/gqa/trainval_label2ans.json \
--save_file conceptNet/gqa/conceptNet_aas.json \
```

### Generate AAS using Counter-Fit-Vector 
```
cd aas_generation
python counter_fit_synonyms/run.py \
--label_file data/gqa/trainval_label2ans.json \
--save_file counter_fit_synonyms/gqa/cfv_aas.json \
```

### Generate AAS using WordNet
```
cd aas_generation
python wordNet/run.py \
--label_file data/gqa/trainval_label2ans.json \
--save_file wordNet/gqa/wordNet_aas.json \
```

### Generate Entailment Score 
```
python entailment/entailment_score.py \
--aas_file aas_generation/bert_embedding/gqa/bert_aas.json \
--save_file aas_generation/bert_embedding/gqa/bert_aas_score.json \
--dataset_type gqa
```