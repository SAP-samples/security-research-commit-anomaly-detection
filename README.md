# security-research-commit-anomaly-detectioni
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/security-research-commit-anomaly-detection)](https://api.reuse.software/info/github.com/SAP-samples/security-research-commit-anomaly-detection)

## Anomaly detection in software development process using Graph Neural Networks based on version history metadata and collaboration graphs

This Repo uses the GraphRepo library (see the original Readme below) to generate graphs from Github repositories.

All new data generation pipeline and model files are in the directory "src"

## Install
Install all pip dependencies:
```
pip3 install -r requirements.txt
```

### Data Gereration Files:
- gen_repo_dataset.py
- gen_graph_dataset.py
- gen_git_graph.py
- gen_node_edge_features_all.py

### Anomaly Injection and Labeling Files:
- anomaly_injection.py
- label_anomalies.py

### Data Analysis Files:
- graph_analysis.py
- utils.py

### Model Files:
- deepSAD_model.py
- GEN_Classifier.py

### Train and Test Files:
- train_test_datasets_GENConv.py
- train_test_datasets.py
- train_test_repo_dataset_GENConv.py
- train_test_repo_dataset.py

### Run Files:
- run_datasets_GENConv.py
- run_datasets.py
- run_repo_dataset_GENConv.py
- run_repo_dataset.py




