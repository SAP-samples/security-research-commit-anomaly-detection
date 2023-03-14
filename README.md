# Detecting Backdoors in Collaboration Graphs of Software Repositories
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/security-research-commit-anomaly-detection)](https://api.reuse.software/info/github.com/SAP-samples/security-research-commit-anomaly-detection)

## Anomaly detection in software development process using Graph Neural Networks based on version history metadata and collaboration graphs

This Repo uses the GraphRepo library (see the original Readme below) to generate graphs from Github repositories.

All new data generation pipeline and model files are in the directory "src"

## Install
Install all pip dependencies:
```
pip3 install -r requirements.txt
```

## Requirements

- Python3
- See requirements.txt for python modules

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

## Contributors

- Martin Härterich
- Tom Ganz

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2022 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt) file.

