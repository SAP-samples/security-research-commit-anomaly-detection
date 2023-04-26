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

## Cite

```
@inproceedings{10.1145/3577923.3583657,
author = {Ganz, Tom and Ashraf, Inaam and H\"{a}rterich, Martin and Rieck, Konrad},
title = {Detecting Backdoors in Collaboration Graphs of Software Repositories},
year = {2023},
isbn = {9798400700675},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3577923.3583657},
doi = {10.1145/3577923.3583657},
abstract = {Software backdoors pose a major threat to the security of computer systems. Minor modifications to a program are often sufficient to undermine security mechanisms and enable unauthorized access to a system. The direct approach of detecting backdoors using static or dynamic program analysis is a daunting task that becomes increasingly futile with the attacker's capabilities. As a remedy, we introduce an orthogonal strategy for the detection of software backdoors. Instead of searching for concealed functionality in program code, we propose to analyze how a software has been developed and locate clues for malicious activities in its version history, such as in a Git repository. To this end, we model the version history as a collaboration graph that reflects how, when and where developers have committed changes to the software. We develop a method for anomaly detection using graph neural networks that builds on this representation and is able to detect spatial and temporal anomalies in the development process. % We evaluate our approach using a collection of real-world backdoors added to Github repositories. Compared to previous work, our method identifies a significantly larger number of backdoors with a low false-positive rate. While our approach cannot rule out the presence of software backdoors, it provides an alternative detection strategy that complements existing work focused only on program analysis.},
booktitle = {Proceedings of the Thirteenth ACM Conference on Data and Application Security and Privacy},
pages = {189–200},
numpages = {12},
keywords = {software repositories, neural networks, anomaly detection},
location = {Charlotte, NC, USA},
series = {CODASPY '23}
}
```
## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2022 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt) file.

