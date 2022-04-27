# Training and saving models with CML on a dedicated AWS EC2 runner

The files in this repository provide an example on how to use a dedicated runner on AWS to train and export a machine learning model. It accompanies [this blog post](https://dvc.org/blog/CML-runners-saving-models-1), which contains a full guide on how to achieve this.

## Contents
This repository contains the following files:

- `requirements.txt`: the packages necessary for training our model.
- `get_data.py`: script that generates sample data to train a model on.
- `train.py`: script that trains a random forest on the generated data and exports that model to a binary file, along with a confusion matrix and some metrics.
- `.github/workflows/cml.yaml`: example workflow that provisions an AWS EC2 instance to run `train.py` and export the resulting model.

## How to install and run
Clone this repository and follow the instructions in [this blog post](). Specifically, make sure satisfy the prerequisites with regards to the AWS and GitHub Workflows set-up.

## Need help? Or want to contribute?
If you need any help in following this guide you can [drop us a message in Discord](https://dvc.org/chat). In case you run into any issues that need fixing, don't hesitate to open an issue in this repository and/or submit a fix in a pull request.
