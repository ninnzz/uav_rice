# uav_rice

This repository contains all the codebase and relevant information for running the experiments used on  the research entitled `Detection of Typhoon Damaged Regions on UAV Captured Rice Field Images Using an Ensemble of CNN and Artificial Neural Network
`.

The main experiment notebook is located in [google colab](https://colab.research.google.com/drive/1v7VPMfzJ06U9DEZBTnjsNFpa1USjDV_U#scrollTo=CsHWOe5oBRd_).
You can copy and edit the notebook depending on the experiment you want to run. The details of the parameters and all relevant information can be found in the step by step instruction in the notebook.



## Project structure

```markdown
├── README.md                       <- Instructions, FAQs
├── data                            <- Data files
├── uav_utils
│   ├── data_classes.py             <- Data format and structures
│   ├── data_utils.py               <- Data loading and other utility
│   ├── display.py                  <- Display of results 
│   ├── metrics.py                  <- Custom metrics used for evaluation
│   ├── models.py                   <- CNN and NN models
│   ├── preprocessing.py            <- Preprocessing functions
│   └── process.py                  <- Training relevant functions
|
├── notebooks
│   ├── analysis.ipynb              <- Analysis script
│   ├── show_annotations.ipynb      <- Display v1 and v2 annotations
│   └── show_heatmap.ipynb          <- Displaying results with heatmap overlay
|
└── pyproject.toml                  <- Poetry config used to build environment
```

## Running the project

To be able to run the local setup, you need to have the following: 

- Python 3.8 or higher
- Install poetry for dependency management.

Since we are using a git version of the package for the notebook, you need to edit your local `pyproject.toml` first. 
Uncomment all the dependencies in the `pyproject.toml` file. Your toml file should look like this:

```toml
[tool.poetry]
name = "uav-rice"
version = "0.1.7"
description = "Thesis codes"
authors = ["ninz <nreclarin@gmail.coml>"]
readme = "README.md"
packages = [{include = "uav_utils"}]

[tool.poetry.dependencies]
python = "^3.8,<3.12"
xmltodict = "^0.13.0"
# NOTE: Uncomment all the dependencies below
scipy = "^1.10.0"
matplotlib = "^3.6.2"
scikit-learn = "^1.2.0"
opencv-python = "^4.7.0.68"
pandas = "^1.5.2"
jupyter = "^1.0.0"
# Uncomment ends here


[tool.poetry.group.dev.dependencies]
fastapi = "^0.89.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

```

Once you are finished, run the following commands:

```shell
poetry install
```
This will install all the dependencies to be able to run the notebooks.

```shell
poetry run jupyter notebook
```
This command will run the notebooks that you need for model assessment and result generation.

> Important Note
> 
> If you also want to run the training scripts on your local, you need to manage the `keras` installation separately.
> Different OS and machine setup require different keras setup. M1 Macs have issues when using Keras on higher python versions as of writing (01/2023).


### Running Training Scripts

Use the main google colab notebook for running the training scripts. 
You can copy it and use it in a local setup, however, the run time might vary depending on the machine you are using.
Use the [google colab](https://colab.research.google.com/drive/1v7VPMfzJ06U9DEZBTnjsNFpa1USjDV_U#scrollTo=CsHWOe5oBRd_) notebook to generate the initial prediction results for both CNN and NN network.

You should end up with several `csv` files which you can use to analyze the performance of each model.

### Running Assessment and Result Analysis

You will need the `csv` files generated in the training script for this step.
There are two notebooks that you can use to assess the result. There are detailed instructions on each notebook.