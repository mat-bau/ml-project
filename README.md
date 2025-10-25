# Machine Learning project

## Organisation du projet 
ml-project/

├── README.md                  # Overview et instructions
├── requirements.txt           # Dependencies
├── data/
│   ├── labeled_data/
│   └── unlabeled_data/
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_part1_linear.ipynb
│   ├── 03_part2_nonlinear.ipynb
│   └── 04_part3_images.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Data preprocessing functions
│   ├── feature_engineering.py # Feature selection/extraction
│   ├── models.py              # Model classes
│   ├── train.py               # Training scripts
│   ├── evaluate.py            # Evaluation metrics
│   └── utils.py               # Helper functions
├── results/                   # Predictions and figures
└── report/                   

test fonctions et parametres dans notebooks -> fonctions qui marchent => dans les fichiers python dans src/

## Installation 

### 1. Installer Python 3.12 via Homebrew

```bash
# Installer Homebrew si pas déjà fait
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Installer Python 3.12
brew install python@3.12

# Vérifier l'installation
/opt/homebrew/bin/python3 --version
```

### Créer un environnement virtuel pour le projet 

```bash
# Aller dans le dossier du projet
cd LELEC2870/ml-project

# Créer un environnement virtuel avec Python 3.12
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv

# Activer l'environnement virtuel
source .venv/bin/activate

# Vérifier le Python utilisé
which python
python --version
```
### 3. Installer les dépendances du projet

```bash
# Mettre à jour pip
pip install --upgrade pip

# Installer les packages requis
pip install -r requirements.txt
```

### 4. Configurer Jupyter Notebook pour utiliser l'environnement virtuel
```bash
# Installer ipykernel dans l'environnement virtuel
pip install ipykernel
```

Et choisir le kernel Python 3.12 (.venv) dans la liste des kernel dispo

### 5. Gestion des kernels 
```bash
jupyter kernelspec list           # liste tous les kernels
jupyter kernelspec remove <name>  # supprimer un kernel
deactivate                        # sortir de l'environnement
rm -rf .venv                       # supprimer l'environnement
```


## Instructions

The project consists of **four parts**:

- Part 1: Work with tabular data and implement a linear model.
- Part 2: Continue using tabular data, but implement a nonlinear model.
- Part 3: Integrate image data into your analysis.
- Part 4: Take a broader perspective to better understand the growing problem of heart
failure within Smurf society.

You must code in Python. You do not need to implement everything from scratch — you may use
any appropriate library (e.g. scikit-learn for traditional models or pytorch for deep learning).
In addition to the code, you must write a report summarizing your methodology, results, and
conclusions. Below are detailed guidelines for each part.

**Part 1 — Linear Model (Baseline)** Your first task is to prepare the dataset for analysis. This
includes: removing features that are clearly irrelevant for prediction; encoding categorical variables
so they can be processed by regression models; and transforming numerical variables if necessary
(e.g. normalization or standardization). Pay special attention to the preprocessing stage, as it is
essential for building robust models and will be reused in later parts. Next, proceed with feature
selection and model selection (if applicable). In your report, describe and justify all your choices,
clearly present your results, and discuss them critically.

**Part 2 — Nonlinear Models** You will now compare several types of nonlinear models (e.g.
non-parametric, tree-based, neural networks, etc). Note that some nonlinear models are sensitive
to uninformative features — good feature selection may be useful. Explore various selection meth-
ods beyond simple correlation filters. In your report, discuss: which features are most important?
Are they different from those in Part 1? Did you modify your preprocessing pipeline?
Also, nonlinear models typically involve many hyperparameters, so model selection and tuning will
be more intensive at this stage. For each model: define a relevant set of hyperparameters and
fine-tune them within the limits of your computational resources. Be mindful of data splitting for
model selection. Clearly explain how you partition the data into training and validation (the test
set is provided). Use cross-validation when computationally feasible. Based on your experiments,
identify the best-performing model and estimate its generalization performance. In your report,
clearly indicate whether reported metrics correspond to the training, validation, or test set.

**Part 3 — Integration of Image Data** In this part, you will integrate heart scan images
into your pipeline. Extract features from the images using a deep neural network and combine
these image-derived features with the tabular data. Retrain (and, if necessary, adapt) your best
nonlinear model from Part 2 on this combined dataset. Compare performance with and without
the image features, and discuss your findings. We will cover how to implement a convolutional
neural network for image data during the practical session in Week 9 .

**Part 4 — Understanding Heart Failure in Smurf Society** In the final part, you will take
a broader analytical perspective. Your goal is to: formulate hypotheses about the causes of heart
failure, identify groups of Smurfs most at risk; and support your analysis with clear visualizations
and graphs. You are free to use any analytical or visualization tools you find appropriate and may
build upon results from the previous parts.

## Schedule
Below you will find the **schedule** for the project.
- As soon as possible: Register your group (maximum two people) on Moodle
- Tuesday 04/11 at 23h55: Intermediate deadline where you submit your work for part 1 as 2
seperate files (a csv file for you first predictions and a pdf for a ”pre-report” on part 1)
- Thursday 6/11 at 8h30: Q/A session #1
- Thursday 20/11 at 8h30: Q/A session #2
- Friday 5/12 at 23h55: final deadline where you submit your work as 3 separate files (a csv
file for your predictions, a pdf for your report, and a compressed folder for all your scripts)
