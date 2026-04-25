#### This Project was Developed as part of the Doctoral Dissertation in the University of Urbino, Italy.

## User Feedback-based Counterfactual Explanations (UFCE)

This ReadMe file provides an overview of the main steps to adopt for generating counterfactual explanations with UFCE. The complete source code and all the utilities are availble in the root folder.
This repository contains the source code for UFCE which is published in international journal of computational intelligence systems, you can read the related paper [here](https://link.springer.com/article/10.1007/s44196-024-00508-6).

### Installing necessary libraries
To install the libraries, download the UFCE folder and navigate to root folder in the terminal and run the following command:

```python
pip install -r requirements.txt
```
If you face any problems, try installing dependencies manually.

### Getting started with UFCE

This example provides demo on Bank Loan dataset.

Import libraries.
```python
import ufce
from ufce import UFCE
from goodness import *
from cfmethods import *
from evaluations import *
from data_processing import *
from generate_text_explanations import *
```

Load data and train ML model
```python
path = r'\~\Bank_Loan.csv'
bankloan = pd.read_csv(path)
dataset = 'bank'
mlp_blackbox, mlp_mean, mlp_std, lr_blackbox, lr_mean, lr_std, testset, Xtrain, X, Y, df = classify_dataset_getModel(bankloan, data_name=dataset)
```

User specified constraints (User Preferences)
```python
# this will return user-constraints specific to data set (default), can be changed as per requirement.
features, catf, numf, uf, f2change, outcome_label, desired_outcome, nbr_features, protectf, data_lab0, data_lab1 = get_bank_user_constraints(bankloan) 
```

Importing functionality of UFCE
```python
ufc = UFCE()
```

Finding Mutual Information
```python
# Mutual information sharing pair of features
MI_FP = ufc.get_top_MI_features(X, features)
print(f'\t Top-5 Mutually-informed feature pairs:{MI_FP[:5]}')
```

### Generating counterfactual explanations

Single feature counterfactuals
```python
no_cf_exp = 1
# this calls to Single_F method
onecfs, methodtimes[i], foundidx1, interval1, testout1 = sfexp(X, 
                                                               data_lab1, 
                                                               testset[:1], uf, step, 
                                                               f2change, numf, catf, 
                                                               lr_blackbox, 
                                                               desired_outcome, 
                                                               no_cf_exp)
print(display(onecfs[:1]) #counterfactuals from ufce1
```

Double feature counterfactuals
```python
no_cf_exp = 1
# this calls to Doule_F method
twocfs, methodtimes[i], foundidx2, interval2, testout2 = dfexp(X, data_lab1, 
                                                               testset[:1], 
                                                               uf, MI_FP[:5], 
                                                               numf, catf, 
                                                               features, protectf, 
                                                               lr_blackbox, 
                                                               desired_outcome, 
                                                               no_cf_exp)
print(display(twocfs[:1]) #counterfactuals from ufce2
```

Tripple feature counterfactuals
```python
no_cf_exp = 1
# this calls to Triple_F method
threecfs, methodtimes[i], foundidx3, interval3, testout3 = tfexp(X, data_lab1,
                                                                  testset[:5], 
                                                                  uf, MI_FP[:5], 
                                                                  numf, catf, 
                                                                  features, 
                                                                  protectf, 
                                                                  lr_blackbox, 
                                                                  desired_outcome,
                                                                  no_cf_exp)
print(display(threecfs[:1]) #counterfactuals from ufce3
```

#### Textual explanation generation
```python
outcome_var = "The personal loan"
actual_class = 'denied'
desired_class = 'accepted'
features, new_values, actual_values = ufc.features_to_explain(testset[:1], onecfs[:1]) # similarly, calling with twocfs and threecfs for double and trippe feature variations of UFCE.
# following snippet reason on the outcome of factual instance
ufc.generate_reason_explanation(outcome_var, desired_class, actual_class, features)
# following snippet suggest the counterfactuals to obtain desired outcome 
ufc.generate_suggestion_explanation(outcome_var, desired_class, actual_class, features, new_values, actual_values)
```

# Running the experiment on multiple data sets:
The file `experiments.py` can be run in PyCharm IDE or in Jupyetr Notebook or on the terminal (cmd). 
To do so, download the code folder (UFCE). Open the file `experiments.py` in any editor and set the paths of your system or working directory.
Open the cmd terminal, navigate to the root folder and run the command: `python experiments.py`

To runt the user feedback analysis, run the `uf_analysis.py`.

##### The already generated results are present in the directory at `UFCE/folds/bank/results/`. Similarly, the results of all CF methods on 5 data sets for different evaluation metrics are present in the directory at `UFCE/folds/`. 

### Cite As

If you use UFCE in your research, please cite it as:

```python
@article{suffian2024introducing,
  title={Introducing User Feedback-Based Counterfactual Explanations (UFCE)},
  author={Suffian, Muhammad and Alonso-Moral, Jose M and Bogliolo, Alessandro},
  journal={International Journal of Computational Intelligence Systems},
  volume={17},
  number={1},
  pages={123},
  year={2024},
  publisher={Springer}
}
```

