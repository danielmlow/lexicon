# Code for manuscript:

Low et al. (2024). Building lexicons with Generative AI result in lightweight and interpretable text models with high content validity. ArXiv. 

<!-- 
These: not mentioned because they are not used:
ctl_cts_feature_extraction.ipynb
ctl_cts_ml_models.ipynb 
-->

# Data

## Crisis Text Line data
Crisis Text Line data is private and sensitive and was obtained de-identified through a collaboration and DUA with Crisis Text Line. If you obtain permission from Crisis Text Line, we are willing to share conversation IDs so the same conversations can be analyzed.

Final datasets (private: contains text):

`./data/input/ctl/`
- train `train10_train_30perc_text_y_balanced_regression.csv`
- val `train10_val_15perc_text_y_regression.csv`
- test `train10_test_15perc_text_y_regression.csv`

Extracted features (private: contains text):
- `./data/input/ctl/ctl_dfs_features_regression.pkl` 


## Suicide Risk Lexicon

Lexicon saved in `construct-tracker` package

```python
!pip install construct-tracker
from construct_tracker import lexicon
srl = lexicon.load_lexicon(name = 'srl_v1-0')
```

<!-- ```python
from construct_tracker import lexicon
srl = lexicon.load_lexicon(name = 'srl_v1-0')
srl_prototypes = lexicon.load_lexicon(name = 'srl_prototypes_v1-0')
``` -->


Unvalidated lexicon:

- `data/input/lexicons/suicide_risk_lexicon_calibrated_unmatched_tokens_unvalidated_24-02-15T21-55-05.pickle`


## Data structure

```
run `$tree -L 2`, for example:

`tree ./data/input -L 2`
```

input
├── CTL
│   ├── 
├── reddit
│   ├── 
├── lexicons
│   ├── suicide_risk_lexicon
│   │	├── preprocessing
│   │	│	├── definitions and examples csv
│   │	│	├── gpt-4 + word score first_pass_annotation csv
├── features
│   ├── suicide_risk_lexicon
│   ├── liwc-22

output
├── descriptive_statistics
├── word_score_analysis
├── ml_performance
├── tables
├── figures



# Code for reproducing preprocessing, figures, and tables 

All final figures in tables are available in `./data/output/figures/` and `./data/output/tables/`

## Preprocessing

### Creating dataset
- `build_dataset.ipynb` obtain data from CTL servers (private, can share)
- `build_post_session_survey_df.ipynb`
- `train_test_split_ids.ipynb` in this repo. Takes downloaded data and creates datasets in the "risk assessment small" section 
- `descriptive_statistics.ipynb` Some initial quick descriptive stats
- Right before running models, there is a balancing done on the training set: `suicide_risk_assessment.ipynb`

### Creating lexicon
- General updated tutorial: see concept-tracker package `tutorials/create_lexicon.ipynb` [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danielmlow/construct-tracker/blob/main/tutorials/construct_tracker.ipynb)

How it was done in this study:
1. Choose constructs and seed examples: `./../data/lexicons/suicide_risk_preprocessing/suicide_risk_constructs_and_definitions.csv`
2. Create preliminary lexicon with Generative AI `create_lexicon.py`
3. Automatically obtain word scores: `word_scores_scattertext_ctl.ipynb`
4. Add and remove tokens from prior lexicons, thesauri: `create_lexicon.py`
5. Consolidate ratings from clinicians and output final validated lexicon including inter-rater agreement: `clinician_annotations.py`

### Feature extraction

- Suicide Risk Lexicon, LIWC: `suicide_risk_assessment.ipynb` 
- Construct-text similarity `ctl_cts.ipynb`


## Results: figures 

### Figure 1: 
`risk_classification_setfit.ipynb`

### Figure 2 (Building lexicon steps)
TODO: link to google slides

### Figure 3 (Suicide Risk Lexicon network analysis)
`lexicon_network_dendrogram.ipynb`

### Figure 4 (distribution of SRL matches)
`barplot_srl_matches.py` 

### Figure 5 (Boxplots prediction vs. true)
`suicide_risk_assessment_results.ipynb`

## Results: tables

### Table 2 
`lexicon_source_descriptives.py`
- input: lexicon pickle file
- output: 

### Table 3 (machine learning and deep learning models)
- `ctl_mpnet_text.ipynb` and `ctl_distilbert_text.ipynb` 
	```conda create -y -n finetuning python=3.10 pandas numpy scikit-learn seaborn matplotlib notebook torch==2.0.1 datasets==2.14.3 transformers==4.28.1 accelerate==0.15.0 optuna==3.2.0 evaluate```
	Ran on MIT OpenMind cluster
- `suicide_risk_assessment.ipynb`
- `suicide_risk_assessment_results.ipynb` formatted table
- `ctl_cts_feature_extraction` construct-text similarity feature extraction
- `ctl_cts_ml_models` construct-text similarity machine learning models

### Table 4 (feature importance )
- `suicide_risk_assessment_results.ipynb`

<!-- ## Other results -->


