# Code for manuscript:
TODO: add 


TODO: move all this to a different repo and import lexicon package


# Data

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
│   │	├── suicide_risk_lexicon_gpt-4-1106-preview_dml_24-01-31T21-06-52.pickle # final one
│   │	├── suicide_risk_lexicon_gpt-4-1106-preview_dml_24-01-31T21-06-52.json
│   │	├── suicide_risk_lexicon_gpt-4-1106-preview_dml_24-01-31T21-06-52_metadata.json
│   │	├── suicide_risk_lexicon_gpt-4-1106-preview_dml_24-01-31T21-06-52.csv
│   │	├── suicide_risk_lexicon_gpt-4-1106-preview_dml_24-01-31T21-06-52_annotation.csv
├── features
│   ├── suicide_risk_lexicon
│   ├── liwc-22

output
├── descriptive_statistics
├── word_score_analysis
├── ml_performance
├── tables
├── figures





## Suicide Risk Lexicon
TODO: add files and paths
- Final lexicon v0.1 `./../data/lexicons/suicide_risk_lexicon/`
- Expert judgements: 

## Crisis Text Line data
Crisis Text Line data is private and sensitive and was obtained de-identified through a collaboration and DUA with Crisis Text Line. If you obtain permission from Crisis Text Line, we are willing to share conversation IDs so the same conversations can be analyzed.

All final figures in tables are available in `./../data/output/lexicon_paper/`

# Code for reproducing preprocessing, figures, and tables 

## Preprocessing


### Creating dataset
- Pulling data from CTL
- Building datasets

`train_test_split_ids.ipynb` in this repo. Takes downloaded data and creates datasets in the "risk assessment small" section 

- cleaning


### Creating lexicon
- General tutorial: `./../tutorials/create_lexicon.ipynb`

1. Choose constructs and seed examples: `./../data/lexicons/suicide_risk_preprocessing/suicide_risk_constructs_and_definitions.csv`
2. Create preliminary lexicon with Generative AI `create_lexicon.py`
3. Add word scores: `./../tutorials/word_scores_scattertext_ctl.ipynb`
4. Add and remove tokens from prior lexicons, thesauri: `create_lexicon.py`


TODO How it should look:
1. Choose constructs and seed examples: `./../data/lexicons/suicide_risk_preprocessing/suicide_risk_constructs_and_definitions.csv`

All of this would be in the same script and I'd save and load csv's throughout
2. Create preliminary lexicon with Generative AI `create_lexicon.py`
3. Add and remove tokens: `add_tokens.py` TODO create
3.1. Automatically obtain word scores: `./../tutorials/word_scores_scattertext_ctl.ipynb` # modify to be automatically add to certain construct after inspection
3.2. Automatically add synonyms from wordnet 
3.3. Manually obtain questionnaire items and add (perhaps loading from csv, any col with `<construct_name>_add`) will be added
3.4. Save as csv with `<construct>_include` and `<construct>_add` columns.
3.5. Manually remove irrelevant tokens, edit tokens, and any other tokens from prior lexicons: `suicide_risk_lexicon_annotate_24-01-15T19-08-09_dml.csv` 
4. Load manual annotation and updated lexicon `add_and_remove.py` 
5. extract on training set and calibrate 
6. Save final lexicon

### Feature extraction
TODO: right now it's in - `suicide_risk_assessment.ipynb` but should be its own script
This could include a calibration step if someone is using their own dataset.




## Figures 

### Figure 1: 
`risk_classification_setfit.ipynb`


### Figure 2 (Building lexicon steps)
TODO: link to google slides

### Figure 3 (Suicide Risk Lexicon network analysis)
`lexicon_network_dendrogram.ipynb`

### Figure 4 (distribution of SRL matches)
`barplot_srl_matches.py` 

### Figure 5 (Inter-rater reliability)
TODO add:


## Tables 

### Table 2 
`lexicon_source_descriptives.py`
- input: lexicon pickle file
- output: 

### Table 3 (machine learning models)
- `ctl_roberta_text.ipynb`
- `suicide_risk_assessment.ipynb`

### Table 4 (feature importance )
- `suicide_risk_assessment.ipynb`


