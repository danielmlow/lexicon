import pandas as pd
import numpy as np
import dill 
import datetime


ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')
input_dir = './data/input/lexicons/suicide_risk_lexicon_clinician_ratings/'
output_dir = './data/lexicons/'

import os
files = os.listdir(input_dir)

from srl_constructs import constructs_in_order

# !pip install openpyxl
# for file_i in files:
# 	annotation_df_i = pd.read_excel(input_dir+file_i, engine='openpyxl')

# type(annotation_df_i)

import sys


sys.path.append('./../../concept-tracker/')
from concept_tracker import lexicon

def avg_above_thresh(annotations, thresh = 1.3):
	annotations_avg = {}
	annotations_removed = {}
	for construct in annotations.keys():
		annotations_avg [construct] = []
		annotations_removed [construct] = []
		for token in annotations[construct].keys():
			avg_score = np.mean(annotations[construct][token])
			# var = np.var(annotations[construct][token])
			if 0 not in annotations[construct][token] and avg_score>=thresh:
				annotations_avg[construct].append(token)
			else:
				annotations_removed[construct].append(token)
	# remove _include from dict keys
	annotations_avg2 = {}
	annotations_removed2 = {}
	
	for construct in annotations_avg.keys():
		annotations_avg2[construct.replace('_include','')] = annotations_avg[construct]
		annotations_removed2[construct.replace('_include','')] = annotations_removed[construct]
		

	return annotations_avg2.copy(), annotations_removed2.copy()




# Get average across dfs
annotations = {}

annotation_df_i = pd.read_excel(input_dir+'suicide_risk_lexicon_calibrated_matched_tokens_with_unmatched_unvalidated_24-02-09T15-42-48_annotation_dc.xlsx', engine='openpyxl', sheet_name='suicide_risk_lexicon_calibrated')
annotation_df_i = annotation_df_i.iloc[4:, :	]

constructs = list(annotation_df_i.columns[1:])
constructs = [n for n in constructs if not '_' in n]
set(constructs) - set(constructs_in_order)
set(constructs_in_order) - set(constructs) 
constructs_code = list(annotation_df_i.columns[1:])
constructs_code = [n for n in constructs_code if '_include' in n]
# annotation_df_i = annotation_df_i[constructs_code]

for construct in constructs_code:
	
	annotation_df_i_construct_i = annotation_df_i[[construct.replace('_include',''), construct]].dropna()
	annotations_construct_i = annotation_df_i_construct_i.values
	annotations_construct_i = dict(zip(annotations_construct_i[:,0],[[n] for n in annotations_construct_i[:,1]]))
	annotations[construct] = annotations_construct_i
	


for file_i in ['suicide_risk_lexicon_calibrated_matched_tokens_with_unmatched_unvalidated_24-02-09T15-42-48_annotation_or.xlsx','suicide_risk_lexicon_calibrated_matched_tokens_with_unmatched_unvalidated_24-02-09T15-42-48_annotation_kb.xlsx']:
	annotation_df_i = pd.read_excel(input_dir+file_i, engine='openpyxl', sheet_name='suicide_risk_lexicon_calibrated')
	annotation_df_i = annotation_df_i.iloc[4:, :	]
	for construct in constructs_code:
	
		annotation_df_i_construct_i = annotation_df_i[[construct.replace('_include',''), construct]].dropna()
		annotations_construct_i = annotation_df_i_construct_i.values
		annotations_construct_i = dict(zip(annotations_construct_i[:,0],annotations_construct_i[:,1]))
		
		for token in annotations[construct].keys():
			score = annotations_construct_i.get(token)
			if str(score) != 'None':
				annotations[construct][token].append(score)
	




annotations_avg, annotations_removed = avg_above_thresh(annotations, thresh = 1.3)
annotations_avg_prototypical, annotations_removed_prototypical = avg_above_thresh(annotations, thresh = 3)

[(k,v) for k,v in annotations_avg_prototypical.items() if len(v)<10]


old_lexicon = dill.load(open('./data/input/lexicons/suicide_risk_lexicon_calibrated_unmatched_tokens_unvalidated_24-02-15T21-55-05.pickle', "rb"))


len(constructs_in_order)
srl = lexicon.Lexicon()
srl.name = 'Suicide Risk Lexicon v0.1'
srl.description = 'Lexicon for 50 risk factors validated by clinical experts. If you use, please cite publication: Low et al (in prep.). Creating a Suicide Risk Lexicon with Generative AI and word scores.'
srl.attributes['version'] = 'v0.1'
srl.attributes['created'] = ts
import dill
import datetime
ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S') # so you don't overwrite, and save timestamp
for i, construct in enumerate(constructs_in_order):
	print(i, construct)

	srl.constructs[construct] = {}
	for k in old_lexicon.constructs[construct].keys():
		srl.constructs[construct][k] = old_lexicon.constructs[construct][k] = []
	srl.constructs[construct]['prompt_name']  = old_lexicon.constructs[construct]['prompt_name']
	srl.constructs[construct]['examples']  = old_lexicon.constructs[construct]['examples']
	# Definition
	srl.constructs[construct]['definition'] = old_lexicon.constructs[construct]['definition']
	srl.constructs[construct]['definition_references'] = old_lexicon.constructs[construct]['definition_references']
	srl.constructs[construct]['tokens'] = annotations_avg[construct]
	
	
srl.save(f'./data/input/lexicons/suicide_risk_lexicon_validated')




len(constructs_in_order)
srl = lexicon.Lexicon()
srl.name = 'Suicide Risk Lexicon v0.1 tokens = 3/3 prototypicality according to clinicians'
srl.description = 'Lexicon for 50 risk factors validated by clinical experts. If you use, please cite publication: Low et al (in prep.). Creating a Suicide Risk Lexicon with Generative AI and word scores.'
srl.attributes['version'] = 'v0.1'
srl.attributes['created'] = ts
import dill
import datetime
ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S') # so you don't overwrite, and save timestamp
for i, construct in enumerate(constructs_in_order):
	print(i, construct)

	srl.constructs[construct] = {}
	for k in old_lexicon.constructs[construct].keys():
		srl.constructs[construct][k] = old_lexicon.constructs[construct][k] = []
	srl.constructs[construct]['prompt_name']  = old_lexicon.constructs[construct]['prompt_name']
	srl.constructs[construct]['examples']  = old_lexicon.constructs[construct]['examples']
	# Definition
	srl.constructs[construct]['definition'] = old_lexicon.constructs[construct]['definition']
	srl.constructs[construct]['definition_references'] = old_lexicon.constructs[construct]['definition_references']
	srl.constructs[construct]['tokens'] = annotations_avg_prototypical[construct]

annotations_avg_prototypical['Borderline Personality Disorder']
	
srl.save(f'./data/input/lexicons/suicide_risk_lexicon_validated_prototypical_tokens')
	

# Inter-rater agreements


import numpy as np
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

def cohens_kappa(ratings, weights=None):
    """
    Calculate Cohen's weighted kappa for two raters.

    :param ratings: a 2D NumPy array with two columns, each column representing a rater's ratings
    :param weights: a string, either 'linear' or 'quadratic', determining the type of weights to use
    :return: the weighted kappa score
    """
    kappa = cohen_kappa_score(ratings[:, 0], ratings[:, 1], weights=weights)
    return kappa

def calculate_fleiss_kappa(ratings):
    """
    Calculate Fleiss' kappa for three or more raters.

    :param ratings: a 2D NumPy array where rows represent items and columns represent raters
    :return: the Fleiss' kappa score
    """
    # Count the number of times each rating occurs per item
    n_items, n_raters = ratings.shape
    max_rating = ratings.max() + 1
    rating_matrix = np.zeros((n_items, max_rating))
    
    for i in range(n_items):
        for j in range(n_raters):
            rating_matrix[i, ratings[i, j]] += 1

    kappa = fleiss_kappa(rating_matrix, method='fleiss')
    return kappa

# Example usage
ratings_2_raters = np.array([[1, 2], [2, 2], [3, 3], [0, 1], [1, 1]])
ratings_3_raters = np.array([[1, 2, 1], [2, 2, 2], [3, 3, 2], [0, 1, 0], [1, 1, 2]])

if ratings_2_raters.shape[1] == 2:
    kappa = cohens_kappa(ratings_2_raters)
    print(f"Cohen's Weighted Kappa (2 raters): {kappa}")
elif ratings_2_raters.shape[1] >= 3:
    kappa = calculate_fleiss_kappa(ratings_2_raters)
    print(f"Fleiss' Kappa (3 or more raters): {kappa}")

if ratings_3_raters.shape[1] >= 3:
    kappa = calculate_fleiss_kappa(ratings_3_raters)
    print(f"Fleiss' Kappa (3 or more raters): {kappa}")



def binary_inter_rater_reliability(rater1, rater2):
    """
    Calculate Cohen's Kappa for binary inter-rater reliability.

    Parameters:
    rater1 (list or array): Ratings from the first rater
    rater2 (list or array): Ratings from the second rater

    Returns:
    float: Cohen's Kappa score
    """
    kappa = cohen_kappa_score(rater1, rater2)
    return kappa


cohens_kappa_all = {}
fleiss_kappa_all = {}
for construct, tokens in annotations.items():
	construct_c_annotations = list(tokens.values())
	construct_c_annotations_mode = int(np.median([len(n) for n in construct_c_annotations])) # construct_c_annotations)
	construct_c_annotations_all_annotated = []
	for token_i_annotations in construct_c_annotations:
		token_i_annotations = list(token_i_annotations)
	
		
		if len(token_i_annotations)==construct_c_annotations_mode and np.mean(token_i_annotations)>1.3:
		# 	token_i_annotations = token_i_annotations + [np.round(np.mean(token_i_annotations),0)]

			construct_c_annotations_all_annotated.append(token_i_annotations)
			
	# If I dont have them all the same shape, can't calculate
	construct_c_annotations = np.array(construct_c_annotations_all_annotated)
	construct_c_annotations = construct_c_annotations.astype(int)
	
	if construct_c_annotations.shape[1] == 2:
		# kappa = binary_inter_rater_reliability(construct_c_annotations[:,0], construct_c_annotations[:,1])
		kappa = cohens_kappa(construct_c_annotations)
		cohens_kappa_all[construct.replace('_include','')] = kappa
		# print(f"Cohen's Weighted Kappa (2 raters): {kappa}")
	elif construct_c_annotations.shape[1] >= 3:
		kappa = calculate_fleiss_kappa(construct_c_annotations)
		fleiss_kappa_all[construct.replace('_include','')] = kappa
		# print(f"Fleiss' Kappa (3 or more raters): {kappa}")

pd.DataFrame(cohens_kappa_all, index = ['weighted_kappa']).T.mean()


pd.DataFrame(fleiss_kappa_all, index = ['fleiss_kappa']).T

fleiss_kappa_all


import numpy as np
from sklearn.metrics import cohen_kappa_score

def binary_inter_rater_reliability(rater1, rater2):
    """
    Calculate Cohen's Kappa for binary inter-rater reliability.

    Parameters:
    rater1 (list or array): Ratings from the first rater
    rater2 (list or array): Ratings from the second rater

    Returns:
    float: Cohen's Kappa score
    """
    kappa = cohen_kappa_score(rater1, rater2)
    return kappa

# Example data
rater1 = [1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
rater2 = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]

# Calculate Cohen's Kappa
kappa_score = binary_inter_rater_reliability(rater1, rater2)
print(f"Cohen's Kappa for binary data: {kappa_score}")

