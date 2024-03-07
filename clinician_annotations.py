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
	
	









			


