import warnings
import numpy as np
import pandas as pd
import sys
sys.path.append('./../../concept-tracker/')
from concept_tracker.lexicon import load_lexicon
from concept_tracker.lexicon import generate_timestamp

def is_any_substring_in_string(substrings, main_string):

    # Loop through each substring in the list
    for substring in substrings:
        # Check if the current substring is in the main string
        if substring in main_string:
            # Return True as soon as a substring is found in the main string
            return True
    # Return False if no substring is found in the main string
		
    return False

'''
	# Example usage
	substrings = ['hell', 'world', 'test']
	main_string = 'hello everyone'

	# Check if any substring is in the main string
	is_any_substring_in_string(substrings, main_string)
'''


# Table with percentage of tokens from each source
# from concept_tracker.lexicon import * # TODO: do not import everythign
# srl = load_lexicon('./../data/lexicons/suicide_risk_lexicon_gpt-4-1106-preview_dml_24-01-31T21-06-52.pickle')
# srl.constructs['Existential meaninglessness & purposelessness']['examples']  = '; '.join(srl.constructs['Existential meaninglessness & purposelessness']['examples'] )
# srl.save('./../data/lexicons/suicide_risk_lexicon_gpt-4-1106-preview_dml')

# Inspect source types


import json
with open('./../data/lexicons/suicide_risk_lexicon_gpt-4-1106-preview_dml_24-02-02T20-43-23_metadata.json', 'r') as file:
    lexicon_constructs = json.load(file)


len(lexicon_constructs.keys())

sources_all = []
for construct in lexicon_constructs.keys():
	# Add source tokens to a source type
	sources = lexicon_constructs[construct]['tokens_metadata'].keys()	
	
	for source in sources:
		source_action = lexicon_constructs[construct]['tokens_metadata'][source]['add_or_remove']
		
		if source_action == 'add':
			sources_all.append(source)

sources_all = set([' '.join(n.split(' ')[:-1]) for n in sources_all])

# TODO: move this to package

gpt4_sources = ['gpt-4-1106-preview']
word_score_sources = ['word score']
manual_sources = [
	'manually added by DML',
	'DML adding',
	'DML added',
	'DML adding manually', 
	'DML cleaning up tokens', # this includes all the gtp-4 ones
	'Added manually by DML',
	'examples by DML',

]

source_types = ['gpt-4-1106-preview', 'Word score', 'Manual'] # 

counts = {}
counts_final = {}

for construct in lexicon_constructs.keys():

	# Add source tokens to a source type
	# counts[construct] = dict(zip(source_types, [[]]*len(source_types)))

	counts[construct] = {source_type: [] for source_type in source_types}

	sources = lexicon_constructs[construct]['tokens_metadata'].keys()
	
	for source in list(sources):
		
		# source = list(sources)[4]

		# for specific addition
		len(source_tokens_i)
		source_tokens_i = lexicon_constructs[construct]['tokens_metadata'][source]['tokens']
		source_action = lexicon_constructs[construct]['tokens_metadata'][source]['add_or_remove']
		if source_action == 'add':
			# GenAI
			if is_any_substring_in_string(gpt4_sources, source):
				counts[construct]['gpt-4-1106-preview'].extend(source_tokens_i)
			# Word scores
			elif is_any_substring_in_string(word_score_sources, source):
				counts[construct]['Word score'].extend(source_tokens_i)
			# Manual
			elif is_any_substring_in_string(manual_sources, source):
				counts[construct]['Manual'].extend(source_tokens_i)
				continue
			else:
				warnings.warn(f'{construct}, {source}, Did not find source type')
				continue




for construct in lexicon_constructs.keys():
	# print(construct)
	final_tokens = lexicon_constructs[construct]['tokens']
	counts_final[construct] = {source_type: [] for source_type in source_types}
	
	for source_type in source_types:
		source_tokens = counts[construct][source_type]
		source_types
		final_intersection = len(set(final_tokens).intersection(set(source_tokens)))
		counts_final[construct][source_type] = final_intersection
	
	counts_final[construct]['Manually added'] = len(final_tokens)-(counts_final[construct]['gpt-4-1106-preview']+counts_final[construct]['Word score'])
	counts_final[construct]['Final tokens'] = len(final_tokens)
	
	

counts_final = pd.DataFrame(counts_final)		
counts_final  = counts_final.T
counts_final = counts_final.drop('Manual', axis=1) # was not able to fully capture it with the above rule, so it's the  final tokens minus (gpt4+word_score)
# Add percentage of Final tokens next to each column

for col in counts_final.columns[:-1]:
	counts_final[col] = counts_final[col] / counts_final['Final tokens']
	counts_final[col] = counts_final[col].round(2)

# add mean and std as the last row
counts_final.loc['Mean [min-max]'] =  [f"{m:.2f} [{mininum:.2f}-{maximum:.2f}]" for m,mininum, maximum in zip(counts_final.mean(), counts_final.min(), counts_final.max())]
# counts_final.loc['Mean (SD)'] = [f"{m:.2f} ({s:.2f})" for m,s in zip(counts_final.mean(), counts_final.std())] 


counts_final

seed_examples = {}
for c in lexicon_constructs.keys():
	seed_examples[c] = lexicon_constructs[c]['examples']

counts_final['Seed examples'] = counts_final.index.map(seed_examples)

# rename columns
counts_final.rename(columns={'gpt-4-1106-preview': 'GPT-4 Turbo'},inplace=True)
#Re order columns
counts_final = counts_final[['Seed examples', 'GPT-4 Turbo', 'Word score', 'Manually added', 'Final tokens']]


counts_final.to_csv('./../data/output/lexicon_paper/tables/lexicon_source_descriptives.csv')
counts_final



# construct = 'Direct self-injury'
# # are tokens different
# source_tokens1 = counts[construct]['gpt-4-1106-preview']
# source_tokens2 = counts[construct]['word score']
# set(source_tokens1).symmetric_difference(set(source_tokens2))

# final_intesection = len(set(final_tokens).intersection(set(source_tokens)))