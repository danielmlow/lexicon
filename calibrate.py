import os
import sys
import pandas as pd
import numpy as np
sys.path.append('/Users/danielmlow/Dropbox (MIT)/datum/concept-tracker/') # TODO remove
from concept_tracker import lexicon
from concept_tracker.lexicon import * # TODO remove
from concept_tracker.utils import lemmatizer
from srl_constructs import constructs_in_order, categories, colors, colors_list # local
from concept_tracker.lexicon import lemmatize_tokens
from srl_constructs import constructs_in_order # local
pd.set_option("display.max_columns", None)

def gen_add_remove_dict(constructs):

	add_remove_dict = {}
	for construct in constructs:
		add_remove_dict[construct] = {'add':[], 'remove':[]}
	return add_remove_dict



def extract_context(text, token, window=(2, 3), exact_match=False, exact_match_n = 4):
	"""
	Extracts a context around the given token in the provided text.

	:param text: The input text to search for the token.
	:param token: The token to find in the text.
	:param window: A tuple representing the window size before and after the token occurrence.
	:param exact_match: A boolean flag indicating whether to perform an exact match.
	:param exact_match_n: An integer indicating the maximum length for exact matching.
	:return: A string representing the context around the token, or None if no match is found.
	"""
	# Remove punctuation and convert to lowercase for standardization
	normalized_text = text.replace(',', ' , ').replace('.', ' . ').lower()
	words = normalized_text.split()
	
	# Normalize token for matching
	normalized_token = token.lower().split()
	
	# Find the token in the list of words
	for i in range(len(words)):
		if exact_match or len(token) <= exact_match_n:
			# Check for exact match with the token
			if words[i:i+len(normalized_token)] == normalized_token:
				start_index = max(i - window[0], 0)
				end_index = min(i + len(normalized_token) + window[1], len(words))
				text = ' '.join(words[start_index:i]) + ' ' + ' '.join(words[i:end_index])
				text = text.replace(' ,', ',').replace(' .', '.').replace('  ', ' ')
				return text
		else:
			# Allow partial matches, but form a continuous match for multi-word tokens
			if ' '.join(words[i:i+len(normalized_token)]).startswith(token.lower()):
				start_index = max(i - window[0], 0)
				end_index = min(i + len(normalized_token) + window[1], len(words))
				text = ' '.join(words[start_index:i]) + ' ' + ' '.join(words[i:end_index])
				text = text.replace(' ,', ',').replace(' .', '.').replace('  ', ' ')
				return text

	# If exact_match is True and no exact match is found, or no match is found at all
	return None

def get_docs_matching_token(docs, token, window = (10,10), exact_match_n = 4, exact_match = False):
	"""
	Get documents containing the given token within a specified window. not sensitive to case. 

	Args:
		docs (list): List of documents to search through.
		token (str): The token to search for within the documents.
		window (tuple, False): The window around the token to include in the returned documents. Defaults to (10, 10).
		exact_match_n (int, optional): The maximum length of the token to be considered for an exact match. Defaults to 4.
		exact_match (bool, optional): Flag to indicate whether an exact match is required. Defaults to False.

	Returns:
		list: List of documents containing the given token within the specified window.
	"""
	
	docs_matching_token = []
	for doc in docs:
		doc_windowed = extract_context(doc, token, window=window, exact_match=exact_match, exact_match_n = exact_match_n)
		if isinstance(doc_windowed, str):
			docs_matching_token.append(doc_windowed)

			
	return docs_matching_token


# # Test the function with the given example
# input_string = "I really like bananas, fuji apples, and potatoes, butters don't like arugula"
# token = "butter"
# window = (4,3)
# extract_context(input_string, token, window, exact_match = False)
# token = "hospital"
# get_docs_matching_token(docs, token, window = (10,10), exact_match_n = 4, exact_match = False)
# TODO: dont include acronyms that are words in the lexicon

input_dir = "./data/input/lexicons/suicide_risk_lexicon_preprocessing/"



# srl2 = Lexicon()
# TODO: load new lexicon add existing info to it if it exists in the old lexicon. for key in newlexicon, 
srl = lexicon.load_lexicon(input_dir+"suicide_risk_lexicon_gpt-4-1106-preview_dml_24-01-22T03-25-06.pickle")
srl.attributes = {} # TODO: it would already exist, so you can remove from tutorial
srl.attributes['remove_from_all_constructs'] = [0, '0', "I", "I'm"] # TODO replace with this: srl.set_attribute('remove_from_all_constructs', [0, '0', "I", "I'm"]) # srl.get_attribute('remove_from_all_constructs')
# srl.get_attribute('remove_from_all_constructs')
srl.exact_match_n = 4
srl.exact_match_tokens = []


srl.add('Hospitalization', section = 'tokens', value = ['hospital', 'hospitalized','hospitalization','hospitalize', 'ER', 'emergency room','emergency','emergency department', 'psych unit', 'psych ward', 'psychiatric unit',
														 'psychiatric ward', 'psychiatric hospital', 'in and out of the clinic','in and out of the hospital', 'inpatient', 'in patient', 'clinic', 'urgent care'],
		source = ['DML adding'])
srl.add('Hospitalization', section = 'examples', value = 'hospitalized; psych ward; inpatient unit; asylum; psychiatric facility')
srl.constructs['Existential meaninglessness & purposelessness']['examples'] = '; '.join(srl.constructs['Existential meaninglessness & purposelessness']['examples'] )


assert ('suicide' in srl.constructs['Suicide exposure']['tokens']) == True 
assert ('suicide' in srl.constructs['Suicide exposure']['remove']) == True
assert ('suicide ' in srl.constructs['Suicide exposure']['tokens']) == False



# this should have been done in add() and remove(), already added it to those functions, doing it here to make sure it is done
for construct in srl.constructs.keys():
	tokens = srl.constructs[construct]['tokens']
	tokens = [n.strip() for n in tokens]
	srl.constructs[construct]['tokens'] = tokens
	for source in srl.constructs[construct]['tokens_metadata'].keys():
		tokens = srl.constructs[construct]['tokens_metadata'][source]['tokens']
		tokens = [n.strip() for n in tokens if type(n)!=float ]
		srl.constructs[construct]['tokens_metadata'][source]['tokens'] = tokens
	tokens = srl.constructs[construct]['remove']
	tokens = [n.strip() for n in tokens]
	srl.constructs[construct]['remove'] = tokens

assert ('suicide' in srl.constructs['Suicide exposure']['tokens']) == True 
assert ('suicide' in srl.constructs['Suicide exposure']['remove']) == True
assert ('suicide ' in srl.constructs['Suicide exposure']['tokens']) == False



for constructs in srl.constructs.keys():
	srl.build(construct)

assert ('suicide' in srl.constructs['Suicide exposure']['tokens']) == True 
assert ('suicide' in srl.constructs['Suicide exposure']['remove']) == True
assert ('suicide ' in srl.constructs['Suicide exposure']['tokens']) == False


srl.build('Suicide exposure')
assert ('suicide' in srl.constructs['Suicide exposure']['tokens']) == False # changed
assert ('suicide' in srl.constructs['Suicide exposure']['remove']) == True
assert ('suicide ' in srl.constructs['Suicide exposure']['tokens']) == False

# TODO: this needs to be done sometimes when loading 





# word score on Reddit data: I already put the functions in concept_tracker.utils.context.py
# ======================================================================

run_this = False


if run_this:
	rmhd = pd.read_csv('./data/input/reddit/rmhd_27subreddits_1300posts_train.csv', index_col = 0)

	control_group = [ 'ukpolitics', 'teaching',
		'personalfinance', 'mindfulness', 
		'legaladvice', 'guns', 'conspiracy', 'UKPersonalFinance', 'unitedkingdom']
	mental_health_subreddits = rmhd.subreddit.unique().tolist()
	mental_health_subreddits = [n for n in mental_health_subreddits if n not in control_group]


	# N from 1300k only mental health without mentalhealth, ForeverAlone, and Divorce
	mental_health_subreddits
	mental_health_subreddits.remove('mentalhealth')
	mental_health_subreddits.remove('ForeverAlone')
	mental_health_subreddits.remove('divorce')
	len(mental_health_subreddits)
	rmhd_mh_all = rmhd[rmhd.subreddit.isin(mental_health_subreddits)]

	# TODO: add CV from bayes_compare_language
	cv = CV(decode_error = 'ignore', min_df = 1, max_df = 0.99, ngram_range=(1,5),
					binary = False,
			stop_words =None,
		max_features = 20000)

	# Get odds ratio of each subreddit against the rest
	word_scores_per_subreddit = {}
	for subreddit in rmhd_mh_all.subreddit.unique():
		rmhd_subreddit_i = rmhd_mh_all[rmhd_mh_all.subreddit == subreddit]['post'].tolist()
		rmhd_other_i = rmhd_mh_all[rmhd_mh_all.subreddit != subreddit].sample(n=1300, random_state=42)['post'].tolist()
		zscores = word_scores.bayes_compare_language(rmhd_subreddit_i, rmhd_other_i, ngram = 1, prior=.01, 
												cv = cv, threshold_zscore=threshold_zscore, 
												l1_name = subreddit, l2_name = 'other')
		
		
		
		word_scores_per_subreddit[subreddit] = zscores[zscores['zscore']>3][::-1][['token', 'zscore']].values

	word_scores_per_subreddit_df = []
	word_scores_per_subreddit_df_summary = []
	# Table where each cell is a subreddit and each row is a token
	word_scores_per_subreddit.keys()
	for subreddit in word_scores_per_subreddit.keys():
		rmhd_subreddit_i = rmhd_mh_all[rmhd_mh_all.subreddit == subreddit]['post'].tolist()

		
		subreddit_df_i = pd.DataFrame(word_scores_per_subreddit[subreddit])
		
		subreddit_df_i['token'] = [f'{token} ({np.round(zscore, 1)})' for token, zscore in subreddit_df_i.values]
		word_scores_per_subreddit_df_summary.append([subreddit+'\n'+'\n'.join(subreddit_df_i['token'].iloc[:15].tolist())])
		subreddit_df_i['subreddit'] = [subreddit]* subreddit_df_i.shape[0]
		
		
		examples = [get_docs_matching_token(rmhd_subreddit_i, token, window = (6,6) ) for token in subreddit_df_i.values[:,0]]
		example_docs = []
		for example in examples:
			if len(example) == 0:
				example = np.nan
			else:
				example = example[:3]
			example_docs.append(example)
		
		subreddit_df_i['example'] = example_docs
		subreddit_df_i['included'] = ['']* len(subreddit_df_i)
		subreddit_df_i = subreddit_df_i.drop([0,1],axis=1)

		word_scores_per_subreddit_df.append(subreddit_df_i)
		

	word_scores_per_subreddit_df = pd.concat(word_scores_per_subreddit_df, ignore_index=True)
	word_scores_per_subreddit_df.to_csv('./data/output/tables/word_scores_per_subreddit_above_3.csv', index = False)
	word_scores_per_subreddit_df_summary = pd.DataFrame(word_scores_per_subreddit_df_summary).T
	# Flatten the DataFrame, append NaN, and reshape
	word_scores_per_subreddit_df_summary.shape
	flattened = word_scores_per_subreddit_df_summary.values.flatten()  # Convert the DataFrame to a 1D array
	# if flattened.shape%4 != 0:
	# 	flattened = np.append(flattened, np.nan)  # Append NaN to the array
	reshaped = flattened.reshape(3, 5)  # Reshape the array into 
	# Convert the reshaped array back into a DataFrame
	new_df = pd.DataFrame(reshaped)
	new_df.to_csv('./data/output/tables/word_scores_per_subreddit_15.csv', index = False, header=False)

else:
	# gen_add_remove_dict(constructs_alphabetical)
	add_and_remove = {'Alcohol use': {'add': ['alcohol', 'drinking', 'drunk', 'sober', 'beer', 'AA', 'rehab', 'withdrawal', 'binge drinking', 'sober', 'sobriety', 'AA meetings', 'bar', 'cold turkey', 'liver damage', 'vodka', 'detox', 'booze', 'quit drinking', 'stop drinking', 'bender', 'relapse', 'liquor', 'shots of', 'a few shots', 'glass of', 'a few glasses', 'urge to drink', 'black out', ],'remove': ['drink']},
	'Anxiety': {'add': ['anxiety', 'panic', 'panic attack', 'worry', 'fear', 'nervous', 'GAD', 'overwhelm', 'stress', 'my mind is racing', 'my heart is racing', 'eye twitch', 'scare', 'afraid',"can't breathe",'xanax', 'dread', 'calm down', 'fear', 'OCD', 'anxious', 'overthinking',  ], 'remove': []},
	'Bipolar Disorder': {'add': ['bipolar', 'mood episode', 'manic episode', 'unstable mood', 'mood settled down', 'mood issue', 'depression', 'depress', 'stable mood', 'ablify', 'seroquel', 'lithium', 'lamictal', 'mania', 'energy boost', 'enegy at night', 'high energy', 'wellbutrin', 'stopped sleeping', 'mood swings', 'antidepressant', 'antipsychotic', 'weight gain', 'lamotrigine', 'klonopin', 'gabapentin'], 'remove': []},
	'Borderline Personality Disorder': {'add': ['bpd', 'relationship fell apart', 'relationship erode', 'relationship issues', 'unstable relationship', 'too angry', 'angry so quickly', 'angry quickly', 'borderline', 'borderline personality disorder', 'personality disorder', 'DBT', 'self harm', 'emotions are very intense', 'manage my emotions', 'suicide', 'going to leave me', 'empty', 'anger', 'outburst','rage', 'sexual', 'mood swing', 'broke up', 'splitting', 'lack of identity', "he's my person", "favorite person", "breaking up", "fear of abandonment",'fp',  ], 'remove': ['split']},
	'Depressed mood': {'add': ['depression', 'depress', 'empty', 'sad', 'cry', 'crying','antidepressant', 'despair', 'laying in bed', 'without purpose', 'hopeless', 'empty', 'emptiness', 'miserable', 'this all sucks', 'life sucks', 'burden', 'feel like shit', ], 'remove': []},
	'Eating disorders': {'add': ['overweight', 'fasting', 'weight', 'eating less', 'not eating', 'stopped eating', 'eating disorder', 'ate a lot of food', 'food feels like poison', 'body is hungry', 'dysmorphia', 'pressured to eat', 'enough to eat', 'try not to eat', 'binge', 'look so skinny', 'thighs look big','lose weight', 'lose any more weight', 'skinny', 'underweight', 'obese', 'bulimia', 'anorexia', '0 appetite', 'zero appetite', 'big appetite', 'bloated', 'fat', 'my body', 'unhealthy', 'unhealthy weight', 'pigged out', 'ate very little', 'ate so much', 'ED', 'restrict', 'gain weight', 'gain any weight', 'calories', 'diet', 'pounds', 'lbs', 'kg', 'kilo', 'purge', 'junk food', 'massive amounts of food', 'number on the scale'], 
		'remove': ['pound', 'fast']},
	'Loneliness & isolation': {'add': ['lonely', 'alone', 'no one cares about me', 'no one to talk to', 'someone to talk to', 'unwanted', "don't feel like talking", 'loneliness', 'no friends', 'no one to hang', "don't have friends", "no one to talk", "have no one", "have no friends", "to hang out with" "I'm single", "new friends", "find someone", "feel alone", "need someone", "make friends", "shy", "isolate", "be alone", "don't have anyone", "want to love", "I'm boring", "broke up", "no social life", "need someone","left out", "have no one", "one friend", "make new friends", "no one to talk", "just want someone", "been reject", "being rejected", "disconnected", "have nobody", "had friends" ], 'remove': ["have friends"]},
	'Other substance use': {'add': ['addiction', 'drug', 'weed', 'sober', 'addict', 'smoke', 'rehab', 'attempt recovery', 'in recovery', 'pills', 'withdrawal', 'coke', 'cocaine', 'relapse', 'craving', 'cigarete', 'substance', 'cold turkey', 'sober', 'sobriety', 'xanax', 'nicotine', 'rehab', 'opiate', 'adderall', 'detox', 'vape', 'suboxone', 'benzo', 'did blow', 'pot', 'klonopin', 'overdose'], 'remove': []},
	'Psychosis & schizophrenia': {'add': ['schizophrenia', 'her voices', 'paranoid', 'hallucination', 'delusion', 'thought broadcast', 'psychosis', 'psychotic', 'paranoia', 'psychiatric hospital', 'voices in my head', 'antipsychotic', 'abilify', 'voice calling my name', 'are watching me', 'voices in my head','risperidone', 'hear voices', 'she is god', 'he is god', "I'm god", 'psych ward', 'seroquel', 'olanzapine', 'odd beliefs', 'the government is coming', 'the feds are coming', 'they are watching me', 'hallucinate' ], 'remove': []},
	'Trauma & PTSD': {'add': ['ptsd', 'trauma', 'abuse', 'trigger', 'nightmare', 'flashback', 'sexual assault', 'abusive', 'replaying the event', 'TW', 'trigger warning', 'sexually assault', 'wake up screaming', 'screaming in my sleep', 'panic attack', 'physically abused','rape', 'EMDR', 'violence', 'violent', 'assault', 'military', 'war', 'verbal abuse', 'emotional abuse', 'combat'], 'remove': []}}


	for construct in add_and_remove.keys():
		srl.add(construct, section = 'tokens', value = add_and_remove[construct]['add'], source = 'DML manually added from word score Reddit Monroe et al. (2008) method.')
		if len(add_and_remove[construct]['remove'])	> 0:
			srl.remove(construct, remove_tokens = add_and_remove[construct]['remove'], source = 'DML manually removed after looking at word score Reddit Monroe et al. (2008) method.')





# edit: Remove some first-person expression for constructs that are not self directed (e.g, 'self disgust', "self injury") such as "Burdensomeness" or "Guilt"
# ============================================================================================================



srl.constructs['Burdensomeness']['tokens']
# These will be captured by lemmatization
srl.add('Burdensomeness', source = 'DML removing without first person so they are captured by lemmatization more broadly', section = 'tokens', value = [
	"I'm a bother",
 "I'm burden",
 "I'm chore",
 "I'm disappointment",
 "I'm a drag",
 "I'm a hassle",
 "I'm a hindrance",
 "I'm a load",
 "I'm a nuisance",
 "I'm a problem",
 "I'm a weight",
 "I'm a weight on others",
 "I'm burdening",
 "I'm demanding",
 "I'm draining",
 "I'm exhausting",
 "I'm hard to love",
 "I'm high-maintenance",
 "I'm holding them back",
 "I'm in the way",
 "I'm such a burden",
 "I'm too much trouble",
 "I'm too needy",
 "I'm wasting space",

])
srl.remove('Burdensomeness', remove_tokens = [
	"I'm a bother",
	"I'm a burden",
	"I'm a chore",
	"I'm a disappointment",
	"I'm a drag",
	"I'm a hassle",
	"I'm a hindrance",
	"I'm a load",
	"I'm a nuisance",
	"I'm a problem",
	"I'm a weight",
	"I'm a weight on others",
	"I'm burdening",
	"I'm demanding",
	"I'm draining",
	"I'm exhausting",
	"I'm hard to love",
	"I'm high-maintenance",
	"I'm holding them",
	"I'm holding them back",
	"I'm in the way",
	"I'm not worth it",
	"I'm such a burden",
	"I'm too much trouble",
	"I'm too needy",
	"I'm wasting space",], source = 'DML removing without first person so they are captured by lemmatization more broadly')

srl.remove('Guilt', remove_tokens = [
	'I be sorry',
 'I be to blame',
 'I apologize',
 'I blame myself',
 'I can not forgive myself',
 'I do not mean to',
 'I do not think it through',
 'I feel bad',
 'I feel terrible',
 'I fuck up',
 'I have a conscience',
 'I make a mistake',
 'I mess up',
 'I owe you an apology',
 'I should have know well',
 'I should not have',
 'I should not have do that',
 'I should not have say that',
 'I take responsibility',
 'I take the blame',
 'I be wrong',
 'I wish I could take it back',
 'I wish I have not',
 'I be sorry',
 'I be to blame',], source = 'DML removing without first person so they are captured by lemmatization more broadly')

srl.add('Guilt', source = 'DML removing without first person so they are captured by lemmatization more broadly', section = 'tokens', value = [
	"I'm sorry",
 "I'm to blame",
 'apologize',
 'blame myself',
 'forgive myself',
 "don't mean to",
 "didn't think it through",
 'feel bad',
 'feel terrible',
 'I fucked up',
 'have a conscience',
 'make a mistake',
 'mess up',
 'owe him an apology',
 'owe her an apology',
 'owe them an apology',
 'should have known',
 "shouln't have",
 "shouln't have do that",
 "shouln't have say that",
 'take responsibility',
 'take the blame',
 "I'm wrong",
 'wish I could take it back',

])




srl.remove('Barriers to treatment', remove_tokens = [
'I am unsure of who to see',
 "I can't be fixed",
 "I can't be helped",
 "I don't have enough money for therapy",
 "I don't have time for therapy",
 "I don't need therapy",
 "I don't need treatment",
 "I don't trust psychiatrists",
 "I don't trust psychologists",
 "I haven't been reffered yet",
 'I talk to friends instead',
 "I wouldn't look for help",
 'I be unsure of who to see',
 'I can not be fix',
 'I can not be help',
 'I do not have enough money for therapy',
 'I do not have time for therapy',
 'I do not need therapy',
 'I do not need treatment',
 'I do not trust psychiatrist',
 'I do not trust psychologist',
 'I have not be reffere yet',
 'I talk to friend instead',
 'I would not look for help',
 'I be not treat well because I be old',
 'I be scared of how people will react if they find out',
 "I 've be discriminate",

], source = 'DML removing without first person so they are captured by lemmatization more broadly')

srl.add('Barriers to treatment', source = 'DML removing without first person so they are captured by lemmatization more broadly', section = 'tokens', value = [
	'unsure of who to see',
 "can't be fixed",
 "can't be helped",
 "don't have enough money for therapy",
 "don't have time for therapy",
 "don't need therapy",
 "don't need treatment",
 "don't trust psychiatrists",
 "don't trust psychologists",
 "haven't been reffered yet",
 'talk to friends instead',
 "wouldn't look for help",
 'be unsure of who to see',
 "can't be fixed",
 "can't be helped",
 "don't have enough money for therapy",
 "don't have time for therapy",
 "don't need therapy",
 "don't need treatment",
 "don't trust psychiatrists",
 "don't trust psychologists",
 "haven't be reffered yet",
 'talk to friends instead',
 "wouldn't look for help",
 "not treated well because I'm old",
 'scared of how people will react if they find out',
 "discriminated by doctor",

])



# Calibration section begins on CTL data
# ====================================================================================================
# ====================================================================================================

# load docs

import pickle
with open('./data/input/ctl/ctl_dfs_features.pkl', 'rb') as f:
	dfs = pickle.load(f)

train_df = dfs['train']['df_text'][['text', 'y']]
docs = train_df['text'].tolist()









# Before extraction: lemmatize srl tokens


# Extract
run_this = False


if run_this:
	srl = lemmatize_tokens(srl)
	feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = lexicon.extract(docs,
																					  srl.constructs,normalize = False, return_matches=True,
																					  add_lemmatized_lexicon=True, lemmatize_docs=False,
																					  exact_match_n = srl.exact_match_n,exact_match_tokens = srl.exact_match_tokens)



else:
	# Calibrate (edit): Initial check of matches_counter_d, and looking through documents. 
	# These tokens should be found as tokens not within words because they'll have many false positives
	# ============================================================================================================


	# [[],[]],
	add_and_remove_d = {
		'Active suicidal ideation & suicidal planning': [['drink myself to death', 'on the edge', "I'll die", "quick and easy"],['self']],
		'Anxiety': [['on the edge']],
		'Aggression & irritability': [['the last straw', 'name calling', 'blocked me','blocked him', 'blocked her', 'blocked them', 'on the edge', 'heated discussion', 'heated conversation'], ['heat']],
		'Alcohol use': [['drinking'],['therapy', 'counsel', 'hard']],
		'Borderline Personality Disorder': [['borderline'],['cut']],
		'Burdensomeness': [['waste of space', 'better off without me', 'am I needy', 'feel needy', 'put that upon them', 'put that on them'], ['weight']],
		'Barriers to treatment': [['no psych', 'no therap']],
		'Bipolar Disorder': [[],['elation', 'rambling']],
		'Bullying':[['mocked my','blocked me', 'name calling'], ['depression', 'anxiety', 'shame', 'isolate', 'trauma', 'fear']],
		'Direct self-injury':[[], ['overdose']],
		'Defeat & feeling like a failure': [["can't take any more of this","giving up", "I'm broken", 'feel broken'],["don't know what to do", 'break', 'beated', 'crush']],
		'Depressed mood': [['knocked down', 'darkness'],[]],
		'Eating disorders': [['binge eat', 'fasting', 'weightlift', 'overweight', 'weightloss', 'eating disorder'],['control', 'therapy', 'fast']],
		'Existential meaninglessness & purposelessness':[['not adding anything to the world'], ['life']],
		'Emotional pain & psychache': [['hollow', 'pain', 'no relief'],[]],
		'Emptiness': [["don't feel anything"],['sense']],

		'Fatigue & tired': [["can't do this anymore"],['sick of']],
		'Finances & work stress': [['million', 'billion'], ['anxiety', 'stress', 'harassment', 'conflict']],
		'Gender & sexual identity': [['transitioning', 'transgender', 'transness'], ['question']],
		'Grief & bereavement': [[],['die', 'numbness']],
		"Hopelessness":[['nothing can help',],['broken', 'break']],
		'Lethal means for suicide': [['drink myself to death', 'handgun', '9mm'], ['hanging', 'exhaust']],
		'Loneliness & isolation': [["turn to anyone", 'no real friends',"no one to turn to", "ever love me"],['cut', 'apart', 'emptiness']],
		'Other substance use': [['painkiller' ,'blackout']],
		'Other suicidal language': [["world will be better without me", "can't take any more of this", "can't do this anymore", 'no drive for life','done with this life', 'sick of my life',"I'll die","being saved", 'keeping me alive']],
		'Passive suicidal ideation': [['think about dying','stop existing', 'stop living', 'want to die', "don't really want to live", "wish I would not wake up", 'better off dead'], ['no relief','drowning',  'lost', 'pointless', 'no motivation', 'hopelessness', 'no meaning', 'trap', 'despair', 'darkness', 'trapped', 'peace', 'rest', 'lose', 'burden', 'overwhelmed', 'apathy', 'no future', 'fade away']],
		'Perfectionism': [["straight A's" ,'blackout']],
		'Physical health issues & disability':[['diagnos', 'hormone', 'high tension'], ['stress', 'phobia', 'psychosis', 'adhd', 'eating disorder', 'cold', 'ptsd', 'depression', 'anxiety']],
		"Physical abuse & violence":[[],["shoot my", 'cut my', 'harm my', 'control me']],

		'Poverty & homelessness': [[], ['want']],
		'Relationships & kinship': [['family','parent', 'boyfriend', 'father', 'son','husband', 'wife', 'married', 'marriage', 'caregiver', 'marry', 'wifey', 'hubby', 'baby', 'faithful', 'grandma', 'boyfriend']],
		'Relationship issues': [['faithful','ended it', 'leaving me' ],['family', 'ended it', 'leaving me', 'dad', 'mom','therapy', 'parent', 'love','boyfriend', 'counsel', 'father', 'son','husband', 'wife', 'married', 'marriage', 'caregiver', 'marry' ]],
		'Shame, self-disgust, & worthlessness':[["not deserving of life","I'm a freak","akward","lack of self-confidence", "world will be better without me", "never amounting to", 'waste of space',"I'm boring", "I'm a peice of shit", "what's wrong with me"], ['break']],
		'Sexual abuse & harassment': [['touched her'],['attacked me','anxiety']],
		'Suicide exposure':[[],['suicide', 'stigma']],
		'Mental health treatment':[['mental disorder','mental health disorder', 'disorder'],[]]
	}

	exact_match_tokens = ['anger', #['anger', 'changer', 'danger', 'dangerous', 'stranger', 'strangers']
					'stone', # ['gravestones', 'grindstone', 'milestone', 'stone', 'stoner']
					'panic', # hispanic
					'break', #'breakdown', 'breakdowns', 'breakfast', 'breaking', 'breaks', 'breaktime', 'breakup', 'breakups', 'heartbreak'
					'hit I',
					'trans', # add transitioning, transgender
					'owner', # downer
					'my ex', #'my exam', 'my exams', 'my existence','my expression', 'my extended', 'my extreme'
					'charge',
					'drained',
					'hit me',
					'splitting',
					'tense',
					'cycling',
				'cholo',
					'weight',
					'aching',
					'needle',
					'resent',
					'ponder',
					'pen in',
					'her pass',
					'his pass'
						'fasting']





	srl.exact_match_tokens = exact_match_tokens

	# TODO: change this to output the format of the dictionary systematically, then this might change cause we'll always have both entries.
	for construct, add_and_remove_i in add_and_remove_d.items():

		add_tokens_i = add_and_remove_i[0]
		if add_tokens_i != []:
			srl.add(construct, source = 'DML added after calibration', section = 'tokens', value = add_tokens_i)
		if len(add_and_remove_i)>1:



			remove_tokens_i = add_and_remove_i[1]
			if remove_tokens_i != []:
				srl.remove(construct, source = 'DML removed after calibration', remove_tokens = remove_tokens_i)


	# Tests
	assert ('family' in srl.constructs['Relationships & kinship']['tokens']) == True
	assert ('borderline' in srl.constructs['Borderline Personality Disorder']['tokens']) == True
	assert ('drowning' in srl.constructs['Passive suicidal ideation']['tokens']) == False
	assert ('cut' in srl.constructs['Borderline Personality Disorder']['tokens']) == False
	assert ('apathy' in srl.constructs['Passive suicidal ideation']['tokens']) == False
	assert ('suicide' in srl.constructs['Suicide exposure']['tokens']) == False
	assert ('suicide ' in srl.constructs['Suicide exposure']['tokens']) == False
	assert ('suicide' in srl.constructs['Suicide exposure']['remove']) == True



# Calibrate (false positives): View most frequent matches per construct for false positives
# ============================================================================================================

# Calibrate (edit, add, remove): Data-driven approach to finding tokens that are more likely in control group than mental health group.


# Load reddit data
# ======================================================================
rmhd = pd.read_csv('./data/input/reddit/rmhd_27subreddits_1300posts_train.csv', index_col = 0)

control_group = [ 'ukpolitics', 'teaching',
	   'personalfinance', 'mindfulness', 
	   'legaladvice', 'guns', 'conspiracy', 'UKPersonalFinance', 'unitedkingdom']
mental_health_subreddits = rmhd.subreddit.unique().tolist()
mental_health_subreddits = [n for n in mental_health_subreddits if n not in control_group]
non_suicidal = ['suicidewatch', 'depression']

# 10k with suicidal and depression
rmhd_mh = rmhd[~rmhd['subreddit'].isin(control_group)]
rmhd_mh = rmhd_mh.sample(n=10000, random_state=42)
rmhd_mh['subreddit'].value_counts()
docs_mh  = rmhd_mh['post'].values


# without suicidal and depression
rhmd_nonsuicidal = rmhd[~rmhd['subreddit'].isin(['suicidewatch', 'depression']+control_group)]
rhmd_nonsuicidal = rhmd_nonsuicidal.sample(n=10000, random_state=42)
docs_nonsuicidal  = rhmd_nonsuicidal['post'].values
rhmd_nonsuicidal['subreddit'].value_counts().shape
len(docs_nonsuicidal)

# !pip install convokit
from convokit import Corpus, download
corpus = Corpus(filename=download("reddit-corpus-small"))
corpus.print_summary_stats()
utt = corpus.random_utterance()

corpus_df = []
for utt in corpus.iter_utterances():
	utterance = [utt.text, utt.meta['subreddit']]
	utterance = pd.DataFrame(utterance, index = ['text', 'subreddit']).T
	corpus_df.append(utterance)

corpus_df = pd.concat(corpus_df)
corpus_df.shape
corpus_df.subreddit.value_counts().index.tolist()
corpus_df = corpus_df[~corpus_df.subreddit.isin(mental_health_subreddits)]
corpus_df.shape
corpus_df = corpus_df.sample(n=10000, random_state=42)
corpus_df.to_csv('./data/input/reddit/control_100_subreddits_convokit_small.csv', index = False)
corpus_df.to_csv('./../concept-tracker/concept_tracker/data/reddit/control_100_subreddits_convokit_small.csv', index = False)
docs_control = corpus_df['text'].values
len(docs_control)
constructs_alphabetical = constructs_in_order.copy()
constructs_alphabetical.sort()

# 21k SW reddit posts
reddit = pd.read_csv('./data/input/reddit/first_sw_submission_2021-07-20-23-12-27.csv',sep=',',engine='python' , index_col = 0)
reddit.columns
# Clean
reddit.shape
reddit.subreddit.value_counts()
text_col = 'text'
reddit = reddit[~reddit[text_col].isna()]
reddit.shape
reddit.subreddit.value_counts()
docs_sw = reddit.sample(n = 10000, random_state=42)[text_col].values


# Before extraction: lemmatize srl tokens
# ====================================================================================================

run_this = False


if run_this:
	
	# Before extraction: lemmatize srl tokens
	srl = lemmatize_tokens(srl)
	# Extract


	if run_this:
		feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = lexicon.extract(docs_sw,
																						srl.constructs,normalize = False, return_matches=True,
																						add_lemmatized_lexicon=True, lemmatize_docs=False,
																						exact_match_n = srl.exact_match_n,exact_match_tokens = srl.exact_match_tokens)




	# compute zscore log odds ratio
	# ======================================================================
	from concept_tracker import word_scores
	threshold_zscore = 3 #abs value
	ngram_range = (1,4)
	from sklearn.feature_extraction.text import CountVectorizer as CV
	cv = CV(decode_error = 'ignore', min_df = 1, max_df = 1, ngram_range=ngram_range,
					binary = False,
			stop_words =None,
		max_features = 15000)

	print(len(docs_mh), len(docs_control), len(docs_sw), len(docs_nonsuicidal))
	l1_docs = docs_mh.copy()
	l1_name = 'mental health'
	l2_docs = docs_control.copy()
	l2_name = 'controls'
	zscores = word_scores.bayes_compare_language(l1_docs, l2_docs, ngram = 1, prior=.01, cv = cv, threshold_zscore=threshold_zscore, l1_name = l1_name, l2_name = l2_name )

	top_matches_all = []
	# for top tokens, if they appear often in other docs, then flag token. 
	for construct in constructs_in_order:		
		top_matches = [k for k,v in matches_counter_d[construct].items() if v > 5] # only look at matches with at least 5 docs
		print(len(top_matches))
		top_matches = pd.DataFrame({'token': top_matches, 'construct':construct})
		top_matches_all.append(top_matches)

	top_matches_all = pd.concat(top_matches_all)


	# Only look at the ones that were matched
	top_matches_all_zscore = set(top_matches_all['token'].values).intersection(set(zscores['token'].values))
	matched_tokens_zscores = zscores[zscores['token'].isin(top_matches_all_zscore)]
	token_construct_map = dict(zip(top_matches_all['token'].values, top_matches_all['construct'].values))
	matched_tokens_zscores['construct'] = matched_tokens_zscores['token'].map(token_construct_map)
	matched_tokens_zscores.iloc[:30]
	len(mental_health_subreddits)
	for construct in constructs_in_order:	

		top_tokens_for_construct = matched_tokens_zscores[matched_tokens_zscores['construct'] ==construct].iloc[:20]
		display(top_tokens_for_construct)
		for token, zscore in top_tokens_for_construct[['token','zscore']].values:
			print()
			print(f'============== {token}   {zscore}')
			n = 5
			# target
			matched_docs = get_docs_matching_token(l1_docs, token, window = (n,n))
			print(len(matched_docs)/len(l1_docs), matched_docs)
			# print(len(get_docs_matching_token(docs, token, window = (n,n))))

			# control
			matched_docs = get_docs_matching_token(l2_docs, token, window = (n,n))
			print(len(matched_docs)/len(l2_docs), matched_docs)

			# third group: suicidal
			matched_docs = get_docs_matching_token(docs_sw, token, window = (n,n))
			print(len(matched_docs)/len(docs_sw), matched_docs)

else:
	add_and_remove_d = {
	'Active suicidal ideation & suicidal planning': {'add': [], 'remove': []},
	'Aggression & irritability': {'add': ['attacked'], 'remove': ['fight', 'attack']},
	'Agitation': {'add': ['tension', 'tensed', 'pacing'], 'remove': ['pace']},
	'Alcohol use': {'add': ['shot of', 'AA meeting'], 'remove': ['shot', 'sake', 'AA', 'spirit']},
	'Anhedonia & uninterested': {'add': [], 'remove': []},
	'Anxiety': {'add': [], 'remove': ['timid', 'problem', 'trial', 'gad', 'GAD']},
	'Barriers to treatment': {'add': [], 'remove': []},
	'Bipolar Disorder': {'add': [], 'remove': ['cycle']},
	'Borderline Personality Disorder': {'add': [], 'remove': ['threat', 'void', 'empty']},
	'Bullying': {'add': [], 'remove': ['vulnerable']},
	'Burdensomeness': {'add': ['drain on my'], 'remove': ['demand', 'unwanted']},
	'Defeat & feeling like a failure': {'add': [], 'remove': []},
	'Depressed mood': {'add': [], 'remove': []},
	'Direct self-injury': {'add': ['scratch my', 'scratched my', 'scratching my'], 'remove': ['scratch', 'overdose']},
	'Discrimination': {'add': ['racial profiling'], 'remove': ['unfair', 'profile', 'anti', 'separate']},
	'Eating disorders': {'add': ['skip a meal', 'skip meals'], 'remove': ['meal', 'insecure', 'my stomach']},
	'Emotional pain & psychache': {'add': [], 'remove': []},
	'Emptiness': {'add': [], 'remove': ['meaningless']},
	'Entrapment & desire to escape': {'add': ['chains', 'chained'], 'remove': ['chain', 'stick', 'doom']},
	'Existential meaninglessness & purposelessness': {'add': ['meaningless', 'no reason to live', 'feel lost'], 'remove': ['no reason', 'lose', 'lost']},
	'Fatigue & tired': {'add': ['drained'], 'remove': ['beat', 'drain', 'spend', 'spent']},
	'Finances & work stress': {'add': [], 'remove': []},
	'Gender & sexual identity': {'add': [], 'remove': ['date', 'dating']},
	'Grief & bereavement': {'add': [], 'remove': []},
	'Guilt': {'add': [], 'remove': ['wrong']},
	'Hopelessness': {'add': ['doom', 'crushing', 'crushed'], 'remove': ['crush', 'do for']},
	'Impulsivity': {'add': ['is wild'], 'remove': ['wild', 'irrational']},
	'Incarceration': {'add': ['jail sentence', 'in custody'], 'remove': ['custody']},
	'Lethal means for suicide': {'add': ['jumping off', 'burning my', 'burning charcoal', 'suicide bag'], 'remove': ['chemical', 'burn', 'jumping', 'fall', 'toxic', 'burning']},
	'Loneliness & isolation': {'add': ['unwanted', 'am single', 'single mom', 'single parent', 'single dad', 'cutt off all ties', 'cut off from'], 'remove': ['single', 'cut off']},
	'Mental health treatment': {'add': ['pump my stomach', 'stomach pumping'], 'remove': ['ECT', 'ER', 'ACT', 'CAM']},
	'Other substance use': {'add': ['pump my stomach', 'stomach pumping', 'stoner'], 'remove': ['blow', 'tolerance', 'waste', 'stone']},
	'Other suicidal language': {'add': [], 'remove': []},
	'Panic': {'add': [], 'remove': ['horror','out of control']},
	'Passive suicidal ideation': {'add': [], 'remove': []},
	'Perfectionism': {'add': [], 'remove': ['excellent', 'exact', 'first place', 'be the best', 'blackout', 'on time', 'success', 'control']},
	'Physical abuse & violence': {'add': ['beat me', 'beat her', 'get beat up', 'physically attacked'], 'remove': ['box', 'attack']},
	'Physical health issues & disability': {'add': [], 'remove': []},
	'Poverty & homelessness': {'add': ['financial aid'], 'remove': ['sentence', 'aid']},
	'Psychosis & schizophrenia': {'add': [], 'remove': []},
	'Relationship issues': {'add': ['custody', 'toxic'], 'remove': []},
	'Relationships & kinship': {'add': ['my relative'], 'remove': ['bro', 'relative']},
	'Rumination': {'add': [], 'remove': []},
	'Sexual abuse & harassment': {'add': [], 'remove': []},
	'Shame, self-disgust, & worthlessness': {'add': [], 'remove': []},
	'Sleep issues': {'add': ['bad dream'], 'remove': ['dream']},
	'Social withdrawal': {'add': [], 'remove': []},
	'Suicide exposure': {'add': [], 'remove': []},
	'Trauma & PTSD': {'add': [], 'remove': []}}


	
	for construct in add_and_remove_d.keys():

		if len(add_and_remove_d[construct]['add'])>0:
			source = 'Added: Data-driven approach to detecting false positives by looking at tokens that appear more frequently in Reddit control group than Reddit suicidal posts'
			srl.add(construct, section = 'tokens', value = add_and_remove_d[construct]['add'], source = source)
		if len(add_and_remove_d[construct]['remove'])>0:
			source = 'Removed: Data-driven approach to detecting false positives by looking at tokens that appear more frequently in Reddit control group than Reddit suicidal posts'
			srl.remove(construct, remove_tokens = add_and_remove_d[construct]['remove'], source = source)
	
# os.listdir()
# srl.save(f'./data/input/lexicons/suicide_risk_lexicon_preprocessing/suicide_risk_lexicon_gpt-4-1106-preview_dml', ['pickle'], order = constructs_in_order, timestamp = True)

# # zscores on CTL data
# # ========================================================================================

# import pickle
# with open('./data/input/ctl/ctl_dfs_features.pkl', 'rb') as f:
# 	dfs = pickle.load(f)

# train_df = dfs['train']['df_text'][['text', 'y']]

# l1_docs = train_df[train_df['y']==4]['text'].values
# l1_name = 'high'
# l2_docs = train_df[train_df['y']==2]['text'].values
# l2_name = 'low'

# l3_docs = train_df[train_df['y']==3]['text'].values

# cv = CV(decode_error = 'ignore', min_df = 1, max_df = 0.99, ngram_range=(1,3),
# 				binary = False,
# 		stop_words =None,
# 	  max_features = 15000)
# zscores = word_scores.bayes_compare_language(l1_docs, l2_docs, ngram = 1, prior=.01, cv = cv, threshold_zscore=threshold_zscore, l1_name = l1_name, l2_name = l2_name )
# zscores

# These tokens are particularly useful






# calibrate on CTL data
# ================================================================




# TODO make sure that if I add tokens to exact_match_tokens that they aren't lemmatized and added to tokens lemmatized. 
srl.exact_match_tokens += ['othering']


srl.remove('Discrimination', remove_tokens = ['othere', 'threaten', 'harass', 'threatened', 'harassed', 'harassment'], source = 'DML: Not sure why that was there')





import random

# lemmatize_tokens = 0
# from concept_tracker.lexicon import lemmatize_tokens





run_this = False


if run_this:

	srl = lemmatize_tokens(srl) # TODO: integrate this to class: self.lemmatize_tokens() adds tokens_lemmatized

	feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = lexicon.extract(train_df['text'].tolist(),
																						srl.constructs,normalize = False, return_matches=True,
																						add_lemmatized_lexicon=True, lemmatize_docs=False,
																						exact_match_n = srl.exact_match_n,exact_match_tokens = srl.exact_match_tokens)


	
	train_df_features = pd.concat([train_df, feature_vectors],axis=1)


	# 1. Check top matches_counter_d
	
	matches_counter_d2 = {}
	for k in constructs_in_order:
		matches_counter_d2[k] = matches_counter_d[k].copy()

	matches_counter_d = {k: v for k, v in sorted(matches_counter_d2.items(), key=lambda item: constructs_in_order.index(item[0])) }

	threshold = 2
	for construct in matches_counter_d.keys():
		print(construct, '============================')
		
		matches_counter_d_top_matches = {k: v for k, v in matches_counter_d[construct].items() if v > threshold}

		
		for token in matches_counter_d_top_matches:
			print(token, f'({matches_counter_d2[construct][token]} matches)')
			matched_docs = get_docs_matching_token(train_df['text'].tolist(), token, exact_match = False, window=(10,10))
			if len(matched_docs) > 3:
				
				print(random.sample(matched_docs, 3))
			else:
				print(matched_docs[:3])

			print()


		# 2. Look through 20-40 for false negatives
	srl.constructs['Direct self-injury']


	# 'digging my nails in my skin', 'pinching', scratching, pulling hair

	srl.constructs['Active suicidal ideation & suicidal planning']['tokens']
else:
	# gen_add_remove_dict(constructs_alphabetical)
	add_and_remove_d = {
	'Active suicidal ideation & suicidal planning': {'add': [], 'remove': ['drown', 'kill I']},
	'Aggression & irritability': {'add': [], 'remove': ['intrusive', 'anti', 'on the edge', 'disturb', 'oppose', 'block he', 'put out']},
	'Agitation': {'add': [], 'remove': []},
	'Alcohol use': {'add': ['margarita'], 'remove': ['cheer', 'barley', 'patron', 'tonic']},
	'Anhedonia & uninterested': {'add': [], 'remove': []},
	'Anxiety': {'add': ['alarmed'], 'remove': ['trouble I', 'alarm']},
	'Barriers to treatment': {'add': ['put on drugs'], 'remove': []},
	'Bipolar Disorder': {'add': [], 'remove': []},
	'Borderline Personality Disorder': {'add': [], 'remove': ['suicide', 'sexual', ]},
	'Bullying': {'add': ['threatened me', 'shamed me', 'pick on me'], 'remove': ['hit I', 'threat', 'blocked me', 'torture','powerless', 'pick on', 'mean to I', 'block he', 'block she']},
	'Burdensomeness': {'add': [], 'remove': []},
	'Defeat & feeling like a failure': {'add': ["can't do this anymore"], 'remove': []},
	'Depressed mood': {'add': ['sadening', 'feeling blue'], 'remove': ['hopeless', 'burden', 'blue', 'dysphoria', ]},
	'Direct self-injury': {'add': [], 'remove': []},
	'Discrimination': {'add': ['racism'], 'remove': ['tribe']},
	'Eating disorders': {'add': ["I'm heavy", "hungry", "not hungry"], 'remove': ['mirror', 'unhealthy']},
	'Emotional pain & psychache': {'add': [], 'remove': []},
	'Emptiness': {'add': [], 'remove': []},
	'Entrapment & desire to escape': {'add': ['on the edge'], 'remove': ['hem', 'undo']},
	'Existential meaninglessness & purposelessness': {'add': ['my life is insignificant'], 'remove': ['drift', 'disconnect', 'empty', 'drift', 'insignificant', 'vain', 'wander']},
	'Fatigue & tired': {'add': ['dozing off'], 'remove': ['dozing']},
	'Finances & work stress': {'add': [], 'remove': ['communication', 'deplete']},
	'Gender & sexual identity': {'add': ["I'm straight", "attractive", "LGBTQ", 'femenine'], 'remove': ['straight', 'attract', 'questioning']},
	'Grief & bereavement': {'add': ['brother died', 'sister died', 'tragic', 'passed away'], 'remove': []},
	'Guilt': {'add': [], 'remove': []},
	'Hopelessness': {'add': [], 'remove': []},
	'Hospitalization': {'add': ['mental health facility', 'psychiatric facility', 'asylum', 'I was sedated', 'mental hospital'], 'remove': []},
	'Impulsivity': {'add': [], 'remove': ['passionate', 'unexpected', 'inconsiderate']},
	'Incarceration': {'add': [], 'remove': ['lock up']},
	'Lethal means for suicide': {'add': [], 'remove': ['benzo', 'narcotic']},
	'Loneliness & isolation': {'add': ['human interaction', 'avoid talking', 'drifting apart'], 'remove': ['avoid I', 'care about I']},
	'Mental health treatment': {'add': ['PCP', 'primary care'], 'remove': []},
	'Other substance use': {'add': ['oxy', 'in recovery', 'withdrawals', 'withdrawls'], 'remove': ['drink', 'drunk', 'drank', 'recovery', 'binge', 'withdraw', 'crave', "I'm high", "PCP", "downer", 'tonic']},
	'Other suicidal language': {'add': [], 'remove': []},
	'Panic': {'add': [], 'remove': []},
	'Passive suicidal ideation': {'add': ["don't want to wake up"], 'remove': []},
	'Perfectionism': {'add': [], 'remove': ['guilt']},
	'Physical abuse & violence': {'add': ['beat me', 'beat up', 'beating up', 'cops', 'shove'], 'remove': ['beat', 'batter', 'hurt me', 'hurt I', 'threw', 'kicked', 'beat I', 'beat', 'beating', 'punish', 'kick I', 'throw I', 'push I', 'cut I', 'cut me', 'overpower', 'force I', 'mistreat']},
	'Physical health issues & disability': {'add': ['kidney disease', 'liver disease'], 'remove': ['weakness']},
	'Poverty & homelessness': {'add': [], 'remove': ['beg', 'deprive']},
	'Psychosis & schizophrenia': {'add': ['disowned me'], 'remove': ['crazy', 'lunatic']},
	'Relationship issues': {'add': ['blocked me', 'blackmail', 'mistreat'], 'remove': []},
	'Relationships & kinship': {'add': ['sibling'], 'remove': ['supportive', 'sible']},
	'Rumination': {'add': ['rumination'], 'remove': []},
	'Sexual abuse & harassment': {'add': ["sex", "took advantage of m"], 'remove': ['straight', 'PTSD', 'pussy', 'dick', 'touch I', 'take advantage of m', "cum"]},
	'Shame, self-disgust, & worthlessness': {'add': [], 'remove': []},
	'Sleep issues': {'add': [], 'remove': []},
	'Social withdrawal': {'add': [], 'remove': ['withdraw', 'inaccessible']},
	'Suicide exposure': {'add': ['13 reasons why','Anthony Bourdain', 'Avici','Jonathan Brandis', 'Robin Williams', ], 'remove': ['mental health crisis', 'commit suicide', 'die by suicide']},
	'Trauma & PTSD': {'add': [], 'remove': []}
	}

	add_to_exact_match_tokens = ['sibling', 'siblings', 'the bar', 'shots of', 'laced', "I'm a cow", "price"]
	srl.exact_match_tokens = srl.exact_match_tokens + add_to_exact_match_tokens
	
	# TODO: to avoid overwriting add and remove. We need to add metadata to the source name automatically, the author, the action. etc. 

	for construct in add_and_remove_d.keys():
		if len(add_and_remove_d[construct]['add'])>0:
			source = 'Added: Select 3 documents for each token that was matched at least 3 times in training set. Removed false positives (if did not make sense in any of the 3 examples). Added false negatives that I found '
			srl.add(construct, section = 'tokens', value = add_and_remove_d[construct]['add'], source = source)
		if len(add_and_remove_d[construct]['remove'])>0:
			source = 'Removed: Select 3 documents for each token that was matched at least 3 times in training set. Removed false positives (if did not make sense in any of the 3 examples). Added false negatives that I found '
			srl.remove(construct, remove_tokens = add_and_remove_d[construct]['remove'], source = source)

	construct = 'Suicide exposure'
	

	srl.constructs[construct]['tokens']
	source = list(srl.constructs[construct]['tokens_metadata'].keys())[-2:]
	
	assert ('13 reasons why' in srl.constructs['Suicide exposure']['tokens']) == True
	assert ('lunatic' not in srl.constructs['Psychosis & schizophrenia']['tokens']) == True


	



# Look through docs for false negatives
# ===================================================
run_this = False

if run_this:
	docs_subset = train_df['text'].sample(n=100).values
	srl = lemmatize_tokens(srl) # TODO: integrate this to class: self.lemmatize_tokens() adds tokens_lemmatized
	feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = lexicon.extract(docs_subset,
																						srl.constructs,normalize = False, return_matches=True,
																						add_lemmatized_lexicon=True, lemmatize_docs=False,
																						exact_match_n = srl.exact_match_n,exact_match_tokens = srl.exact_match_tokens)
	for i in range(len(docs_subset)):
		print(docs_subset.values[i])
		constructs_alphabetical = constructs_in_order.copy()
		constructs_alphabetical.sort()
		display(pd.DataFrame(matches_per_doc[i])[constructs_alphabetical])
		print()

else:
	add_and_remove_d = {
	'Active suicidal ideation & suicidal planning': {'add': ['OD',"I'll be dead","want to jump", "over dose", "run in front of traffic", "shooting myself", "make it look like a accident", "will be dead"], 'remove': []},
	'Aggression & irritability': {'add': [], 'remove': []},
	'Agitation': {'add': [], 'remove': []},
	'Alcohol use': {'add': [], 'remove': []},
	'Anhedonia & uninterested': {'add': [], 'remove': []},
	'Anxiety': {'add': [], 'remove': []},
	'Barriers to treatment': {'add': [], 'remove': []},
	'Bipolar Disorder': {'add': ["haven't slept"], 'remove': []},
	'Borderline Personality Disorder': {'add': [], 'remove': []},
	'Bullying': {'add': [], 'remove': []},
	'Burdensomeness': {'add': ["I'm annoying", "be better without me"], 'remove': []},
	'Defeat & feeling like a failure': {'add': ["nothing works","don't know what to do"], 'remove': []},
	'Depressed mood': {'add': ['cried'], 'remove': []},
	'Direct self-injury': {'add': ["slice my wrist"], 'remove': []},
	'Discrimination': {'add': ['racist'], 'remove': []},
	'Eating disorders': {'add': ["throwing up"], 'remove': []},
	'Emotional pain & psychache': {'add': [], 'remove': []},
	'Emptiness': {'add': [], 'remove': []},
	'Entrapment & desire to escape': {'add': ['running out of options'], 'remove': []},
	'Existential meaninglessness & purposelessness': {'add': ["don't care about life", "what's the point of life"], 'remove': []},
	'Fatigue & tired': {'add': [], 'remove': []},
	'Finances & work stress': {'add': [], 'remove': []},
	'Gender & sexual identity': {'add': [], 'remove': []},
	'Grief & bereavement': {'add': ['friends died'], 'remove': []},
	'Guilt': {'add': [], 'remove': []},
	'Hopelessness': {'add': [], 'remove': []},
	'Hospitalization': {'add': [], 'remove': []},
	'Impulsivity': {'add': [], 'remove': []},
	'Incarceration': {'add': [], 'remove': []},
	'Lethal means for suicide': {'add': ['OD',"over dose","slice my wrist", "run in front of traffic", "shooting myself", "muscle relaxer", "weapon", "anti freeze", "antifreeze", "hanging"], 'remove': []},
	'Loneliness & isolation': {'add': ["everybody would forget about me", "never there for me", "people are not caring", "no partner"], 'remove': []},
	'Mental health treatment': {'add': ["need help", 'medicine', 'therapist', 'dosage', 'prescribe', 'behavioral health', 'facility'], 'remove': []},
	'Other substance use': {'add': ['drugs', 'dosage'], 'remove': []},
	'Other suicidal language': {'add': ["I'll be dead", "can't live like this", "needs to be dead", "kill me", "dying", 'breaking point', "tired of life", "ready to die", "will be dead", "it's my time to go", "won't matter if I die", "nothing to lose"], 'remove': []},
	'Panic': {'add': ["dissociation", "dissociate", "need help", "meltdown"], 'remove': []},
	'Passive suicidal ideation': {'add': ["permanent rest", "want it all to end", "tired of life", "don't care about life"], 'remove': []},
	'Perfectionism': {'add': [], 'remove': []},
	'Physical abuse & violence': {'add': ["hits me", "stabbing me"], 'remove': []},
	'Physical health issues & disability': {'add': ["throwing up"], 'remove': []},
	'Poverty & homelessness': {'add': [], 'remove': []},
	'Psychosis & schizophrenia': {'add': ['spying on me', 'losing my mind'], 'remove': []},
	'Relationship issues': {'add': ["had left me"], 'remove': []},
	'Relationships & kinship': {'add': ['sibling'], 'remove': []},
	'Rumination': {'add': ['want my brain to shut off'], 'remove': []},
	'Sexual abuse & harassment': {'add': [], 'remove': []},
	'Shame, self-disgust, & worthlessness': {'add': ['embarrass'], 'remove': []},
	'Sleep issues': {'add': [], 'remove': []},
	'Social withdrawal': {'add': ["don't trust anyone"], 'remove': []},
	'Suicide exposure': {'add': [], 'remove': []},
	'Trauma & PTSD': {'add': [], 'remove': []}}

	exact_match_tokens = ['hanging']
	srl.exact_match_tokens+=exact_match_tokens
	srl.exact_match_tokens = list(np.unique(srl.exact_match_tokens))

	'''
	[print(n) for n in ['OD',"over dose","slice my wrist", "run in front of traffic", "shooting myself", "muscle relaxer", "weapon", "anti freeze", "antifreeze", "hanging"]]

	'''


	for construct in add_and_remove_d.keys():
		if len(add_and_remove_d[construct]['add'])>0:
			source = 'Added: Looked through docs for false negatives'
			srl.add(construct, section = 'tokens', value = add_and_remove_d[construct]['add'], source = source)
		if len(add_and_remove_d[construct]['remove'])>0:
			source = 'Added: Looked through docs for false negatives'
			srl.remove(construct, remove_tokens = add_and_remove_d[construct]['remove'], source = source)


	from concept_tracker.lexicon import lemmatize_tokens
	srl = lemmatize_tokens(srl) 
	srl.save('./data/input/lexicons/suicide_risk_lexicon_preprocessing/suicide_risk_lexicon_calibrated_unmatched_tokens_unvalidated')


# Remove unmatched tokens
# ================================================================================
import copy

run_this = False
if run_this: 			
	feature_vectors_sw, matches_counter_d_sw, matches_per_doc_sw, matches_per_construct_sw  = lexicon.extract(docs_sw,
																					srl.constructs,normalize = False, return_matches=True,
																					add_lemmatized_lexicon=True, lemmatize_docs=False,
																					exact_match_n = srl.exact_match_n,exact_match_tokens = srl.exact_match_tokens)






	

	feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = lexicon.extract(train_df['text'].tolist(),
																						srl.constructs,normalize = False, return_matches=True,
																						add_lemmatized_lexicon=True, lemmatize_docs=False,
																						exact_match_n = srl.exact_match_n,exact_match_tokens = srl.exact_match_tokens)


	train_df_features = pd.concat([train_df, feature_vectors],axis=1)

	
	
	all_matched_tokens = []
	for c in matches_counter_d:
		all_matched_tokens += matches_counter_d[c].keys()

	all_matched_tokens_sw = []
	for c in matches_counter_d_sw:
		all_matched_tokens_sw += matches_counter_d_sw[c].keys()

	# original_vs_matched = {}
	# for c in srl.constructs.keys():
	# 	tokens = srl.constructs[c]['tokens'] 
	# 	matched_tokens = [n for n in tokens if n in all_matched_tokens]
	# 	matched_tokens_sw = [n for n in tokens if n in all_matched_tokens_sw]
	# 	matched_tokens_ctl_sw = [n for n in tokens if (n in all_matched_tokens or n in matched_tokens_sw)]
	# 	original_vs_matched[c] = [len(tokens), len(matched_tokens),len(matched_tokens_ctl_sw)] # original tokens, matched_tokens
	# original_vs_matched = pd.DataFrame(original_vs_matched, index = ['Total', 'Matched CTL', 'Matched CTL and SW']).T
	# original_vs_matched['%'] = (original_vs_matched['Matched CTL and SW'] / original_vs_matched['Total']).round(2)
	# original_vs_matched = original_vs_matched.T[constructs_in_order].T

	

	all_matched_tokens_ctl_sw = list(np.unique(all_matched_tokens+all_matched_tokens_sw))
	len(all_matched_tokens_ctl_sw)

	srl_matched = copy.deepcopy(srl)

	from concept_tracker.utils import lemmatizer

	# TODO re-do the count with all this. 
	# TODO check whether token is within other token

	add_and_remove_d = gen_add_remove_dict(constructs_in_order)
	original_vs_matched = []
	for construct in constructs_in_order:
		tokens = srl.constructs[construct]['tokens']
		# if it isn't in matched tokens, remove it
		tokens_to_remove = [n for n in tokens if n.lower() not in all_matched_tokens_ctl_sw] # tokens that are in the unmatched list are all lower case
		# now check if the lemmatized version was found, if it was, include the original token, because it is not in tokens and we don't want to validated all lemmatized forms
		tokens_to_remove_lemmatized = lemmatizer.spacy_lemmatizer(tokens_to_remove)
		tokens_to_remove_lemmatized = [' '.join(n).replace(' - ', '-').replace('-',' ').replace('  ', ' ') for n in tokens_to_remove_lemmatized]
		
		tokens_to_remove_final = tokens_to_remove.copy()
		for token, token_lemmatized in zip(tokens_to_remove,tokens_to_remove_lemmatized):	

			if token!= token_lemmatized and (token_lemmatized.lower() in all_matched_tokens_ctl_sw): 
				print(token,  ' ------ ', token_lemmatized)
				

				# example: "stop existing" > stop exist which is matched. 
				try: tokens_to_remove_final.remove(token)
				except: 
					print('original unlemmatized token was not in tokens_to_remove_final:', token, 'making sure lemmatized form is not removed:', token_lemmatized)
					tokens_to_remove_final.remove(token_lemmatized)
				
		print('removing these tokens')	
		print(construct, tokens_to_remove_final)
		add_and_remove_d[construct]['remove'] = tokens_to_remove
		print()


		tokens_to_keep = len(tokens)-len(tokens_to_remove_final)
		perc_of_original = tokens_to_keep/len(tokens)
		original_vs_matched.append([construct, len(tokens), len(tokens_to_remove_final), tokens_to_keep, np.round(perc_of_original*100,1)])
		

	original_vs_matched = pd.DataFrame(original_vs_matched, columns = ['Construct', 'Original', 'Unmatched in CTL and SW', 'Matched CTL and SW', '% of original'])

	original_vs_matched = original_vs_matched.set_index('Construct')
	means = original_vs_matched.mean()
	mins = original_vs_matched.min()
	maxs = original_vs_matched.max()

	# Create a new row with the formatted string "mean [min; max]"
	new_row = {col: f"{means[col]:.2f} [{mins[col]}; {maxs[col]}]" for col in original_vs_matched.columns}
	new_row = pd.DataFrame(new_row, index = ['Mean [min-max]'])

	# Append the new row to the DataFrame
	original_vs_matched_df = pd.concat([original_vs_matched, new_row], axis=0)
	
	original_vs_matched_df.to_csv('./data/output/tables/srl_original_vs_matched_tokens.csv')

	# construct = 'Active suicidal ideation & suicidal planning'
	# matches_counter_d[construct].keys()
	# token = 'gave my things away' # where lemma is found
	# token in srl.constructs[construct]['tokens']
	# 'attempt' in srl.constructs[construct]['tokens']
	# 'stop exist' in srl.constructs[construct]['tokens_lemmatized']
	# The problem is the lemma won't be validated. so we need to add tokens that lemmatized are matched. 
	# TODO: wonder if punctuation and lower case affect things as well. 

	# [n for n in all_matched_tokens_stl_sw if token.lower() in n]
	# [n for n in docs_sw if token.lower() in n.lower()]
	# [n for n in train_df['text'].values if token.lower() in n.lower()]


	# # TODO: when I'm happy with results, then remove those tokens 
	# # srl_matched.remove(construct, remove_tokens = tokens_to_remove_final, source ='Remove: these tokens were not found in CTL training data nor 10k r/SuicideWatch posts')

	# assert ('13 reasons why' in srl.constructs['Suicide exposure']['tokens']) == True
	# assert ('suicide grief'  in srl.constructs['Suicide exposure']['tokens']) == True
	# assert ('13 reasons why' in srl_matched.constructs['Suicide exposure']['tokens']) == True
	# assert ('suicide grief' not in srl_matched.constructs['Suicide exposure']['tokens']) == True
	import json
	with open('./data/input/lexicons/suicide_risk_lexicon_preprocessing/unmatched_tokens.json', 'w') as json_file:
		json.dump(add_and_remove_d, json_file, indent=4)  # Using 4 spaces for indentation
	
	with open(f'./data/input/lexicons/suicide_risk_lexicon_preprocessing/unmatched_tokens_edited_{generate_timestamp()}.json', 'w') as json_file:
		json.dump(add_and_remove_d, json_file, indent=4)  # Using 4 spaces for indentation

		

else:
	
	import copy
	# Tokens that were unmatched. 
	# manually removed some so they were kept in the lexicon. Careful with json formatting: you can sue jsonlint to find mistakes (trailing commas)
	with open('./data/input/lexicons/suicide_risk_lexicon_preprocessing/unmatched_tokens_24-02-15T23-25-11_edited.json', 'r') as json_file:
		add_and_remove_d =  json.load(json_file)

	srl_matched = copy.deepcopy(srl)
	for construct in add_and_remove_d.keys():
		if len(add_and_remove_d[construct]['remove'])>0:
			source = 'DML removed by removing unmatched tokens in CTL and SW, but then manually kept some by erasing lines of the json file'
			srl_matched.remove(construct, remove_tokens = add_and_remove_d[construct]['remove'], source = source)

	# Test
	# construct = 'Finances & work stress'
	# 'colleagues' in srl_matched.constructs[construct]['remove']
	# 'colleagues' not in srl_matched.constructs[construct]['tokens']
	# source = list(srl_matched.constructs[construct]['tokens_metadata'].keys())[-2:]
	
	srl_matched = lemmatize_tokens(srl_matched) 
	srl_matched.save('./data/input/lexicons/suicide_risk_lexicon_preprocessing/suicide_risk_lexicon_calibrated_matched_tokens_unvalidated')

	
	


	# TODO: see if tokens being removed make sense
	# TODO: create new SRL withou removed tokens. maybe just remove from

	


	
# Rank features for validation 
# ===================================================

# Encode tokens as embeddings 

run_this = False

if run_this:
	srl_matched = lexicon.load_lexicon('./data/input/lexicons/suicide_risk_lexicon_preprocessing/suicide_risk_lexicon_calibrated_matched_tokens_unvalidated_24-02-15T22-12-18.pickle')
	import tensorboard
	from sentence_transformers import SentenceTransformer, util 
	embeddings_name = 'all-MiniLM-L6-v2'
	# Encode the documents with their sentence embeddings 
	# a list of pre-trained sentence transformers
	# https://www.sbert.net/docs/pretrained_models.html
	# https://huggingface.co/models?library=sentence-transformers
	
	# all-MiniLM-L6-v2 is optimized for semantic similarity of paraphrases
	sentence_embedding_model = SentenceTransformer(embeddings_name)       # load embedding
	
	# TODO: Change max_seq_length to 500
	# Note: sbert will only use fewer tokens as its meant for sentences, 
	print(sentence_embedding_model .max_seq_length)

	import dill 
	lexicon_embeddings = dill.load(open('./data/input/lexicons/embeddings_lexicon-tokens_all-MiniLM-L6-v2.pickle', "rb"))	
	len(lexicon_embeddings.keys())
	total_tokens = []
	for construct in constructs_in_order:
		tokens = srl_matched.constructs[construct]['tokens']
		print(construct, tokens)
		total_tokens.extend(tokens)
	print(len(total_tokens))
	srl.attributes['remove_from_all_constructs'] = [0, '0'] # TODO replace with this: srl.set_attribute('remove_from_all_constructs', [0, '0', "I", "I'm"]) # srl.get_attribute('remove_from_all_constructs')
	srl_matched.get_attribute('remove_from_all_constructs')
	tokens_to_encode = [n for n in total_tokens if n not in lexicon_embeddings.keys()]
	print(len(tokens_to_encode))
	
	lexicon_embeddings_new = sentence_embedding_model.encode(tokens_to_encode, convert_to_tensor=True,show_progress_bar=True)
	
	
	list(lexicon_embeddings.items())[0][1]
	type(lexicon_embeddings_new[0].numpy()) == type(list(lexicon_embeddings.items())[0][1])
	lexicon_embeddings_new = [n.numpy() for n in lexicon_embeddings_new]
	# a lot lighter

	lexicon_embeddings.update(dict(zip(tokens_to_encode, lexicon_embeddings_new)))
	dill.dump(lexicon_embeddings, open('./data/input/lexicons/embeddings_lexicon-tokens_all-MiniLM-L6-v2_new.pickle', "wb"))

	
	# TODO: make sure this was added before (I added it to the top) 
	
	srl_matched.constructs['Existential meaninglessness & purposelessness']['examples'] = '; '.join(srl_matched.constructs['Existential meaninglessness & purposelessness']['examples'] )

	# Make sure examples are in list. 
	for construct in constructs_in_order:
		examples = srl_matched.constructs[construct]['examples']
		examples = [n.strip() for n in examples.split(';')]
		tokens = srl_matched.constructs[construct]['tokens']
		tokens += examples
		srl_matched.constructs[construct]['tokens'] = list(np.unique(tokens))

	# compute similarities with average of example embeddings	
	average_embedding_of_examples = {}
	for construct in constructs_in_order:
		examples = srl_matched.constructs[construct]['examples']
		examples = [n.strip() for n in examples.split(';')]
		examples_not_encoded = [n for n in examples if n not in lexicon_embeddings.keys()]
		if len(examples_not_encoded) > 0:
			examples_not_encoded_embeddings = sentence_embedding_model.encode(examples_not_encoded, convert_to_tensor=True,show_progress_bar=True)
			examples_not_encoded_embeddings = [n.numpy() for n in examples_not_encoded_embeddings]
			lexicon_embeddings.update(dict(zip(examples_not_encoded, examples_not_encoded_embeddings)))
		example_embeddings = [lexicon_embeddings[n] for n in examples]
		average_embedding_of_examples[construct] = np.mean(example_embeddings, axis = 0)

	# For each constructs, rank the similarity between each token and the average embedding of the examples
	from sklearn.metrics.pairwise import cosine_similarity

	use_average_of_example = False #False, because empirically it's better the rank is related to just one word
	closest_to_first = False # if False, do recursive similarity which works better. Humans go in order, so once theyve judged a token, it's best to show the most related to the prior one
	for construct in constructs_in_order:

		if use_average_of_example:
			reference_embedding = average_embedding_of_examples.get(construct)
		else:
			# embedding of first example
			examples = srl_matched.constructs[construct]['examples']
			examples = [n.strip() for n in examples.split(';')]
			example_1 = examples[0]
			reference_embedding = lexicon_embeddings.get(example_1)


		tokens = srl_matched.constructs[construct]['tokens']
		tokens_embeddings = [lexicon_embeddings.get(n) for n in tokens]
		tokens_embeddings = dict(zip(tokens, tokens_embeddings))
		
		
		if closest_to_first:
			
			# compute cosine similarity
			similarity_scores_matrix = cosine_similarity(reference_embedding.reshape(1,-1), np.array(list(tokens_embeddings.values())))

			# Flatten the resulting matrix to a 1D array of scores
			similarity_scores = similarity_scores_matrix.flatten()

			# Rank tokens based on their cosine similarity
			ranked_tokens = sorted(tokens_embeddings.keys(), key=lambda token: similarity_scores[list(tokens_embeddings.keys()).index(token)], reverse=True)

			srl_matched.constructs[construct]['tokens_by_similarity'] = ranked_tokens

		else:
			# Find closest recursively
			tokens_list =  tokens.copy()

			examples = srl_matched.constructs[construct]['examples']
			examples = [n.strip() for n in examples.split(';')]
			example_1 = examples[0]
			start_token = example_1
			# Initialize the ordered list with the starting token
			ordered_tokens = [start_token]
			# Remove the starting token from the tokens list to avoid comparing it with itself
			tokens_list.remove(start_token)
			while tokens_list:
				# Get the last token's embedding in the ordered list
				last_token_embedding = tokens_embeddings[ordered_tokens[-1]].reshape(1, -1)
				# Calculate cosine similarity with remaining tokens
				similarities = {token: cosine_similarity(last_token_embedding, tokens_embeddings[token].reshape(1, -1))[0][0] for token in tokens_list}
				# Find the most similar token
				next_token = max(similarities, key=similarities.get)
				# Add the most similar token to the ordered list
				ordered_tokens.append(next_token)
				# Remove this token from the tokens list
				tokens_list.remove(next_token)			
			srl_matched.constructs[construct]['tokens_by_similarity'] = ordered_tokens




	
	def to_pandas(self, add_annotation_columns=True,add_metadata_rows = True, order=None, tokens = 'tokens'):
	# def to_pandas(self, add_annotation_columns=True, order=None, tokens = 'tokens'):
		"""
		TODO: still need to test
		lexicon: dictionary with at least
		{'construct 1': {'tokens': list of strings}
		}
		:return: Pandas DF
		"""
		if order:
			warn_missing(self.constructs, order, output_format='pandas / csv')
		lexicon_df = []
		constructs = order.copy() if isinstance(order, list) else self.constructs.keys()
		for construct in constructs:
			df_i = pd.DataFrame(self.constructs[construct][tokens], columns=[construct])
			if add_annotation_columns:
				df_i[construct + "_include"] = [np.nan] * df_i.shape[0]
				df_i[construct + "_add"] = [np.nan] * df_i.shape[0]
			lexicon_df.append(df_i)

		lexicon_df = pd.concat(lexicon_df, axis=1)

		if add_metadata_rows:
			metadata_df_all = []
			if order:
				constructs_in_order = order
			else:
				constructs_in_order = self.constructs.keys()
			for construct in constructs_in_order:
				# add definition, examples, prompt_name as rows below each construct's column
				definition = self.constructs[construct]['definition']
				definition_references = self.constructs[construct]['definition_references']
				examples = self.constructs[construct]['examples']
				prompt_name = self.constructs[construct]['prompt_name']
				metadata_df = pd.DataFrame([prompt_name, definition, definition_references, examples], columns = [construct])
				metadata_df[f'{construct}_include'] = ['']*len(metadata_df)
				metadata_df[f'{construct}_add'] = ['']*len(metadata_df)
				metadata_df_all.append(metadata_df)
		
		
			metadata_df_all = pd.concat(metadata_df_all, axis = 1)
			lexicon_df = pd.concat([metadata_df_all, lexicon_df], axis = 0, ignore_index=True)

			metadata_indeces = ['Prompt name', 'Definition', 'Reference', 'Examples']
			new_index = metadata_indeces + [n-len(metadata_indeces) for n in lexicon_df.index[len(metadata_indeces):].tolist()]
			lexicon_df.index = new_index

		return lexicon_df


	pd.options.display.width = 0
	srl_matched.constructs['Passive suicidal ideation']['tokens']
	srl_matched.constructs['Passive suicidal ideation']['tokens_by_similarity']
	srl_df = to_pandas(srl_matched, order = constructs_in_order, tokens = 'tokens_by_similarity')
	
	srl_df.to_csv(f'./data/input/lexicons/suicide_risk_lexicon_preprocessing/suicide_risk_lexicon_calibrated_matched_tokens_unvalidated_{generate_timestamp()}_annotation.csv')


	# Different order for each annotator
	
	srl_constructs  = 0
	from constructs_annotation_order import constructs_in_order, constructs_kb, constructs_dc,constructs_or 
	
	# # TEST: Make sure no constructs are missing from the ones in construct in order
	# any_missing = constructs_in_order
	# set(constructs_kb) ^ set(constructs_in_order)
	# set(constructs_or) ^ set(constructs_in_order)
	# set(constructs_dc) ^ set(constructs_in_order)
	# [n for n in constructs_dc if n not in constructs_in_order]
	# [n for n in constructs_or if n not in constructs_in_order]
	# [n for n in constructs_kb if n not in constructs_in_order]


	

	
	
	for name, order in {'or':constructs_or, 'kb':constructs_kb, 'dc':constructs_dc}.items():
		srl_df = to_pandas(srl_matched, order = order, tokens = 'tokens_by_similarity')
		srl_df.to_csv(f'./data/input/lexicons/suicide_risk_lexicon_preprocessing/suicide_risk_lexicon_calibrated_matched_tokens_unvalidated_{generate_timestamp()}_annotation_{name}.csv')

	
	srl_matched_with_unmatched = copy.deepcopy(srl_matched)
	# add tokens that were removed at the bottom so coders can see them.
	for construct in constructs_in_order:
		original_tokens = srl.constructs[construct]['tokens']
		new_tokens = srl_matched.constructs[construct]['tokens_by_similarity']
		original_tokens = [n for n in original_tokens if n not in new_tokens]
		new_tokens = new_tokens + ['']*5+['OLD TOKENS: DO NOT CODE']+ original_tokens
		srl_matched_with_unmatched.constructs[construct]['tokens_by_similarity'] = new_tokens

	for name, order in {'or':constructs_or, 'kb':constructs_kb, 'dc':constructs_dc}.items():
		srl_df = to_pandas(srl_matched_with_unmatched, order = order, tokens = 'tokens_by_similarity')
		srl_df.to_csv(f'./data/input/lexicons/suicide_risk_lexicon_preprocessing/suicide_risk_lexicon_calibrated_matched_tokens_with_unmatched_unvalidated_{generate_timestamp()}_annotation_{name}.csv')

	

		











"""
# 'ECT' is relatively uncommon treatment, but very commonly the typo for etc. 
# 'coach' is mor elikely in control group, where it's often about sports and not a mental health treatment. 
# Keep: 'buy' and "price" are much more likely in control group. in suicide they say "buy a gun" or "buy a rope" so gun and rope would come up in other constructs. Here we 
# "race" generally has the sense of racing thoughts rather than demographic (more common in control group)
# "unfair" is rarely associated to discrimination
		
# TODO: should probably remove questioning from Gender & sexual identity
		

# lemmatizer.spacy_lemmatizer(['stoner'])
		


# Do on training set, 2 vs 4. 

# Went through csv and added tokens from the first 20 tokens and examples.






# Calibration

'ed' in srl.constructs['Eating disorders']['tokens']

# 


# matched_tokens_zscores[matched_tokens_zscores['token']=='cut']

# zscores['zscore'].hist(bins=100)
# from matplotlib import pyplot as plt
# plt.xlim((-30,30))

# top_matches_all = []
# counts = {}
# # for top tokens, if they appear often in other docs, then flag token. 
# for construct in constructs_in_order:		

# 	top_matches = list(matches_counter_d[construct].keys())[:20]
# 	top_matches = pd.DataFrame({'token': top_matches, 'construct':construct})

# 	top_matches_all.extend(top_matches)


# 	# REVIEW THSE which were matched in SW 909. Or repeat for matches in CTL. But tokenizer here and is different. 
# 	matched_tokens_zscores = zscores[zscores['token'].isin(top_matches_all_zscore)]

# 	for token in top_matches:
# 		sw = len(get_docs_matching_token(docs, token, window = (n,n)))/len(docs)
# 		nonsw = len(get_docs_matching_token(docs_nonsuicidal, token, window = (n,n)))/len(docs_nonsuicidal)
# 		controls = len(get_docs_matching_token(docs_control, token, window = (n,n)))/len(docs_control)
# 		counts[token] = [construct, sw/(controls+0.000001), sw/(nonsw+0.000001)]

		
# counts_df = pd.DataFrame(counts).T
# counts_df.columns = ['Construct', 'sw/controls', 'sw/nonsw']
# counts_df = counts_df.round(2)
# counts_df = counts_df.sort_values('sw/controls')
# counts_df


# Make sure lemmatized token is not counted if in remove. Add lemmatized unles in remove. Or count unless in remove. 







# Review 20 r/SuicideWatch documents for false negatives
# ============================================================================================================
if run_this:
	n = 20 
	# for each doc, get the matches
	for i, doc in enumerate(docs[:20]):
		print(doc)
		matches = [(k,v) for k,v in matches_per_doc[i].items() if v[0]>0] 
		
		for n in matches:
			print(n)
		print()
	



	'''
	lemmatizer.spacy_lemmatizer(['threatening'])
	'distressed' > "distress", only "distress" is kept, to avoid counting twice. 
	"drained" > "drain", only "drain" is kept unless its in the except_exact_match list"
	"die" and "died" will be kep because they are in the exact match list, because <= exact_match_n
	"catastrophizing" > "catastrophize", both are kept
	'forced to exist' > 'force to exist'  
	"I'm a drag"> "I am a drag", both will be kept, because one is not a substring of the other 
	"grabbed me"> "grab me", both will be kept, because one is not a substring of the other
	'''




srl.save(f'data/lexicons/suicide_risk_lexicon/suicide_risk_lexicon_preprocessing/suicide_risk_lexicon_gpt-4-1106-preview_dml', ['pickle'], order = constructs_in_order, timestamp = True)
# srl_df = srl.to_pandas()
# srl_df.to_csv(f'data/srls/suicide_risk_lexicon_gpt-4-1106-preview_dml_{ts}_annotation.csv')
# srl_df = srl.to_pandas(add_annotation_columns=False)
# srl_df.to_csv(f'data/srls/suicide_risk_lexicon_gpt-4-1106-preview_dml_{ts}.csv')

# TODO: source should be a bit more systematic: word score Reddit, word score CTL, maybe slit important info from other notes with ;










# Count: Extract concurrent (text psychometrics)
# ================================================

srl = load_srl("./data/srls/suicide_risk_lexicon_gpt-4-1106-preview_dml_24-01-31T21-06-52.pickle")

concurrent  = pd.read_csv('./data/ctl/train10_train_concurrent_metadata_messages_preprocessed_23-07-20T02-00-58.csv', index_col = 0)
concurrent.columns
exact_match_tokens = ['anger', 'stone', 'panic', 'break', 'hit I', 'trans', 'owner', 'my ex', 'charge', 'hit me', 'cholo', 'weight', 'aching', 'needle', 'resent', 'ponder', 'pen in', 'her pass', 'his pass']
docs = concurrent['message'].tolist()

feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = extract(docs,
																					  srl.constructs,normalize = False, return_matches=True,
																					  add_lemmatized_lexicon=True, lemmatize_docs=False,
																					  exact_match_n = 4,exact_match_tokens = exact_match_tokens)

assert concurrent.shape[0] == feature_vectors.shape[0]
df_srl = concurrent.copy()
df_srl[feature_vectors.columns] =feature_vectors.values
df_srl

df_srl.to_csv(f'./data/ctl/train10_train_concurrent_metadata_messages_preprocessed_23-07-20T02-00-58_suicide-risk-lexicon.csv', index = False)
df_srl.to_csv(f'./../../construct_tracker/data/input/features/train10_train_concurrent_metadata_messages_preprocessed_23-07-20T02-00-58_suicide-risk-lexicon.csv', index = False)

for k, v, in matches_counter_d.items():
	print(k, v)
	print()


i = 5000
docs[i]
matches_per_doc[i]
df_srl.iloc[i,:]['Shame, self-disgust, & worthlessness']



# Extract train10_subset (srl paper) 
# ============================================================================================================

import pickle
# Load training set balanced across risk levels
with open('./tutorials/ctl_dfs.pkl', 'rb') as f:
	dfs = pickle.load(f)
	

# # from concept_tracker.srl import extract
# # for split in dfs.keys():
# for split in dfs.keys():
#
# 	name = dfs[split]['name']
# 	df = dfs[split]['messages']
# 	docs =df['message'].tolist()
# 	if type(docs[0])==np.str_:
# 		docs = [n.item() for n in docs]
# 	if toy:
# 		docs = docs[:10]
# 		srl_constructs_toy = {}
# 		for c in ['Loneliness & isolation', 'Active suicidal ideation & suicidal planning']:
# 			srl_constructs_toy[c] = srl.constructs[c].copy()
#
#
# 		feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = extract(docs,
# 		                                                                                      srl_constructs_toy,normalize = False, return_matches=True,
# 		                                                                                      add_lemmatized_lexicon=True, lemmatize_docs=False,
# 		                                                                                      exact_match_n = 4,exact_match_tokens = exact_match_tokens)
#
# 	else:
# 		feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = extract(docs,
# 	                                                                                      srl.constructs,normalize = False, return_matches=True,
# 	                                                                                      add_lemmatized_lexicon=True, lemmatize_docs=False,
# 	                                                                                      exact_match_n = 4,exact_match_tokens = exact_match_tokens)
#
# 		df_srl = pd.concat([df, feature_vectors], axis=1)
# 		df_srl.to_csv(f'./data/ctl/{name}_suicide-risk-lexicon.csv')
# 	# for i, doc in enumerate(docs):
# 	#     print(doc)
# 	# matches = [(k,v) for k,v in matches_per_doc[i].items() if v[0]>0]
# 	# for n in matches:
# 	#     print(n)
# 	# print()
#




# =====================================================================================

# counting / extraction issue: TODO: should just add all tokens with " me ", endswith(' me') add the same but with "my", because lemmatization won't help.
# TODO: figure out how to set instance variables as class variables. Or don't write code that uses self.class_variable if it doesn't exist in saved srl. Or Load old srl into new srl template.
# TODO make sure exact_match_tokens works when it is a list.
# TODO add the timelogger for forloops within slow parts of extract.




# TODO If I want to capture "killing myself" I have it in the srl, but if it's lemmatized it will be "kill myself",
#  so I need to combine lists of lemmatized and unlemmatized. Remove repeated tokens that contain other tokens
#  so they're not counted twice for the same match


# TODO: is this parsed as an apostrophe? "can’t" do this anymore
# lemmatizer.spacy_lemmatizer(['feel like dying'])

# Add:
# TODO: did not capture "my ex" which is in the exact match list
# Todo create this from a function
# TODO remove: "sible", 'patron', 'penned in', 'pen in','overdo','',



# Remove punctuation from doc: "self-harm" did not match "self-harming"
# TODO See which tokens never come up.
# TODO See which docs don't have any matches.



# for k,v in add_d:
# 	srl.add(k, section='tokens', value=v, source = 'DML manually added after positive and negative matches calibration')


# TODO consider removing many "I " or "I'm " like in "I want to die" people say "want to die".
# TODO try lemmatizing docs as well
# TODO: side by side comparison of lemmatized words that are different and the construct
# TODO Make a tutorial like the TextBlob post



# TODO remove a lot of "I from the beginning of phrases like "I want to die".

# TODO: search for a word and see where it came up (you can do this in the csv).


# TODO: lemmatize option

# TODO add method for lemmatizing all constructs srl_new.lemmatize()
# TODO place lemmatizer in lemmatizer.py





# litellm.set_verbose=False
"""