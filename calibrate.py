
import os
import sys
import pandas as pd
import numpy as np

sys.path.append('/Users/danielmlow/Dropbox (MIT)/datum/concept-tracker/') # TODO remove

from concept_tracker import lexicon
from concept_tracker.lexicon import * # TODO remove
from concept_tracker.utils import lemmatizer

input_dir = "./data/input/lexicons/suicide_risk_lexicon_preprocessing/"



def gen_add_remove_dict(constructs):

	add_remove_dict = {}
	for construct in constructs:
		add_remove_dict[construct] = {'add':[], 'remove':[]}
	return add_remove_dict

constructs_in_order = [
	"Passive suicidal ideation",
	"Active suicidal ideation & suicidal planning",
	"Lethal means for suicide",
	"Direct self-injury",
	"Suicide exposure",
	"Other suicidal language",
	"Loneliness & isolation",
	"Social withdrawal",
	"Relationship issues",
	"Relationships & kinship",
	"Bullying",
	"Sexual abuse & harassment",
	"Physical abuse & violence",
	"Aggression & irritability",
	"Alcohol use",
	"Other substance use",
	"Impulsivity",
	"Defeat & feeling like a failure",
	"Burdensomeness",
	"Shame, self-disgust, & worthlessness",
	"Guilt",
	"Anxiety",
	"Panic",
	"Trauma & PTSD",
	"Agitation",
	"Rumination",
	"Depressed mood",
	"Anhedonia & uninterested",
	"Emotional pain & psychache",
	"Grief & bereavement",
	"Existential meaninglessness & purposelessness",
	"Emptiness",
	"Hopelessness",
	"Entrapment & desire to escape",
	"Perfectionism",
	"Fatigue & tired",
	"Sleep issues",
	"Psychosis & schizophrenia",
	"Bipolar Disorder",
	"Borderline Personality Disorder",
	"Eating disorders",
	"Physical health issues & disability",
	"Incarceration",
	"Poverty & homelessness",
	"Gender & sexual identity",
	"Discrimination",
	"Finances & work stress",
	"Barriers to treatment",
	"Mental health treatment",

]

# Calibrate looking at reddit dataset
# =======================================================

# srl2 = Lexicon()
# TODO: load new lexicon add existing info to it if it exists in the old lexicon. for key in newlexicon, 
srl = lexicon.load_lexicon(input_dir+"suicide_risk_lexicon_gpt-4-1106-preview_dml_24-01-22T03-25-06.pickle")
srl.attributes = {} # TODO: it would already exist, so you can remove from tutorial
srl.attributes['remove_from_all_constructs'] = [0, '0', "I", "I'm"] # TODO replace with this: srl.set_attribute('remove_from_all_constructs', [0, '0', "I", "I'm"]) # srl.get_attribute('remove_from_all_constructs')
# srl.get_attribute('remove_from_all_constructs')
srl.exact_match_n = 4
srl.exact_match_tokens = None



# lemmatize srl tokens
# =========================

c= 'Passive suicidal ideation'

from concept_tracker.utils import lemmatizer
for c in tqdm(srl.constructs.keys()):
	srl_tokens = srl.constructs[c]['tokens'].copy()


	# If you add lemmatized and nonlemmatized you'll get double count in many cases ("plans" in doc will be matched by "plan" and "plans" in srl)
	srl_tokens_lemmatized = lemmatizer.spacy_lemmatizer(srl_tokens, language='en') # custom function
	srl_tokens_lemmatized = [' '.join(n) for n in srl_tokens_lemmatized]
	srl_tokens += srl_tokens_lemmatized
	srl_tokens = [n for n in srl_tokens if n not in srl.constructs[c]['remove']]
	srl_tokens = list(np.unique(srl_tokens)) # unique set
	
	
	srl.constructs[c]['tokens_lemmatized']=srl_tokens


# # setattr(srl,'remove_from_all_constructs', ['0'])
reddit = pd.read_csv('./data/input/reddit/first_sw_submission_2021-07-20-23-12-27.csv',sep=',',engine='python' , index_col = 0)
reddit.columns

# # Extract
text_col = 'text'

# Clean
reddit.shape
reddit.subreddit.value_counts()
reddit = reddit[~reddit[text_col].isna()]
reddit.shape
reddit.subreddit.value_counts()
docs_sw = reddit.sample(n = 10000, random_state=42)[text_col].values

# TODO: add doc to table?



# Extract
run_this = False

if run_this:
	feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = lexicon.extract(docs,
																					  srl.constructs,normalize = False, return_matches=True,
																					  add_lemmatized_lexicon=True, lemmatize_docs=False,
																					  exact_match_n = srl.exact_match_n,exact_match_tokens = srl.exact_match_tokens)



top_matches_all = []
# for top tokens, if they appear often in other docs, then flag token. 
for construct in constructs_in_order:		
	
	top_matches = [k for k,v in matches_counter_d[construct].items() if v > 5]
	print(len(top_matches))
	# top_matches = list(matches_counter_d[construct].keys())[:25]
	top_matches = pd.DataFrame({'token': top_matches, 'construct':construct})

	top_matches_all.append(top_matches)

top_matches_all = pd.concat(top_matches_all)


# matches_counter_d2 = matches_counter_d.copy()
# matches_counter_d = {}
# for c in matches_counter_d2.keys():
# 	matches_counter_d[c] = 	{k: v for k, v in sorted(matches_counter_d2[c].items(), key=lambda item: item[1], reverse=True)}



# View most frequent matches per construct for false positives
# ============================================================================================================


# obtain m words before and after token in docs_match


def get_context(doc, token, n_words_pre = 10,n_words_post = 10):
	doc_pre_token = ' '.join(doc.split(token)[0].split(' ')[-n_words_pre:])
	doc_post_token = ' '.join(doc.split(token)[1].split(' ')[:n_words_post])
	doc_windowed = doc_pre_token + token + doc_post_token
	return doc_windowed


get_docs_matching_token(['get paranoid and I think this is also a'],'thin', window = (10,10), exact_match_n = 4)

def get_docs_matching_token(docs, token, window = (10,10), exact_match_n = 4):
	
	docs_matching_token = [n for n in docs if token in n]
	if len(token)<=exact_match_n:
		#exact match
		docs_matching_token2 = docs_matching_token.copy()
		docs_matching_token = []
		for doc in docs_matching_token2:
			words = doc.split(' ')
			if token in words:
				docs_matching_token.append(doc)
				
	if window:
		docs_matching_token_windowed = []
		for doc in docs_matching_token:
			# doc = docs_matching_token[1]
			doc_windowed = get_context(doc, token, n_words_pre = window[0], n_words_post = window[1])
			docs_matching_token_windowed.append(doc_windowed)
		return docs_matching_token_windowed
		
	else:
		return docs_matching_token

# TODO: dont include acronyms that are words in the lexicon
	

fv, _, _, _ = lexicon.extract(['i ACT wierd sometimes'], srl.constructs)
fv['Mental health treatment']
rmhd = pd.read_csv('./data/input/reddit/rmhd_27subreddits_1300posts_train.csv', index_col = 0)

mental_health_subreddits = rmhd.subreddit.unique().tolist()
control_group = [ 'ukpolitics', 'teaching',
	   'personalfinance', 'mindfulness', 
	   'legaladvice', 'guns', 'conspiracy', 'UKPersonalFinance', 'unitedkingdom']
mental_health_subreddits = [n for n in mental_health_subreddits if n not in control_group]
non_suicidal = ['suicidewatch', 'depression']

# with suicidal and depression
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

# feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = lexicon.extract(['cute boy'], srl.constructs, normalize = False, add_lemmatized_lexicon=True, lemmatize_docs=False)
# feature_vectors['Active suicidal ideation & suicidal planning']

# n = 5
# get_docs_matching_token(docs, token, window = (n,n))
# print(len(get_docs_matching_token(docs, token, window = (n,n))))

# get_docs_matching_token(docs_control, token, window = (n,n))
# print(len(get_docs_matching_token(docs_control, token, window = (n,n))))

# get_docs_matching_token(docs_nonsuicidal, token, window = (n,n))
# print(len(get_docs_matching_token(docs_nonsuicidal, token, window = (n,n))))


# top_matches_all = []
# # for top tokens, if they appear often in other docs, then flag token. 
# for construct in constructs_in_order:		
	
# 	top_matches = [k for k,v in matches_counter_d[construct].items() if v > 5]
# 	print(len(top_matches))
# 	# top_matches = list(matches_counter_d[construct].keys())[:25]
# 	top_matches = pd.DataFrame({'token': top_matches, 'construct':construct})

# 	top_matches_all.append(top_matches)

	
# top_matches_all = pd.concat(top_matches_all, axis=0)
# from concept_tracker import word_scores



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
# annotation_names = zscores.annotation_names.values


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


# word score on Reddit data
# ======================================================================
mental_health_subreddits
rmhd_mh_all = rmhd[rmhd.subreddit.isin(mental_health_subreddits)]

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
	
	
	
	word_scores_per_subreddit[subreddit] = zscores[zscores['zscore']>3][::-1]['token'].tolist()
	
# Could 
	
for subreddit in rmhd_mh_all.subreddit.unique():
	print()
	print(f'{subreddit}')
	tokens = word_scores_per_subreddit[subreddit]
	for token in tokens:
		print()
		print(f'{token}')
		matched_docs = get_docs_matching_token(train_df['text'].values, token, window = (n,n))
		print(matched_docs[:5])
		
	break
srl.constructs.keys()
'want to be skinny' in srl.constructs['Eating disorders']['tokens']



	
	





# zscores on CTL data
# ========

import pickle
with open('./data/input/ctl/ctl_dfs_features.pkl', 'rb') as f:
	dfs = pickle.load(f)

train_df = dfs['train']['df_text'][['text', 'y']]

l1_docs = train_df[train_df['y']==4]['text'].values
l1_name = 'high'
l2_docs = train_df[train_df['y']==2]['text'].values
l2_name = 'low'

l3_docs = train_df[train_df['y']==3]['text'].values

cv = CV(decode_error = 'ignore', min_df = 1, max_df = 0.99, ngram_range=(1,3),
				binary = False,
		stop_words =None,
	  max_features = 15000)
zscores = word_scores.bayes_compare_language(l1_docs, l2_docs, ngram = 1, prior=.01, cv = cv, threshold_zscore=threshold_zscore, l1_name = l1_name, l2_name = l2_name )
zscores

# These tokens are particularly useful


zscores.iloc[-50:]

'razor' in srl.constructs['Lethal means for suicide']
'hospital' in srl.constructs['Lethal means for suicide']


# calibrate on CTL data
# ================================================================

srl.add('Hospitalization', section = 'tokens', value = ['hospital', 'hospitalized','hospitalization','hospitalize', 'ER', 'emergency room', 'psych unit', 'psych ward', 'psychiatric unit', 
												 'in and out of the clinic','in and out of the hospital', 'inpatient', 'in patient', 'clinic', 'urgent care'],
		source = ['DML adding'])



# Extract on l1_docs, l2_docs, l3_docs
feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = lexicon.extract(train_df['text'].tolist(),
																					  srl.constructs,normalize = False, return_matches=True,
																					  add_lemmatized_lexicon=True, lemmatize_docs=False,
																					  exact_match_n = srl.exact_match_n,exact_match_tokens = srl.exact_match_tokens)

matches_counter_d['Hospitalization']


pd.concat([train_df, feature_vectors],axis=1)
train_df = pd.concat([train_df, feature_vectors],axis=1)



# false negatives on 20-40 documents






# 'ECT' is relatively uncommon treatment, but very commonly the typo for etc. 
# 'coach' is mor elikely in control group, where it's often about sports and not a mental health treatment. 
# Keep: 'buy' and "price" are much more likely in control group. in suicide they say "buy a gun" or "buy a rope" so gun and rope would come up in other constructs. Here we 
# "race" generally has the sense of racing thoughts rather than demographic (more common in control group)
# "unfair" is rarely associated to discrimination
		
# TODO: should probably remove questioning from Gender & sexual identity
		

# lemmatizer.spacy_lemmatizer(['stoner'])
		


# Do on training set, 2 vs 4. 




# gen_add_remove_dict(constructs_alphabetical)
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
 'Guilt': {'add': ['in the wrong', 'am wrong',  'I am sorry', 'am to blame',
 'apologize', 'blame myself', "can't forgive myself", "didn't mean to",
 "didn't think it through",'feel bad',
 'feel terrible','I fucked up','have a conscience','made a mistake','messed up','owe you an apology',
 'should have known better',
 "I shouldn't have",
 "shouldn't have done that",
 "shouldn't have said that",
 'take responsibility',
 'take the blame',
 'was wrong',
 'wish I could take it back',
 "wish I hadn't",], 'remove': ['wrong']},
 'Hopelessness': {'add': ['doom', 'crushing', 'crushed'], 'remove': ['crush', 'do for']},
 'Impulsivity': {'add': ['is wild'], 'remove': ['wild', 'irrational']},
 'Incarceration': {'add': ['jail sentence', 'in custody'], 'remove': ['custody']},
 'Lethal means for suicide': {'add': ['jumping off', 'burning my', 'burning charcoal', 'suicide bag'], 'remove': ['chemical', 'burn', 'jumping', 'fall', 'toxic', 'burning']},
 'Loneliness & isolation': {'add': ['unwanted', 'am single', 'single mom', 'single parent', 'single dad', 'cutt off all ties', 'cut off from'], 'remove': ['single', 'cut off']},
 'Mental health treat ment': {'add': ['pump my stomach', 'stomach pumping'], 'remove': ['ECT', 'ER', 'ACT', 'CAM']},
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


# Remove some first-person expression for constructs that are not self directed (e.g, 'self disgust', "self injury") such as "Burdensomeness" or "Guilt"
srl.constructs['Burdensomeness']['tokens']
# These will be captured by lemmatization
srl.add('Burdensomeness', source = 'DML removing without first person so they are captured by lemmatization more broadly', section = 'tokens', value = [
	"am a bother",
 "am a burden",
 "am a chore",
 "am a disappointment",
 "am a drag",
 "am a hassle",
 "am a hindrance",
 "am a load",
 "am a nuisance",
 "am a problem",
 "am a weight",
 "am a weight on others",
 "am burdening",
 "am demanding",
 "am draining",
 "am exhausting",
 "am hard to love",
 "am high-maintenance",
 "am holding them back",
 "am in the way",
 "am such a burden",
 "am too much trouble",
 "am too needy",
 "be wasting space",

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
	'am sorry',
 'am to blame',
 'apologize',
 'blame myself',
 'forgive myself',
 'do not mean to',
 'not think it through',
 'feel bad',
 'feel terrible',
 'I fuck up',
 'have a conscience',
 'make a mistake',
 'mess up',
 'owe you an apology',
 'should have know well',
 'should not have',
 'should not have do that',
 'should not have say that',
 'take responsibility',
 'take the blame',
 'am wrong',
 'wish I could take it back',
 'am sorry',
 'am to blame',
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
 'can not be fixed',
 'can not be helped',
 'do not have enough money for therapy',
 'do not have time for therapy',
 'do not need therapy',
 'do not need treatment',
 'do not trust psychiatrist',
 'do not trust psychologist',
 'have not be reffered yet',
 'talk to friends instead',
 'would not look for help',
 "not treated well because I'm old",
 'scared of how people will react if they find out',
 "discriminated by doctor",

])






# These tokens should be found as tokens not within words because they'll have many false positives
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

matched_tokens_zscores[matched_tokens_zscores['token']=='cut']

zscores['zscore'].hist(bins=100)
from matplotlib import pyplot as plt
plt.xlim((-30,30))

top_matches_all = []
counts = {}
# for top tokens, if they appear often in other docs, then flag token. 
for construct in constructs_in_order:		

	top_matches = list(matches_counter_d[construct].keys())[:20]
	top_matches = pd.DataFrame({'token': top_matches, 'construct':construct})

	top_matches_all.extend(top_matches)


	# REVIEW THSE which were matched in SW 909. Or repeat for matches in CTL. But tokenizer here and is different. 
	matched_tokens_zscores = zscores[zscores['token'].isin(top_matches_all_zscore)]

	for token in top_matches:
		sw = len(get_docs_matching_token(docs, token, window = (n,n)))/len(docs)
		nonsw = len(get_docs_matching_token(docs_nonsuicidal, token, window = (n,n)))/len(docs_nonsuicidal)
		controls = len(get_docs_matching_token(docs_control, token, window = (n,n)))/len(docs_control)
		counts[token] = [construct, sw/(controls+0.000001), sw/(nonsw+0.000001)]

		
counts_df = pd.DataFrame(counts).T
counts_df.columns = ['Construct', 'sw/controls', 'sw/nonsw']
counts_df = counts_df.round(2)
counts_df = counts_df.sort_values('sw/controls')
counts_df


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
	


srl.exact_match_tokens = exact_match_tokens

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

# TODO: change this to output the format of the dictionary systematically, then this might change cause we'll always have both entries.
for construct, add_and_remove_i in add_and_remove_d.items():

	add_tokens_i = add_and_remove_i[0]
	if add_tokens_i != []:
		srl.add(construct, source = 'DML added after calibration', section = 'tokens', value = add_tokens_i)
	if len(add_and_remove_i)>1:



		remove_tokens_i = add_and_remove_i[1]
		if remove_tokens_i != []:
			if construct == 'Suicide exposure':
				print(remove_tokens_i)
				break

			srl.remove(construct, source = 'DML removed after calibration', remove_tokens = remove_tokens_i)


# Tests
assert ('family' in srl.constructs['Relationships & kinship']['tokens']) == True
assert ('borderline' in srl.constructs['Borderline Personality Disorder']['tokens']) == True
assert ('drowning' in srl.constructs['Passive suicidal ideation']['tokens']) == False
assert ('cut' in srl.constructs['Borderline Personality Disorder']['tokens']) == False
assert ('apathy' in srl.constructs['Passive suicidal ideation']['tokens']) == False
srl.constructs['Relationship issues']['remove']




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










"""

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





