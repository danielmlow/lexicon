import datetime
import re
import warnings
from collections import Counter
import time
import dill
import numpy as np
import pandas as pd
from utils.word_count import word_count  # local
import pickle
import json


# Look for code where I obtain window for tokens in a dataset (in word scores or create_lexicon ipynb)


# Extract
# ========================================================================

#
# import pandas as pd
# import re
# from collections import Counter
# from .utils.count_words import word_count
# # from text.utils.count_words import word_count
# import numpy as np


# return_matches = True
# normalize = False
# features, matches = lexicons.extract(docs,lexicons_d, normalize = normalize, return_matches=return_matches)

# Check for false positives
# =======================================
"""
run_this = True
code = False #if False, just list matches, if True, input whether to keep or remove token in lexicon

if run_this:
	remove_tokens = {}
	for construct in suicide_lexicon_constructs:
		print()
		print('='*30)
		print(construct)
		remove_tokens[construct] = []
		matches_i = matches.get(construct)
		if matches_i!=None:
			matches_i = list(matches_i.items())
		else:
			continue

		# print docs
		for match in matches_i:
			docs_from_sr = messages[messages['message_from']=='Selena Rodriguez']['event'].values # TODO: remove name
			matched_messages = [n for n in docs_from_sr if str(n).lower().count(match[0])>0]
			print(match, 'from SR and other users')
			[print(n) for n in matched_messages]
			print()
			if code:
				remove = input('remove token from construct lexicon? 0=no, 1=yes')
				if remove == '1':
					remove_tokens[construct].append(match[0])
else:
	remove_tokens = {
		'suicidality_passive': ['kill me'],
		'suicidality_general': ['death','dead', 'kill me', 'depress'],
		'suicidality_active': ['hanging'],
		'suicidality_selfinjury': ['stab']
	}

for construct in remove_tokens:
	tokens_i = lexicons_d.get(construct)
	to_remove =  remove_tokens.get(construct)
	for token in to_remove:
		if token in tokens_i:
			tokens_i.remove(token)
	lexicons_d[construct] = tokens_i





lexicon
- name
- constructs
- description


lexicon.add(construct,

lexicon.constructs['loneliness'] = {
- prompt_name
- definition
- definition_references
- seed_examples
- tokens # add all tokens from tokens_metadata and remove tokes from tokens_removed
- tokens_metadata = {
	f'gpt-4, temperature-0, seed-42, {ts}' : {'prompt': prompt, 'tokens':[]}

- tokens_removed = [] # using substrings, token A contains token B, after human coding
}


}

"""





"""
# Actual code used:
# =========

ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S')

lexicon = dill.load(open('data/lexicons/suicide_risk_lexicon_gpt-4-1106-preview_24-01-14T19-41-30.pickle', "rb"))

# Choose which to add
word_score_dir = './data/output/word_scores/'


ctl_map = {'self_harm': 'Self-injury',
		   'bereavement': 'Grief & bereavement',
		   'anxiety_stress': 'Anxiety',
		   'relationship': 'Relationships & relationship issues',
		   'isolated': 'Loneliness & isolation',
		   'abuse_physical': 'Physical abuse & violence',
		   'gender_sexual_identity': 'Gender & sexual identity',
		   'suicide': '', #TBD
		   'abuse_sexual': 'Sexual abuse & harassment',
		   'bully': 'Bullying',
		   'eating_body_image': 'Eating disorders',
		   'substance': ''} # Other substance use

do_manually = {} # 'belongs to multiple constructs. Do manually'
for variable_name in ctl_map.keys():

	construct = ctl_map.get(variable_name)
	print()
	print('=========', construct, variable_name)
	df_i = pd.read_csv(word_score_dir + f'ctl_{variable_name}-examples-words_5-24-01-01T21-15-04_coded.csv', index_col = 0, encoding = 'latin1')
	dataset_tokens = df_i[df_i['keep']==1]['term'].tolist()
	# print(set(dataset_tokens))
	if construct != '':
		tokens = lexicon.constructs[construct]['tokens']
		print('\ntokens from dataset not already in lexicon:\n', set(dataset_tokens)-set(tokens))
		lexicon.add(construct, section = 'tokens', value = dataset_tokens, source = f'CTL word scores one vs 12 other types of crises. Dataset: train10_train_concurrent_metadata_messages_preprocessed_23-07-20T02-00-58.csv. Added by DML.')
	else:
		do_manually[variable_name] = dataset_tokens

print('Belongs to multiple constructs. Do manually')
for k, v in do_manually.items():
	print(k, v)


lexicon.constructs.keys()

lexicon.add('Other suicidal')




import warnings


add_manually_from_dataset = {
	'Active suicidal ideation':['overdose', 'overdosing', 'car off a bridge', 'jump off a bridge', 'jumping off a bridge', 'take a lot of pills', 'want to end it', 'knife to my throat', 'shoot myself', 'point a gun at my head', 'killing myself', 'killing me', 'electrocuting myself', 'drive into something', 'slitting my wrists', 'cutting my wrists', 'hanging myself', 'tried OD', 'tried to OD', 'attempt suicide', 'suicide attempt', 'attempt', 'take a bottle of pills', 'taking all the pills', 'shoot myself', 'stabbing myself', 'stabbing my self', 'stab my self', 'painless way out', 'quick and painless', 'easy and painless', 'jumping off a bridge', 'commit suicide'],
	'Passive suicidal ideation': ['feel like dying', 'why am I still alive', "don't want to be alive", "don't care if I'm alive anymore", 'struggling with wanting to be alive', 'no reason to keep myself alive', 'sleep forever', "don't want to go on", 'thinking about death'],
	'Lethal means for suicide':['overdose', 'overdosing', 'car off a bridge', 'jump off a bridge', 'jumping off a bridge', 'take a lot of pills', 'knife to my throat', 'shoot myself', 'point a gun at my head', 'electrocuting myself', 'drive into something', 'slitting my wrists', 'cutting my wrists', 'hanging myself', 'tried OD', 'tried to OD', 'take a bottle of pills', 'taking all the pills', 'shoot myself', 'stabbing myself', 'stabbing my self', 'stab my self', 'jumping off a bridge'],
	'Other suicidal language': ["I'm dying", 'I will be dying', 'ending my life', 'ending it', 'safety plan', 'suicide', 'severe depression','the end of my rope', 'no reason to live'],
	'Defeat & feeling like a failure': ['the end of my rope'],
	'Burdensomeness': ["won't burden", "I'm a burden"],
	"Shame, self-disgust, & worthlessness": ["I don't deserve to be here"],
	"Hopelessness": ['no hope'],
	'Self-injury': ['knife to my throat', 'shoot myself', 'electrocuting myself', 'slitting my wrists', 'cutting my wrists', 'stabbing myself', 'stabbing my self', 'stab my self'],
	'Alcohol use': ['withdrawal', 'cravings', 'sober', 'booze', 'addiction', 'addicted', 'rehab', 'alcohol', 'drinker', 'hungover', 'blackout', 'detox', 'whiskey', 'liquor', "DUI", "driving under the influence", 'AA meeting', 'alcoholic', 'Alcoholics Anonymous'],
	'Other substance use': ['withdrawal', 'juul', 'vape', 'vaping', 'addiction', 'acid','lsd', 'rehab', 'suboxone', 'opiate', 'drug', 'methadone', 'oxy', 'morphine', 'rehab','detoxing', 'THC', 'huffing', 'percocet', 'laced', 'weed', 'snort', 'pot', 'cannabis', 'detox', 'gabapentin', 'adderal', 'IOP','intensive outpatient', 'partied', 'medicate',  'synthetic drug', 'meds', 'falling back into old habits', 'spun out', 'spun up', 'NA meeting', 'pain pill', 'narcotic', 'Narcotics Anonymous'],
}




for construct, dataset_tokens in add_manually_from_dataset.items():
	if construct not in lexicon.constructs.keys():
		warnings.warn(f"'{construct}' does not exist in lexicon. Creating new construct for it. This warning exists so that if there is a typo in the construct name, you'll know it'll get assigned to a new construct")
	print(construct, dataset_tokens)
	lexicon.add(construct, section = 'tokens', value = dataset_tokens, source = f'CTL word scores one vs 12 other types of crises. Dataset: train10_train_concurrent_metadata_messages_preprocessed_23-07-20T02-00-58.csv. Added by DML.')

lexicon.save('data/lexicons/suicide_risk_lexicon_gpt-4-1106-preview_24-01-14T19-41-30')







constructs = [
	"Passive suicidal ideation",
	"Active suicidal ideation & suicidal planning",
	"Lethal means for suicide",
	"Other suicidal language",
	"Direct self-injury",
	"Suicide exposure",
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
len(constructs)

lexicon = load_lexicon("data/lexicons/suicide_risk_lexicon_gpt-4-1106-preview_24-01-14T19-41-30.pickle")

for construct in lexicon.constructs:
	lexicon.constructs[construct]["remove"] = []
	variable_name = generate_variable_name(construct)
	lexicon.constructs[construct]["variable_name"] = variable_name

	lexicon.constructs[construct]["override_remove"] = []
	for source in lexicon.constructs[construct]["tokens_metadata"]:
		lexicon.constructs[construct]["tokens_metadata"][source]["add_or_remove"] = "add"

# Add examples to lexicon (should be done at the beginning.
# ===
# import litellm
# litellm.drop_params=True
# # load api keys
# import os
# import api_keys
# os.environ["OPENAI_API_KEY"] = api_keys.open_ai  # string, you need your own key and to put at least $5 in the account
# os.environ["COHERE_API_KEY"] = api_keys.cohere_trial  # string, you can get for free by logging into cohere and going to sandbox
# # # pip install -q google-generativeai
# # os.environ["GEMINI_API_KEY"] = api_keys.gemini  # here: https://makersuite.google.com/app/apikey
# # os.environ["VERTEXAI_PROJECT"] = api_keys.vertex

#  Add Other suicidal language metadata
lexicon.constructs['Other suicidal language']['tokens_metadata'].keys()
lexicon.add("Other suicidal language", section="variable_name", value="other_suicidal_language")
lexicon.add(
	"Other suicidal language",
	section="definition",
	value="Suicidal language that does not belong in the other suicidal categories (active and passive SI, self-injury, and lethal means for suicide)"
	)
lexicon.add("Other suicidal language", section="definition_references", value="Ours")

lexicon.add("Other suicidal language", section="examples", value="no reason to live; safety plan; I hate my life")

print(lexicon.constructs["Other suicidal language"]['tokens_metadata'].keys())



# Add examples
for construct in lexicon.constructs:
	examples = lexicon.constructs[construct]["examples"]
	examples = examples.split("; ")
	lexicon.add(construct, section="tokens", value=examples, source="examples by DML")



# TODO remove print when adding

# Manual / qualitative / coding / annotation: remove clearly unrelated and add other tokens from different sources
# ================================================================================================

# TODO: lexicon_df to lexicon object

run_this = False
if run_this:
	lexicon_df = lexicon.to_pandas(
		add_annotation_columns=True, order=constructs
	)  # TODO: add definitions, etc that I added below.

	lexicon_df.to_csv(f"data/lexicons/suicide_risk_lexicon_annotate_{ts}.csv")

# Load
lexicon_df = pd.read_csv(
	"data/lexicons/suicide_risk_lexicon_annotate_24-01-15T19-08-09_dml.csv", encoding="latin1", index_col=0
)
lexicon_df = lexicon_df.reset_index(drop=True)

constructs = [n for n in lexicon_df.columns if "_" not in n]
for construct in constructs:
	# print(construct)
	add_i = lexicon_df[~lexicon_df[construct + "_add"].isna()][construct + "_add"].tolist()
	lexicon.add(construct, section="tokens", value=add_i, source="DML cleaning up tokens from GPT-4 and CTL dataset")
	remove_i = lexicon_df[lexicon_df[construct + "_include"] == 0][construct].tolist()
	lexicon.remove(
		construct, remove_tokens=remove_i, source="DML removing clearly unrelated tokens from GPT-4 and CTL dataset"
	)


# TODO: accept arrays as inputs to functions, then I can turn into lists if needed

# =================================================================

# Rename, change definition
lexicon.constructs["Active suicidal ideation & suicidal planning"] = lexicon.constructs[
	"Active suicidal ideation"
].copy()
lexicon.constructs["Active suicidal ideation & suicidal planning"]["variable_name"] = generate_variable_name(
	"Active suicidal ideation & suicidal planning"
)
del lexicon.constructs["Active suicidal ideation"]

# Rename, change definition, redo GenAI search
lexicon.constructs["Direct self-injury"] = lexicon.constructs["Self-injury"].copy()
del lexicon.constructs["Self-injury"]
lexicon.constructs["Direct self-injury"]["examples"] = "self-injury; cut myself; cutting; burn myself; harm myself"
lexicon.constructs["Direct self-injury"]["definition"] = (
	"Mention of actual or desired direct and deliberate destruction of an individual's own body tissue with suicidal"
	" intent (suicidal self-injury) our without (non-suicidal self-injury)"
)
lexicon.constructs["Direct self-injury"]["variable_name"] = generate_variable_name("Direct self-injury")

# Rename, change definition, redo GenAI search
lexicon.constructs["Relationship issues"] = lexicon.constructs["Relationships & relationship issues"].copy()
del lexicon.constructs["Relationships & relationship issues"]
lexicon.constructs["Relationship issues"][
	"definition"
] = "Mention of potentially negative relationship types (widow, my ex) and processes (divorce, ghosting)"
lexicon.constructs["Relationship issues"]["variable_name"] = generate_variable_name("Relationship issues")
lexicon.constructs["Relationship issues"][
	"examples"
] = "widow; ghosted; cheat; divorce; toxic relatioship; breakup; infidelity"
lexicon.constructs["Relationship issues"]["definition"]

construct = 'Existential meaninglessness & purposelessness'
tokens = ['I have no meaning', 'I have no purpose', 'existence has no meaning', 'existence has no purpose', "it's pointless", "life doesn't make sense", "life doesn't seem worthwhile", 'life has no clear aims', 'life has no clear goals', 'life has no meaning', 'life has no sense of direction', 'life has no sense of meaning', 'life has no sense of purpose', 'life is going nowhere', 'life is insignificant', 'life is meaningless', 'life is pointless', 'life is shit', 'meaningless', 'no meaning', 'no meaning to my life', 'no point', 'no purpose', "no purpose in what I'm' doing", 'no reason to live', 'nothing I want to achieve', 'nothing makes sense', 'pointless', 'purposeless', 'shitty life', "there's no point to life"]
lexicon.add(construct, section = 'tokens',source = 'DML adding manually', value = tokens)
lexicon.add(construct, section = 'prompt_name', value = "lack of meaning or lack of purpose about one's life")
lexicon.add(construct, section = 'variable_name', value = generate_variable_name(construct))
lexicon.add(construct, section = 'definition', value = "One's life lacks meaning, does not make sense, does not have purpose, or is insignificant.")
lexicon.add(construct, section = 'definition_references', value = 'Li et al. (2022). Existential meaninglessness scale: scale development and psychometric properties. Journal of Humanistic Psycholog.')
lexicon.add(construct, section = 'examples', value = examples)

lexicon.constructs[construct]['tokens_metadata'].keys()


# TODO: add manual, from Osiris search, also maybe Reddit
# ================================================================================================

annotation_or = pd.read_csv(
	"./data/lexicons/suicide_risk_preprocessing/OsirisRankinFirstPassForDanLowMarch_3_2023_daniel_added_prototypes.csv",
	index_col=0,
)
construct_or = [n.replace("_add", "").replace("_remove", "") for n in annotation_or.columns]

variable_name_to_or_map = {
	"passive_suicidal_ideation": "passive_si",
	"physical_abuse_violence": "abuse_physical",
	"sexual_abuse_harassment": "abuse_sexual",
	"social_withdrawal": "social_withdrawl",
	"defeat_feeling_like_a_failure": "defeat_failure",
	"burdensomeness": "burdensomeness",
	"shame_self_disgust_worthlessness": "shame_self-disgust",
	"guilt": "guilt",
	"emotional_pain_psychache": "emotional_pain",
	"panic": "panic",
	"grief_bereavement": "grief_bereavement",
	"emptiness": "emptiness",
	"alcohol_use": "alcohol_use",
	"other_substance_use": "substance_use",
	"impulsivity": "impulsivity",
	"aggression_irritability": "aggression_irritability",
	"anhedonia_uninterested": "anhedonia_uninterested",
	"rumination": "rumination",
	"anxiety": "anxiety",
	"trauma_ptsd": "did not exist",
	"entrapment_desire_to_escape": "entrapment",
	"hopelessness": "hopelessness",
	"perfectionism": "perfectionism",
	"physical_health_issues_disability": "did not exist",
	"agitation": "agitation",
	"fatigue_tired": "fatigue_tired",
	"sleep_issues": "sleep_issues",
	"discrimination": "discrimination",
	"barriers_to_treatment": "barriers_to_treatment",
	"bullying": "bully",
	"suicide_exposure": "did not exist",
	"finances_work_stress": "finances_work",
	"gender_sexual_identity": "gender_sexual_identity",
	"lethal_means_for_suicide": "did not exist",
	"psychosis_schizophrenia": "did not exist",
	"borderline_personality_disorder": "did not exist",
	"eating_disorders": "eating_disorder",
	"depressed_mood": "depressed_mood",
	"loneliness_isolation": "loneliness_isolated",
	"poverty_homelessness": "did not exist",
	"bipolar_disorder": "did not exist",
	"mental_health_treatment": "did not exist",
	"incarceration": "did not exist",
	"other_suicidal_language": "did not exist",
	"active_suicidal_ideation_suicidal_planning": "active_si",
	"direct_self_injury": "self-injury",
	"relationships_relationship_issues": "relationship",
}

run_this = False

if run_this:
	for c in lexicon.constructs:
		tokens = lexicon.constructs[c]["tokens"]
		variable_name = lexicon.constructs[c]["variable_name"]
		variable_name_or = variable_name_to_or_map.get(variable_name)

		if variable_name_or not in construct_or:
			print(f"{variable_name} not in annotation_or")
		else:
			print(c)
			# choose tokens OR kept and added
			annotation_or_i = annotation_or[
				(annotation_or[variable_name_or + "_add"] == 1) | (annotation_or[variable_name_or + "_remove"] == "0")
			][variable_name_or].tolist()
			not_in_current_lexicon_i = [n for n in annotation_or_i if n not in tokens]
			print(c, not_in_current_lexicon_i)
		print()
		print()



# TODO look into lexicon.constructs[construct].update(b) for adding multiple at the same time

construct = "Other suicidal language"
lexicon.add(
	construct,
	section="tokens",
	value=[
		"hate my life",
		"no one cares if I live or die",
		"wish I'd never been born",
		"I could just die",
		"when I die",
		"when I'm dead",
		"when I'm gone",
		"I should die"
	],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

lexicon.constructs[construct]['tokens']

# construct = "Lethal means for suicide"
# lexicon.add(
# 	construct,
# 	section="tokens",
# 	value=[],
# 	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
# )



# TODO: Be able to access constructs via name or variable_name



construct = "Passive suicidal ideation"
lexicon.add(
	construct,
	section="tokens",
	value=["I could disappear","wish I was dead","want to be done with this", 'forced to exist', 'want nonexistance', 'wish nonexistance'
		   ],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Active suicidal ideation & suicidal planning"
lexicon.add(
	construct,
	section="tokens",
	value=['I collected pills', 'I gave my things away', 'I wrote a suicide note', 'auto exhaust', 'drive my car into', 'giving away my possessions', 'self-immolation', 'jump into traffic', 'jump off my balcony', 'jump off a ledge', 'make a will',  'rope around my neck', 'set myself on fire', 'take all my pills', 'buying a gun', 'buying a rope', 'buying pills', 'hanging myself', 'thought about jumping off', 'write a will', 'hit by a train', 'hit by the train', 'crash my car', 'crash my truck', 'wreck my car', 'wreck my truck', 'slam my car', 'slam my truck', 'blow my brains out', 'slit my wrist', 'slash my wrist',  'jump out the window', 'jump out of the window', 'jump off the roof', 'jump from the roof', 'jump off the building', 'jump off the top of the building', 'jump off of the top of the building', 'jump off of the top of', 'drink myself to death', 'drown myself', 'burn myself alive', 'take my own life', 'end my own life'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Lethal means for suicide"
lexicon.add(
	construct,
	section="tokens",
	value=['car exhaust', 'hydrogen', 'carbon', 'monoxide', 'helium', 'noose', 'knife', 'razor', 'blade', 'jump into traffic', 'jump from a balcony', 'drive my car into', 'jump off a ledge', 'rope', 'hanging', ],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Direct self-injury"
lexicon.add(
	construct,
	section="tokens",
	value=['bit myself', 'carved my skin', 'cut skin', 'gave myself a tattoo','stick and poke tattoo','homemade tattoo',  'picked at a wound', 'scraped my skin', 'self injury', 'hurt myself on purpose', 'intentionally hurt myself'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Physical abuse & violence"
lexicon.add(
	construct,
	section="tokens",
	value=["boxing", "clobber", "abusive", "attack", "affront",'against my will', 'bash my', 'was beaten', 'used the belt', 'bit me', 'blow to the', 'boo-boo', 'bop', 'broken bones', 'broken nose', 'bruise', 'charge', 'clobber','conflict','harm me', 'hitting',  'kicked', 'murder',  'punish', 'pushed me',  'gunshot', 'shot me', 'took out a gun', 'gun', 'shout', 'slap', 'smack',  'spank', 'violent',  'child abuse'
		   ],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)



construct = "Sexual abuse & harassment"
lexicon.add(
	construct,
	section="tokens",
	value=['abuse', 'assail', 'assault', 'attacked me', 'carnal abuse', 'date rape', 'predator', 'he forced me to','she forced me to', 'rape', 'threatening sex', 'unwanted sex',
		   'sex crime', 'sex offense','statutory rape', 'touched me in a sexual', 'touched my leg', 'touched my ass', 'unlawful sexual', 'violate', 'violation', 'inappropriate touch', 'take advantage of m', 'violate me', 'nonconsensual', 'non-consensual', 'revenge porn', 'blackmail', 'extortion', 'stalk', 'ghb', 'forced me to have sex', 'forced me to touch', 'dick', 'penis', 'cock', 'vagina', 'pussy', 'fanny', 'vag', 'cum', 'anal'
		   ],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)


construct = "Social withdrawal"
lexicon.add(
	construct,
	section="tokens",
	value=["don't go anywhere", 'I avoid people', 'I avoid crowds', "I don't go out", 'I used to go out', 'I used to go to parties', 'I used to go places', "I don't go places", 'I stay at home', 'I keep to myself', 'I stick to myself', 'withdrawn', 'social withdrawl'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Defeat & feeling like a failure"
lexicon.add(
	construct,
	section="tokens",
	value= ['complete failure', 'feel powerless',"am powerless", "I'm a loser", 'no fight left in me',  'beaten', 'conquered', 'crushed', 'disappointed', 'disappointingly', 'discomfited', 'give up', 'unsuccessful'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)


construct = 'Burdensomeness'
lexicon.add(
	construct,
	section="tokens",
	value= ['I annoy', 'I bother', 'demands I make', 'I have negative effects on', 'make things hard on others', "I'm too much trouble", 'difficult for me to ask for help', "can't give anything in return", 'dead weight', 'encumbrance', 'feel guilty about others caring for me', 'imposition','imposiing', 'incumbrance', 'bother'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Shame, self-disgust, & worthlessness"
lexicon.add(
	construct,
	section="tokens",
	value=['ashamed of myself', 'despise myself', 'detest myself', 'I do things I find repulsive', 'humillated', 'unattractive', 'I find myself repulsive', 'I hate some things about me', "can't look at myself", 'demeaned', 'discredited', 'disgraced', 'disgusted with myself', 'dishonor', 'distraught', 'distressed', 'feel disappointed in myself', 'hate myself', 'pathetic', 'remorse', 'shamed', 'sheepish', 'shy', 'stammer', 'stutter',  'stigma', 'am messed up', 'am fucked up'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Hopelessness"
lexicon.add(
	construct,
	section="tokens",
	value=[""],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)



construct = "Loneliness & isolation"
lexicon.add(
	construct,
	section="tokens",
	value=["no one cares if I live or die",],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)


construct = "Guilt"
lexicon.add(
	construct,
	section="tokens",
	value=['my mistake', 'critical of myself', 'I blame myself','complicity', 'culpable','culpability', 'dereliction', 'error', 'fault', 'guilt by association', 'guilty feeling', 'guilt trip', 'guiltiness', 'guilty conscience', 'impeach', 'indict', 'accuse', 'indiscretion', 'infamy', 'lapse of judgement', 'liability', 'misbehavior', 'misconduct', 'misstep', 'offense', 'onus', 'peccability', 'penitence', 'regret', 'remorse', 'responsibility', 'responsible', 'self-condemnation', 'self-reproach', 'shame',  'sinful', 'transgression', 'wicked', 'wrong', 'blame myself', 'messed up', 'made a mistake', 'was wrong', "I'm sorry", 'I am sorry', 'taboo', 'fucked up'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)


construct = "Emotional pain & psychache"
lexicon.add(
	construct,
	section="tokens",
	value=['aching', 'affliction','torment', 'torture'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Panic"
lexicon.add(
	construct,
	section="tokens",
	value=['fright', 'afraid', 'dread', 'fearfulness', 'fit of terror', 'horror', 'hysteria', 'ordinary things looked strange', 'something happening to my body', 'strange sensations', 'terrified', 'terror', 'thought something will kill me'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Grief & bereavement"
lexicon.add(
	construct,
	section="tokens",
	value=['cannot move on', "can't accept her death","can't accept his death", 'difficult to accept his death','difficult to accept her death', 'accepting his death', 'accepting her death', 'heartache', 'heartbreak', 'infidelity','cheated on me','longing', 'our loss','my loss', 'mournfulness', 'moving on has been', 'she died','he died',  'they died',  'he is dead', 'she is dead', 'they are dead', 'he was killed', 'she was killed', 'they were killed', 'he was murdered', 'she was murdered', 'they were murdered', 'he died', 'she died'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Emptiness"
lexicon.add(
	construct,
	section="tokens",
	value=['absent in my own life', 'I feel nothing', 'I feel numb', 'I am nothing'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)


construct = "Alcohol use"
lexicon.add(
	construct,
	section="tokens",
	value=['alcoholic beverage', 'alcoholic drink', 'beverage', 'brewery', 'brewage', 'canned heat', 'cordial', 'a drink','for drinks', 'hard cider','hard stuff', 'home brew', 'homebrew', 'hooch', 'hootch', 'inebriant', 'intoxicant', 'liqour', 'liqueur', 'methanol', 'mixed drink', 'nipa', 'sake',  'whisky', 'shitfaced', 'keg', 'smirnoff',  'bacardi', 'johnnie walker', 'johnny walker', 'jack daniel', 'captain morgan', 'jim beam', 'jameson', 'crown royal', 'baileys', 'Jagermeister', 'Jager', 'Aperol', 'budweiser', 'bud light', 'heineken', 'corona', 'angry orchard', 'guinness', "tito's", 'white claw', 'hard seltzer', 'tonic', 'hard lemonade', 'ciroc', 'patron', 'cuervo', 'riesling', 'hennessy', 'margarita', 'sangria', 'white russian', 'negroni', 'martini', 'daiquiri', 'mojito', 'moscow mule', 'bloody mary', 'mai tai', 'gimlet', 'Caipirinha', 'Piña Colada', 'Sazerac', 'Long Island Iced Tea', 'everclear', 'wild turkey', 'sherry', 'kahlua', 'campari', 'tanqueray', 'tom collins', 'bourbon', 'rye', 'fireball', 'mezcal', 'cognac', 'chardonnay', 'sauvignon blanc', 'pinot gris', 'pinot noir', 'zinfadel', 'syrah', 'cabernet', 'chug', 'take shots of', 'took a shot of', 'drink shots', 'drink a shot', 'bought a round', 'buy a round', 'pay my tab', 'paid my tab', 'the bar', 'the club', '12 oz'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)
construct = "Other substance use"
lexicon.add(
	construct,
	section="tokens",
	value=['medicinal', 'medication', 'meds', 'controlled substance', 'diuretic','do drugs', 'dope', 'dose', 'hallucinate', 'hash',  'intoxicant',  'medicinal','medicine', 'mind-altering', 'popper', 'prescription','psychoactive','psychotropic','salvia', 'sedative', 'stimulant', 'tonic', 'trental', 'we were high', 'I got high', 'cialis', 'viagra'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)


construct = "Impulsivity"
lexicon.add(
	construct,
	section="tokens",
	value=['spur of the moment', 'on impulse', 'I change jobs a lot',"don't plan things", 'say things without thinking', "speak without thinking", 'spend too much', 'ad-lib', 'careless',  'flaky', 'gone off the deep end', 'instinctive', 'intuitive', 'jump the gun', 'make up my mind quickly', 'offhand', 'rash decision', 'unconsidered', 'unexpected', 'unmeditated', 'unpremeditated', 'unprompted', 'winging it', 'urge', 'just did it', 'just do it'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Aggression & irritability"
lexicon.add(
	construct,
	section="tokens",
	value=['animosity', 'anti', 'antipathetic',  'barbaric','bitter',  'competitive', 'complaining', 'cynicism',  'disapproving', 'displeased', 'disruptive', 'dissatisfied', 'disturbing', 'easily offended', 'embittered',  'enraged', 'exacerbated', 'exasperated','fretting', 'gloomy','harsh', 'hateful', 'hostility',  'hypercritical', 'ill will', 'in-your-face','indignant', 'intruding', 'intrusive', 'invading', 'irate', 'livid', 'mad',  'malevolent', 'malicious', 'malignant', 'moody',  'offended', 'offensive', 'opponent', 'opposed', 'opposing', 'outraged', 'oversensitive', 'pissed off',  'predatory', 'put out', 'quarrel',  'quick-tempered', 'raging','resentment', 'riled','spiteful', 'tense', 'threatening','unkind', 'antisocial', 'unwelcoming', 'uptight', 'violent',  'virulent', 'wrath'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)
construct = "Anhedonia & uninterested"
lexicon.add(
	construct,
	section="tokens",
	value=["I can't enjoy", "I don't enjoy", "don't feel pleasure", "I haven't enjoyed", "I won't enjoy", 'bored', 'could care less', 'detached', 'disinterested','dissatisfied',  "don't enjoy things", "don't get real satisfaction", 'go through the motions', 'hard-hearted', 'unconcerned', 'uncurious', 'unsatisfying', 'nothing feels good', 'nothing is fun', "I don't enjoy anything", 'why bother'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Rumination"
lexicon.add(
	construct,
	section="tokens",
	value=["couldn't concentrate", "can't decide whether", "couldn't think", 'torturing me', 'deliberate','feel my head could explode','indecisive', 'my head hurts', 'too many thoughts', 'introspection', 'overwhelming thoughts', 'thinking too much', 'rationalization', 'rationalize', 'feel judged', 'worried','moral dilemma', 'speculation', "thoughts I couldn't control", 'thoughts were racing', 'trouble falling asleep', 'get to the bottom of', 'trying to figure out', 'think about it over and over'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Anxiety"
lexicon.add(
	construct,
	section="tokens",
	value=['acrophobic', 'worst will happen', 'aghast', 'agoraphobic', 'alarmed', 'algophobic', 'anguish', 'ants in my pants',  'aquaphobic', 'bad news',  'bothered', 'butterflies',  'phobic','cold sweat','cold feat', 'concern', 'cowardly',  'distraught', 'distressed', 'disturb', "don't feel at ease", 'fear of losing control','intolerance to uncertainty',  'fidget', 'goose bumps', 'hands trembling',  'horrified',  'hypochondria','indecisive', 'indigestion', 'insecure', 'low self esteem', 'intimidated', 'mistrust', 'nail-biting','nervousness','on edge', 'on pins and needles','petrified', 'problem','rattled', 'run scared', 'scare', 'spooked', 'startled',  'terrified', 'timid', 'torment','torture', 'trial', 'troubled', 'unable to relax', 'uncertainty', 'unease', 'unnerved'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Entrapment & desire to escape"
lexicon.add(
	construct,
	section="tokens",
	value=['no good solution','I have no control',  'want to start again',"a fresh start", "I've lost control", 'captured', 'doom', 'entrap', 'helpless', 'snare', 'there is no escape', 'there is no exit', 'no escape', "I can't avoid", 'avoidance', "I can't evade", "I'm stuck"],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Hopelessness"
lexicon.add(
	construct,
	section="tokens",
	value=["I can't imagine what my life would be like", "can't pursue my goals", "don't expect to succeed", "don't have faith", "don't see good things ahead", 'I never get what I want', 'bad things ahead', "I won't accomplish", "can't make things better", 'cynical', 'desperate',  'discouraged', 'discouraging', 'expect the worse','futureless', 'goner', "it's impossible", 'incurable','irredeemable', 'irreparable', 'irreversible', 'irrevocable', 'my future seems dark', 'no reason to believe', 'no use in trying', 'the future seems uncertain', 'the future seems vague', "things won't work out as I want",  'unachievable','useless pursuit', "won't get it", "won't get the breaks", "won't get what I want", 'done for'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Perfectionism"
lexicon.add(
	construct,
	section="tokens",
	value=['upset with me when I slip up', 'less than excellent', 'being judged', 'judge', 'slip up','top-notch quality', 'perfection','workaholic', 'perfectionistic', 'setting unrealistic goals', 'hate mistakes', "strive to better", 'I criticize',"accepting second best", 'give up too easily', 'I demand nothing less than', 'people who are average','mediocre', 'I expect a lot', "others to excel", 'too demanding', "expectations","high expectation", 'high goals', 'I must always', "I should always", "successful",'straight As', 'I must work',"full potential", 'need to be perfect', 'be the best', 'flawless', 'see an error in', 'try their hardest', 'absolute best', 'My family expects me', 'My parents expected me','excel in all aspects', "I don't succeed", "don't think I'm competent","I make a mistake", 'expect more from me', 'expect nothing less than', 'I work even harder','please others', 'people pleaser', 'the better I do','succeed at everything', 'should never let me down', "I can make mistakes", 'be perfect in everything I do', 'best', 'excellent', 'first place', 'high expectation', 'mistake', 'perfect', 'winner'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)


construct = "Agitation"
lexicon.add(
	construct,
	section="tokens",
	value=['aroused', 'bundle of nerves', 'bustling', 'compulsion to move', 'fidgeting','footloose', 'frenetic', 'frenzied', 'fretful', 'hectic', 'hurried', 'itchy', 'jolted', 'jumpy', 'mentally unsettled', 'phrenetic', 'racing heart', 'restless',  'sleepless', 'strung-out', 'tossing and turning', 'wandering', 'in motion'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)


construct = "Fatigue & tired"
lexicon.add(
	construct,
	section="tokens",
	value=['too tired',"can't move a finger",  "can't do anything at all", "I don't do much", "don't have energy", 'mentally exhausted', 'no desire to do anything', 'exhausted', 'tired', 'problems starting things', 'problems thinking clearly', 'push myself very hard to do anything', 'effort to get started', 'bleary','broken-down', 'burned out', 'burned-out', 'burnt-out','collapsing', 'comatose', "dead on one's feet", 'dead tired', 'draggy', 'drooping', 'droopy', 'exhaustion',  'faint', 'fed up', 'narcolep', 'prostrate', 'ready to drop', 'sick of', 'sleep',  'slumber', 'slumbersome', 'snooz',  'unrefreshed', 'not refreshed', 'unrested', 'worn-out', 'yawn', 'I am spent', 'I was spent'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)


construct = "Sleep issues"
lexicon.add(
	construct,
	section="tokens",
	value=['difficulty sleeping', "don't sleep as well", 'frightening dreams', 'back to sleep', 'restless', 'sleepless', 'sleepy', 'suddenly awake', 'urge to move legs at night', 'wake up during the night', 'wake up earlier than usual', 'trouble sleeping', 'apnea', 'cpap', 'c-pap', 'sleepwalk'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)


construct = "Discrimination"
lexicon.add(
	construct,
	section="tokens",
	value=['harassed', 'threatened', 'treated with less courtesy', 'treated with less respect', 'I receive poorer service', 'call me names', 'insult me', 'able-bodism', 'ableism', 'ablism', 'ageis', 'agis', 'ancestry', 'anti-semitism','antisemitism','antisemite', 'anti','antiracis', 'antifeminis', 'bigot', 'be partial',  'chauvinis',  'discriminatory', 'national origins','ancestry',  'afraid of me', 'think I am dishonest',"think we're dangerous",'think I am not smart', 'preconception', 'prejudiced', 'race', 'racialism', 'zealot', 'fanatical', 'sectarian', 'segregate', 'separate', 'ghetto','concentration camps','gulag', 'skinhead', 'proud boy', 'discrimination', 'sexual orientation','deadname', 'show bias', 'skin color','treat as inferior', 'treat differently', 'tribe', 'ingroup', 'outgroup', 'unfair', 'unfairness', 'victimize', 'fat shaming', 'white supremacy','xenophobia', 'caste system', 'nazi', 'incel', 'MAGA', 'immigrant', 'migrant', 'refugee', 'make america great again','where you came from'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Barriers to treatment"
lexicon.add(
	construct,
	section="tokens",
	value=['avoid telling people about my mental', 'hide my mental', 'hard telling people I have',  "I wouldn't look for help", "I'm not treated well because I'm old", "I'm scared of how people will react if they find out", "I've been discriminated", 'People have avoided me', 'afraid of therapy', "can't find a psychotherapist that works", "can't pay for treatment", "can't travel to the hospital", 'discriminated because of my mental health problems',"stigma", "don't have insurance", 'being talked down to','therapists are condescending', 'doctors are condescending', 'going rate is too high', 'find psychiatrist', 'find psychotherapist', 'long waitlist','long wait list', "medications don't work", 'my insurance limits my psychotherapy', 'no one referred me to a',"I haven't been reffered yet", 'normal people do not go to psychotherapy', "people's reaction to my mental",  'professionals are not qualified',  'worry about telling people I receive psych',"treatments aren't effective", 'handle the problem on my own', 'I talk to friends instead', 'therapy costs too much', 'I am unsure of who to see', "I don't have time for therapy", "I don't have enough money for therapy",  "I don't need treatment", "I don't need therapy", 'side effects', 'my medication makes me', 'my meds make me', 'my medication interferes with', 'my meds interfere with', "don't want to go to the hospital", "don't want to be locked up", "don't want to be hospitalized", 'was hospitalized', 'was forced to go to the hospital', "I can't be helped", "I can't be fixed", "therapy isn't worth it", "treatment isn't worth it", "therapists don't listen", "therapists don't care", "psychiatrists don't listen", "psychiatrists don't care", "doctors don't listen", "doctors don't care", "nurses don't listen", "nurses don't care", "doctors can't", "nurses can't", "therapists can't", "psychiatrists can't", "therapy can't", "treatment can't", "pills can't", "medications can't", "meds can't", "drugs can't"],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Bullying"
lexicon.add(
	construct,
	section="tokens",
	value=["I don't want to go to school", "I'm being bullied", 'threatened',  'aggressor', 'boss around', 'bossy', 'boys can be so cruel','girls can be so cruel','embarrassed me', 'get bullied at school', 'took my lunch money', 'humilliate me', 'hurt by another student', 'everyone ignores me', 'joke about me', 'leave me out', 'make fun of me', 'mean boy', 'mean girl', 'mean kid', 'mean to me', 'memories of being bullied', 'mock me', 'post mean comments about me', 'post mean things about me', 'post mean pictures about me', 'pretended to be sick', 'push me around','share my secrets',   'spread rumors about me', 'stay home from school', 'talk bad about me', 'talk behind my back', 'tease', 'tease me','terrorize', 'text mean messages about me',  'threat', 'tormenter', 'tough guy', 'tried to get me in trouble', 'try to make me feel bad', 'turn others against me', 'yobbo', 'yobo', 'post nude photos', 'leak nude'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = "Finances & work stress"
lexicon.add(
	construct,
	section="tokens",
	value=['dollar', 'asset', 'bank', 'bankroll', 'bankruptcy', 'bill', 'bucks', 'budget', 'buy', 'capital', 'cash', 'cheap', 'coin', 'coinage', 'costly', 'unemployed', 'currency', 'debts', 'depleted', 'destitute', 'do work', 'dollar', 'bitcoin','crypto', 'employee','trust fund','trustfund', 'funds',  'hustler', 'impoverished', 'chapter 11', 'income','deposit',  'inflation', 'insolvent', 'landlord',   'occupation', 'out of business', 'overworked', 'owner', 'paid', 'payroll', 'pesos', 'price', 'property', 'purchase', 'retire', 'rich', 'tenant', 'wealth', 'workplace', 'paycheck', 'unemploy', 'gambling'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = 'Gender & sexual identity'
lexicon.add(
	construct,
	section="tokens",
	value=['bgltq'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = 'Eating disorders'
lexicon.add(
	construct,
	section="tokens",
	value=["I'm not eating", "I'm restricting", 'anorexia nervosa','binge eating', 'binge-vomit syndrome', 'bingeing', 'bulimarexia', 'bulimia', 'burn off calories', 'diet powder', 'diuretics', 'eat a lot', 'eating in secret',  'felt fat', 'flat stomach', 'gain weight', 'go on a diet', 'hyperphagia', 'lose weight', 'losing control over eating', 'loss of appetite', 'overweight', 'polyphagia', 'throw up', 'weight is important to me', 'work out too much', 'worry about my body', 'worry about weight', 'physical appearance', "don't eat"],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = 'Depressed mood'
lexicon.add(
	construct,
	section="tokens",
	value=['sad', 'unhappy', "can't snap out of it", 'cry', 'I feel sad all the time',  'bummed out', 'cast down', 'cheerless', 'crummy', 'desolate', 'despair',  'distressed', 'feeling down', 'down and out',  'down in the dumps',  'gloomy', 'glum', 'grim',  'heavy-hearted', 'heavyhearted',  'in the dumps','languish',  'low-spirited', 'melancholic', 'moody', 'not happy',  'spiritless',  'tearful', 'teary', 'tragic','unhappy', 'upset', 'weep'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)

construct = 'Loneliness & isolation'
lexicon.add(
	construct,
	section="tokens",
	value=["I'm all alone",'companionless', 'lack of companionship', 'dejected', 'me and my shadow', 'me, myself and I', 'no one cares about me', 'no one thinks of me',  'rejected', 'renounced', 'solo', 'left unattended', 'unmarried',  'widow', 'I am alone', 'I am not close to anyone', "I don't belong", "I don't matter to other", "I don't play an important role", 'isolated', 'no one I can talk to'],
	source="DML adding after viewing additions from thesauri, questionnaires, and OR annotation of old draft lexicon",
)



# Regenerate lexicon with GPT-4 for 2 constructs (Direct self-injury, Relationship issues) with new definition and one new construct (Existential meaninglessness & purposelessness)
# ========================================================================
# load api keys
import os
import litellm
import api_keys

os.environ["OPENAI_API_KEY"] = api_keys.open_ai  # string, you need your own key and to put at least $5 in the account
litellm.set_verbose=False
litellm.drop_params=True

# Config
# ==============================
# domain = None
domain = "mental health"
# model = "command-nightly"
model = "gpt-4-1106-preview"
# model = 'gemini-pro'
seed = 42

# Just create and show which items are different. Then add manually

run_this = False
if run_this:
	construct = 'Direct self-injury'
	lexicon.constructs[construct]['definition'] = "Mention of actual or desired direct and deliberate destruction by an individual of their own body tissue, including with suicidal intent (suicidal self-injury) our without suicidal intent (non-suicidal self-injury)"
	definition = lexicon.constructs[construct]['definition']
	examples = lexicon.constructs[construct]['examples']
	prompt = generate_prompt(construct, prompt_name='types of direct self-injury', domain = domain, definition = definition, examples = examples)
	print(lexicon.constructs['Direct self-injury']['tokens_metadata'].keys())
	lexicon.add(construct+' 2', value = 'create', prompt = prompt, source = model, temperature=0,seed=42,max_tokens = 150, top_p=0.9)
	print(len(lexicon.constructs[construct+ ' 2']['tokens']))
	lexicon.add(construct+' 2', value = 'create', prompt = prompt, source = model, temperature=0.5, seed=42, max_tokens = 150, top_p=0.9)
	print(len(lexicon.constructs[construct+ ' 2']['tokens']))
	lexicon.add(construct+' 2', value = 'create', prompt = prompt, source = model, temperature=1, seed=42, max_tokens = 150, top_p=0.9)
	print(len(lexicon.constructs[construct+ ' 2']['tokens']))

	original_tokens = lexicon.constructs[construct]['tokens']
	new_tokens = [n for n in lexicon.constructs[construct+ ' 2']['tokens'] if n not in original_tokens]
	new_tokens_to_add = ['NSSI', 'non-suicidal self-injury', 'break my bones', 'choking myself', 'deliberate self-harm', 'dermotillomania',  'self-amputation',  'self-assault', 'self-banging', 'self-battering', 'self-castration', 'self-choking', 'self-disfigurement', 'self-head banging', 'self-immolation', 'self-injurious', 'self-picking', 'self-piercing','self-stabbing',  'self-tattooing',  'wound myself']
	lexicon.add(construct, section='tokens', value = new_tokens_to_add, source = f'manually added by DML from GPT-4 {model} output with new definition')


# Relationship issues
run_this = False
if run_this:
	construct = 'Relationship issues'
	definition = lexicon.constructs[construct]['definition']
	examples = lexicon.constructs[construct]['examples']
	lexicon.constructs[construct]['prompt_name'] = 'negative relationship types and processes'
	prompt_name = lexicon.constructs[construct]['prompt_name']
	prompt = generate_prompt(construct, prompt_name=prompt_name, domain = None, definition = None, examples = examples)
	print(prompt)
	print(lexicon.constructs[construct]['tokens_metadata'].keys())
	lexicon.add(construct+' 2', value = 'create', prompt = prompt, source = model, temperature=0,seed=42,max_tokens = 150, top_p=0.9)
	print(len(lexicon.constructs[construct+ ' 2']['tokens']))
	lexicon.add(construct+' 2', value = 'create', prompt = prompt, source = model, temperature=0.5, seed=42, max_tokens = 150, top_p=0.9)
	print(len(lexicon.constructs[construct+ ' 2']['tokens']))
	lexicon.add(construct+' 2', value = 'create', prompt = prompt, source = model, temperature=1, seed=42, max_tokens = 150, top_p=0.9)
	print(len(lexicon.constructs[construct+ ' 2']['tokens']))

	original_tokens = lexicon.constructs[construct]['tokens']
	new_tokens = [n for n in lexicon.constructs[construct+ ' 2']['tokens'] if n not in original_tokens]
	new_tokens_to_add = ['abandonment', 'abuse', 'acrimony', 'adultery',  'we got into an argument',  'bickering','co-dependency', 'cold-shoulder','controlling behavior',  'defensiveness', 'disloyalty', 'distance between us', 'distrust', 'dysfunctional', 'emotional abuse','psychological abuse','verbal abuse',  'exploitation','grudge','inattentive', 'jealousy', 'loveless', 'manipulative', 'mismatch', 'we had a misunderstanding', 'neglect', 'passive-aggressive', 'physical abuse', 'possessive', 'power struggle', 'resentment', 'rivalry', 'sexual abuse', 'silent treatment', 'stonewall',  'unfaithful','unsupportive', 'vengeful', 'domestic violence']
	lexicon.add(construct, section='tokens', value = new_tokens_to_add, source = f'manually added by DML from GPT-4 {model} output with new definition')

# Existential meaninglessness & purposelessness
run_this = False
if run_this:
	construct = 'Existential meaninglessness & purposelessness'
	definition = lexicon.constructs[construct]['definition']
	examples = lexicon.constructs[construct]['examples']

	prompt_name = lexicon.constructs[construct]['prompt_name']
	prompt = generate_prompt(construct, prompt_name=prompt_name, domain = None, definition = None, examples = examples)
	print(prompt)
	print(lexicon.constructs[construct]['tokens_metadata'].keys())
	lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=0,seed=42,max_tokens = 150, top_p=0.9)
	print(len(lexicon.constructs[construct]['tokens']))
	lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=0.5, seed=42, max_tokens = 150, top_p=0.9)
	print(len(lexicon.constructs[construct]['tokens']))
	lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=1, seed=42, max_tokens = 150, top_p=0.9)
	print(len(lexicon.constructs[construct]['tokens']))

	original_tokens = lexicon.constructs[construct]['tokens_metadata']['DML adding manually 24-01-22T00-13-36']['tokens']
	new_tokens = [n for n in lexicon.constructs[construct]['tokens'] if n not in original_tokens]
	new_tokens_to_add = ['absurdity of life', 'aimless existance', 'alienated', 'existential crisis', 'existing is futile', 'my life is insignificant', 'life is aimless', 'life is directionless', 'life is empty', 'life is futile', 'life is purposeless', 'life is senseless', 'life is trivial', 'life is unfulfilling', 'life is unimportant',  "I'm so lost", 'meaningless existence','nihilist', 'no ambition', 'no aspiration', 'no direction', 'no drive', 'no end goal', 'no goal in life', 'no meaning in life', 'no motivation', 'no reason', 'no reason for being', 'no sense of purpose', 'no ultimate meaning', 'no ultimate purpose', 'purposeless', 'senseless of life', 'unfulfilled',  'without importance', 'without meaning','without purpose']
	lexicon.add(construct, section='tokens', value = new_tokens_to_add, source = f'manually added by DML from GPT-4 {model} output with new definition')


# Relationships & kinship
run_this = False
if run_this:
	construct = 'Relationships & kinship'
	lexicon.add(construct, section ='definition',value = 'Types of family, friends, and other key social relationships')
	lexicon.add(construct, section ='definition_references',value = 'Ours')
	lexicon.add(construct, section ='examples',value = 'mom; dad; brother; sister; girlfriend; boyfriend; partner; spouse; husband; wife' )
	lexicon.constructs[construct]['prompt_name'] = 'Types of family, friends, and other key social relationships and kinship'
	tokens = ['ancestor', 'aunt', 'bride', 'brother', 'cousin', 'dad', 'mom', 'bro', 'bros', 'daddy', 'daughter', 'family', 'father', 'genetic', 'grandchild', 'grandchildren', 'grandfather', 'grandma', 'grandmother', 'grandpa', 'acquaintance', 'acquainted', 'buddy', 'mate', 'roomate', 'neighbor', 'soulmate', 'granny', 'honeymoon', 'husband', 'kin', 'kindred', 'mama', 'marital', 'marriage', 'niece', 'nephew', 'husband', 'wife', 'sibling', 'house mate', 'I live with', 'spouse', 'partner', 'my baby', 'our baby', 'newborn', 'marry', 'mother', 'nephew', 'papa', 'parent', 'relative', 'sister', 'son', 'twin', 'uncle', 'wedding', 'widow', 'bf', 'gf', 'supportive', 'boyfriend', 'girlfriend', 'child', 'caregiver', 'my kid', 'my ex', 'fiance', 'fiancé', 'married', 'friend', 'wife', 'wives', 'relationship', 'nurture', 'nurturing', 'romance', 'romantic']
	lexicon.add(construct, section ='tokens',value = tokens, source="Added manually by DML based on SEANCE's General Inquirer kinship for relationships")

	definition = lexicon.constructs[construct]['definition']
	examples = lexicon.constructs[construct]['examples']

	prompt_name = lexicon.constructs[construct]['prompt_name']
	prompt = generate_prompt(construct, prompt_name=prompt_name, domain = None, definition = None, examples = examples)
	print(prompt)
	print(lexicon.constructs[construct]['tokens_metadata'].keys())
	lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=0,seed=42,max_tokens = 150, top_p=0.9)
	print(len(lexicon.constructs[construct]['tokens']))
	lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=0.5, seed=42, max_tokens = 150, top_p=0.9)
	print(len(lexicon.constructs[construct]['tokens']))
	lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=1, seed=42, max_tokens = 150, top_p=0.9)
	print(len(lexicon.constructs[construct]['tokens']))

	original_tokens = lexicon.constructs[construct]['tokens_metadata']["Added manually by DML based on SEANCE's General Inquirer kinship for relationships 24-01-22T00-51-48"]['tokens']
	new_tokens = [n for n in lexicon.constructs[construct]['tokens'] if n not in original_tokens]
	new_tokens_to_add =['adopt', 'adoptive child',  'best friend', 'brother in law', 'sister in law', 'classmate', 'co-parent', 'colleague', 'companion','confidant', 'daughter in law', 'father in law', 'foster child', 'foster children', 'foster parent', 'goddaughter', 'godfather', 'godmother', 'godparent', 'godson', 'granddaughter',  'grandparent', 'grandson',  'half-brother', 'half-sister', 'housemate', 'in-laws', 'mentor', 'mother in law', 'peer', 'protege', 'protégé', 'roommate', 'significant other', 'sister in law', 'son in law', 'stepbrother', 'stepdaughter', 'stepfather', 'stepmother', 'stepsister', 'stepson', 'teammate']
	lexicon.add(construct, section='tokens', value = new_tokens_to_add, source = f'manually added by DML from GPT-4 {model} output')

# TODO remove plurals from genAI output?



# Save
lexicon.save(f'data/lexicons/suicide_risk_lexicon_gpt-4-1106-preview_dml_{generate_timestamp()}')

# Testing
# ===================================================
construct ='Bullying'
lexicon.constructs[construct]['tokens_metadata'].keys()
'anxiety' not in lexicon.constructs[construct]['tokens']
'slander' in lexicon.constructs[construct]['tokens']


construct ='Direct self-injury'
lexicon.constructs[construct]['tokens_metadata'].keys()
'draw blood' not in lexicon.constructs[construct]['tokens']
'carve my skin' in lexicon.constructs[construct]['tokens']
'electrocuting myself' in lexicon.constructs[construct]['tokens']










# lexicon = load_lexicon("data/lexicons/suicide_risk_lexicon_gpt-4-1106-preview_dml_24-01-22T01-38-32.pickle")

# # Definintions were missing, add again
# lexicon_definitions = pd.read_csv('data/lexicons/suicide_risk_constructs_and_definitions.csv')
# for construct in lexicon.constructs.keys():
# 	print(construct)
# 	if construct in lexicon_definitions['construct'].values:
# 		definition_references = lexicon_definitions[lexicon_definitions['construct'] == construct]['definition_references'].values[0]
# 	else:
# 		definition_references = input()
# 	lexicon.constructs[construct]['definition_references'] = definition_references
# 	print(definition_references)
# 	print()



# TODO turn into function
# remove_from_all_constructs = ['0', 0, '']
# for construct in lexicon.constructs.keys():
# 	lexicon.constructs[construct]['tokens'] = [n.strip() for n in lexicon.constructs[construct]['tokens'] if n not in remove_from_all_constructs]
#
#
# # Save
# ts = generate_timestamp()
# lexicon.save(f'data/lexicons/suicide_risk_lexicon_gpt-4-1106-preview_dml_{ts}')
# lexicon_df = lexicon.to_pandas()
# lexicon_df.to_csv(f'data/lexicons/suicide_risk_lexicon_gpt-4-1106-preview_dml_{ts}_annotation.csv')
# lexicon_df = lexicon.to_pandas(add_annotation_columns=False)
# lexicon_df.to_csv(f'data/lexicons/suicide_risk_lexicon_gpt-4-1106-preview_dml_{ts}.csv')






len_lexicon = {}
for c in lexicon.constructs.keys():
	print(c)
	len_lexicon[c] = len(lexicon.constructs[c]['tokens'])
	print(lexicon.constructs[c].keys())
	print(lexicon.constructs[c]['tokens_metadata'].keys())
	print()

del lexicon.constructs['Existential meaninglessness & purposelessness 2']
import matplotlib.pyplot as plt

plt.interactive(False)  #need to set to False


plt.hist(list(len_lexicon.values()))
plt.show()

# TODO: Pure GPT-4 models should include tokens from these searches +' 2' not from the first searches.

# TODO: add code to obtain words from wordnet thesauri

#  =====

# Analayze matches by token lengths
# ====================================================
short_tokens = []
for c in lexicon.constructs.keys():
	tokens = lexicon.constructs[c]['tokens']

	short_tokens_i = [n for n in tokens if len(n) == 6]
	short_tokens.append(short_tokens_i)

short_tokens = [n for i in short_tokens for n in i]

for token in short_tokens:
	print(token)
	print(match(docs, token.lower()))
	print()




"""



"""

# For clinicians to evaluate
# ================================================================================

metadata_df = []
df_i = pd.DataFrame(['Prompt name', 'Definition', 'Definition references', 'Examples', 'Tokens'], columns = ['Construct'])
metadata_df.append(df_i)
for construct in lexicon.constructs.keys():
	df_i = pd.DataFrame([
		[lexicon.constructs[construct]['prompt_name']]+[lexicon.constructs[construct]['definition']]+[lexicon.constructs[construct]['definition_references']]+[lexicon.constructs[construct]['examples']]+lexicon.constructs[construct]['tokens']

	], index=[construct]).T
	metadata_df.append(df_i)
	df_i = pd.DataFrame([], columns = [f'{construct}_include', f'{construct}_add'])
	metadata_df.append(df_i)

metadata_df = pd.concat(metadata_df, axis=1, ignore_index=False)


metadata_df.to_csv(f'data/lexicons/suicide_risk_lexicon_annotate_clinicians_{ts}.csv')
metadata_df





import time
import json
import datetime
import api_keys  # this file should be in the same folder
import litellm
import time


input_dir = './data/'
output_dir = './data/lexicons/'

# construct = ['Suicidal constructs'] *3 + ['Interpersonal issues']*4 + ['Perception of self']* 3 + ['Affective'] * 5 + ['Substance use'] *2 + ['Externalizing']*2

# Generate lexicon
# ========================================================================
# load api keys

os.environ["OPENAI_API_KEY"] = api_keys.open_ai  # string, you need your own key and to put at least $5 in the account
# os.environ["COHERE_API_KEY"] = api_keys.cohere_trial  # string, you can get for free by logging into cohere and going to sandbox
# # pip install -q google-generativeai
# os.environ["GEMINI_API_KEY"] = api_keys.gemini  # here: https://makersuite.google.com/app/apikey
# os.environ["VERTEXAI_PROJECT"] = api_keys.vertex


litellm.set_verbose=False
litellm.drop_params=True

# Config
# ==============================
# domain = None
domain = "mental health"
# model = "command-nightly"
model = "gpt-4-1106-preview"
# model = 'gemini-pro'
seed = 42
# constructs = constructs[39:]
definitions_df = pd.read_csv('./data/lexicons/suicide_risk_constructs_and_definitions.csv', encoding='latin1')



constructs = definitions_df['construct'].values

definitions_df.columns
name = 'suicide_risk_lexicon'

# Initialize lexicon
lexicon = Lexicon()
lexicon.name = 'Suicide Risk Lexicon'
lexicon.description = 'Lexicon for 46 risk factors validated by clinical experts. If you use, please cite publication: Low et al (in prep.). Creating a Suicide Risk Lexicon with Generative AI and word scores.'

import dill
ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S') # so you don't overwrite, and save timestamp
for i, construct in enumerate(constructs):
	print(i, construct)
# for construct in ['Active suicidal ideation', 'Loneliness & isolation']:# 'Self-injury', 'Sexual abuse & harassment']:#, 'Anxiety', 'Gender & sexual identity']:
	# construct = "sexual_abuse_harassment"

	prompt_name  = definitions_df[definitions_df['construct']==construct]['prompt_name'].values[0]
	examples  = definitions_df[definitions_df['construct']==construct]['examples'].values[0]

	# Definition

	definition = definitions_df[definitions_df['construct']==construct]['definition'].values[0]
	definition_references = definitions_df[definitions_df['construct']==construct]['definition_references'].values[0]
	lexicon.add(construct, section = 'definition', value = definition) # or lexicon.constructs[construct]['definition'] = definition
	lexicon.add(construct, section = 'definition_references', value = definition_references)
	lexicon.add(construct, section = 'prompt_name', value = prompt_name)
	lexicon.add(construct, section = 'examples', value = examples) # lexicon.constructs[construct]['examples'] = examples
	if construct not in ['active_suicidal_ideation', 'passive_suicidal_ideation', 'self-injury']:
		definition = False


	# GENERATE
	prompt = generate_prompt(construct, prompt_name=prompt_name, prompt="default", domain='mental health', definition=definition, examples = examples, output_format="default", remove_parentheses_definition = True)

	print('generating lexicon 1...')
	lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=0,seed=42,max_tokens = 150, top_p=0.9)
	print(len(np.unique(lexicon.constructs[construct]["tokens"])))

	print('generating lexicon 2...')
	lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=0.5, seed=42, max_tokens = 150, top_p=0.9)
	print(len(np.unique(lexicon.constructs[construct]["tokens"])))

	print('generating lexicon 3...')
	lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=1, seed=42, max_tokens = 150, top_p=0.9)
	print(len(np.unique(lexicon.constructs[construct]["tokens"])))

	#
	# print('generating lexicon 3...')
	# lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=1,  max_tokens = 150, top_p=0.9)
	# print(len(np.unique(lexicon.constructs[construct]["tokens"])))

	# print(lexicon.constructs[construct]["tokens"])


	# print('generating lexicon 3...')
	# lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=0.7, seed=42, max_tokens = 150)
	# print(lexicon.constructs[construct]["tokens"])
	# print(len(np.unique(lexicon.constructs[construct]["tokens"])))
	#
	#
	# print('generating lexicon 3...')
	# lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=1, seed=42, max_tokens = 150)
	# print(lexicon.constructs[construct]["tokens"])
	# print(len(np.unique(lexicon.constructs[construct]["tokens"])))



	# Save lexicon

	with open(output_dir+f'{name}_{model}_{ts}.json', "w") as fp:
		json.dump(lexicon.constructs, fp, indent=4)



	# Save the file
	dill.dump(lexicon, file = open(output_dir+f'{name}_{model}_{ts}.pickle', "wb"))
	# lexicon_reloaded = dill.load(open(output_dir+f'{name}_{model}_{ts}.pickle', "rb"))


	sleep_n = 5
	print(f'sleeping for {sleep_n} seconds...')
	time.sleep(sleep_n)



	# for n in ['prompt_name', 'definition', 'definition_references', 'examples']:





for construct in lexicon.constructs.keys():
	print()
	print(construct, len(lexicon.constructs[construct]['tokens']))
	for source, metadata in lexicon.constructs[construct]['tokens_metadata'].items():
		tokens = metadata['tokens']
		print(len(tokens), source)
		print(tokens)
		print()




for construct in constructs:
	print(construct)
	tokens = lexicon_reloaded.constructs[construct]['tokens']
	# tokens = [n.replace('i ', 'I ').replace("i'", "I'") for n in tokens]
	print(tokens)
	# lexicon.constructs[construct]['tokens'] = tokens

	# re-assign
	for source in lexicon.constructs[construct]['tokens_metadata'].keys():
		tokens = lexicon.constructs[construct]['tokens_metadata'][source]['tokens']
		# tokens = [n.replace('i ', 'I ').replace("i'", "I'") for n in tokens]
		print(tokens)
		# lexicon.constructs[construct]['tokens_metadata'][source]['tokens'] = tokens
# Save lexicon

with open(output_dir+f'{name}_{model}_{ts}.json', "w") as fp:
	json.dump(lexicon.constructs, fp, indent=4)
import pickle
save_object(lexicon, output_dir+f'{name}_{model}_{ts}.pickle')

import dill

# Save the file
dill.dump(lexicon, file = open(output_dir+f'{name}_{model}_{ts}.pickle', "wb"))
lexicon_reloaded = dill.load(open(output_dir+f'{name}_{model}_{ts}.pickle', "rb"))
lexicon == lexicon_reloaded



lexicon_df = []
for construct in constructs:
	df_i = pd.DataFrame(lexicon.constructs[construct]['tokens'], columns = [construct])
	df_i[construct+'_include'] = [np.nan]*df_i.shape[0]
	df_i[construct+'_add'] = [np.nan]*df_i.shape[0]
	lexicon_df.append(df_i)

lexicon_df = pd.concat(lexicon_df, axis=1)
lexicon_df.to_csv(output_dir+f'lexicon_df_{ts}.csv')


lexicon.constructs.keys()
lexicon_df = lexicon.to_pandas()



	# include, add
	# integrate dataset tokens
	# manually add
	# turn lower case into upper case I

# Compare sources
# construct = 'Loneliness & isolation'
#
# sources = lexicon.constructs[construct]['tokens_metadata'].keys()
# sources_command = [n for n in sources if 'command' in n]
# sources_gpt = [n for n in sources if 'command' not in n]
#
# gpt = lexicon.constructs[construct]['tokens_metadata'][sources_gpt[0]]['tokens'] + lexicon.constructs[construct]['tokens_metadata'][sources_gpt[1]]['tokens']
# command = lexicon.constructs[construct]['tokens_metadata'][sources_command[0]]['tokens'] + lexicon.constructs[construct]['tokens_metadata'][sources_command[1]]['tokens']
# set(command)-set(gpt)
# 'kill myself' in command
# examples  = definitions_df[definitions_df['construct']==construct]['examples'].values[0]
#
# [n for n in gpt if 'segr' in n]
# set(gpt)
# examples

	# ## Remove
	# remove_substrings = ["feelings of","feelings ", "feeling ", "emotional "]  # automate: order from longest to shortest, make sure you add space after because if not it make remove a subword (for "feelings of entrapment", "feeling" would result in "s of entrapment")
	#
	# tokens_less_than_5 = [n for n in lexicon.constructs[construct]["tokens"] if len(n) <5]
	# tokens_less_than_5


	# # Remove tokens that contain other tokens
	# lexicon.remove_tokens_containing_token(construct)
	# print(lexicon.constructs[construct]["tokens"])
	# print(len(lexicon.constructs[construct]["tokens"]))
	# lexicon.constructs[construct]
	#
	# tokens_less_than_5 = [n for n in lexicon.constructs[construct]["tokens"] if len(n) <5]
	# print('tokens less than 5 letters:', tokens_less_than_5)
	# remove_tokens = ['lone', 'maze']
	# lexicon.remove(construct, remove_tokens=None, remove_substrings=remove_substrings)
	#
	# Check words containing jobs.
	#
	#
	#
	# # TODO: remove if beginswith()
	# remove_substrings = ["feelings of","feelings ", "feeling ", "emotional "]  # automate: order from longest to shortest, make sure you add space after because if not it make remove a subword (for "feelings of entrapment", "feeling" would result in "s of entrapment")
	# #  Maybe: ["having "]


# OLD
# ============================================================


#
#
# # lexicon.set_prompt(construct, prompt_name = prompt_name, prompt="default", domain=domain, definition=definition, use_definition=use_definition, examples = examples, output_format="default")
# 	# print(lexicon.prompt)
# 	# lexicon.constructs
# 	# print(lexicon.prompt_name)
# 	# new_prompt = "Provide a list of words and phrases related to loneliness in the mental health domain. Each word or phrase should be separated by a semicolon. Do not provide any explanation or additional text beyond the word and phrases. Include loneliness in the list."
# 	# Lexicon.prompt = new_prompt
#
# 	# response = "debt; salary negotiation; burnout; job security; layoffs; inflation; retirement planning; credit score; office politics; performance review; mortgage; student loans; investment loss; healthcare costs; tax audits; budgeting; deadlines; promotion denial; economic recession; gig economy; financial fraud; unpaid overtime; childcare expenses; job hunting; underemployment; stock market crash; pension cuts; business failure; wage stagnation; rent increase; commuting costs; loan default; financial literacy; career change; workplace discrimination; cost of living; downsizing; bankruptcy; interest rates; savings depletion; contract disputes; market volatility; outsourcing; overwork; union strikes; profit loss; asset depreciation; expense tracking; wage gap; financial crisis; overtime pay; job satisfaction; insurance premiums; workplace safety; severance package; employee turnover; salary freeze; gig work insecurity; financial planning; wage garnishment; career stagnation; workplace stress; dividend cuts; credit card debt; job interview stress; workplace harassment; salary disparity; startup failure; retirement savings shortfall; workplace automation; trade tariffs; furloughs; economic uncertainty; employee benefits cut; property taxes; stock options; job competition; contract termination; financial scams; pension fund risk; workplace bullying; income inequality; project delays; expense reduction; job relocation; market competition; hiring freeze; venture capital loss; workplace culture issues; eviction notice; business regulation; inflation adjustment; economic sanctions; financial aid concerns; performance pressure; workplace conflict; salary benchmarking; business downturn; job market saturation; interest payment; market crash anxiety; layoff rumors; pension plan changes; cost-cutting measures; career development concerns; wage discrimination; rent arrears; stock portfolio loss; workplace diversity issues; tax increase; investment risk; wage theft; market downturn; job offer negotiations; promotion competition; labor market trends; expense mismanagement; financial emergency; workplace morale; debt collection; economic instability; rent control; employee rights; stock market speculation; wage negotiation; financial mismanagement; career burnout; job role change; investment fraud; workplace training costs; minimum wage debate; financial uncertainty; salary comparison; job application stress; mortgage refinancing; cost of living increase; performance bonus cut; tax evasion; business loan rejection; wage underpayment; retirement age increase; workplace ethics issues; financial accountability; job outsourcing; wage competition; economic policy changes; financial distress; job transfer; investment loss recovery; workplace efficiency; salary cut; business insolvency; financial goal setting; expense overruns; labor dispute; career progression; market analysis; salary benchmarking; investment portfolio management; job benefits; workplace innovation; economic growth concerns; business tax issues; job stress; wage disparity; salary negotiation failure; retirement planning stress; wage law changes; investment strategy; workplace wellness programs; bankruptcy proceedings; job market competition; financial management stress; wage gap analysis; expense fraud; career uncertainty; workplace injury; salary increment; job role elimination; economic forecast; financial missteps; workplace productivity; job search anxiety; stock market trends; wage law compliance; career opportunities; business competition; job security concerns; financial goal achievement; workplace flexibility; tax planning; salary review; investment decision stress; market speculation; wage negotiation strategies; job market uncertainty; financial solvency; career planning; wage policy; job satisfaction survey; workplace hierarchy; financial stress management; business expansion; salary increment dispute; economic downturn; job training costs; career advancement; wage cut; financial strategy; business growth; job burnout; wage increase; financial planning challenges; job skill development; workplace communication; tax filing; salary delay; investment portfolio adjustment; job responsibilities; workplace leadership; economic recovery; financial advice; business strategy; job role adjustment; wage adjustment; financial risk management; career goals; workplace relationships; tax deductions; salary gap; investment planning; job promotion; workplace efficiency improvement; economic analysis; financial decision making; business marketing; job stability; wage law changes; financial planning stress; career development; workplace policy; tax return; salary expectation; investment market trends; wage law interpretation; career transition; workplace rules; tax liability; salary scale; investment opportunities; job market trends; wage law understanding; career progression planning; workplace training; tax strategy; salary negotiation techniques; investment risk assessment; job performance; workplace safety standards; economic trends; financial portfolio; business planning; job role expectations; wage policy changes; financial planning advice; career development strategies; workplace culture; tax planning strategies; salary range; investment market analysis; wage law enforcement; career advancement opportunities; workplace strategy; tax compliance; salary increase; investment strategy planning; job promotion prospects; workplace safety policies; economic policy; financial portfolio management; business strategy planning; job role responsibilities; wage policy interpretation; financial risk assessment; career guidance; workplace regulations; tax obligations; salary structure; investment market fluctuations; wage law enforcement strategies; career advancement strategies; workplace strategy planning; tax planning advice; salary level; investment market dynamics; wage policy compliance; career development opportunities; workplace culture development; tax reduction strategies; salary negotiation skills; investment market prediction"
# 	# len(lexicon.clean_response(response))
# 	# len(np.unique(lexicon.clean_response(response)))
# 	lexicon.prompt
# 	try:
# 		print('generating lexicon 1...') # TODO add progress bar
# 		temperature = 0.5
# 		lexicon.create_construct(model=model, temperature=temperature, timeout = 60, num_retries = 2, seed = seed)
#
# 		lexicon.add(construct, 'definition', value = definition)
# 		# print(lexicon.constructs[construct])
# 		print(lexicon.constructs[construct]["tokens"])
# 		print(len(np.unique(lexicon.constructs[construct]["tokens"])))
#
#
# 		# print(len(np.unique(lexicon.constructs[construct]["tokens"])))
# 		# print(len(lexicon.constructs[construct]["tokens"]))
# 		#
# 		# unique_tokens = list(np.unique(lexicon.constructs[construct]["tokens"]))
# 		# lexicon.constructs[construct]["tokens"] = unique_tokens
# 		# lexicon.constructs[construct]["tokens_generated"] = len(unique_tokens)
#
#
# 		 # Add
# 		print('generating lexicon 2...') # TODO add progress bar
# 		temperature = 0.9
# 		lexicon.add(construct, "tokens", model=model, temperature=temperature,timeout = 60,examples=examples, definition=definition, use_definition=use_definition)
#
#
# 		# print(len(np.unique(lexicon.constructs[construct]["tokens"])))
#
# 		# lexicon.constructs[construct]["tokens"]
# 		# lexicon.constructs[construct]
# 		# Save
#
#
#
#
#
#
#
#
# 	except:
# 		print(f'\n\n\n\n\n\n\n\nWARNING: likely time out error for {construct} ===========\n\n\n\n\n')

#
# 	#  gpt-4
# 	model = "gpt-4"
# 	lexicon.add(construct, "tokens", model=model, temperature=0.1,timeout = 40)
# 	print(lexicon.constructs[construct])
# 	print(lexicon.constructs[construct]["tokens"])
# 	print(len(lexicon.constructs[construct]["tokens"]))
#
# 	lexicon.add(construct, "tokens", model=model, temperature=0.9,timeout = 40)
# 	print(lexicon.constructs[construct])
# 	print(lexicon.constructs[construct]["tokens"])
# 	print(len(lexicon.constructs[construct]["tokens"]))
#
#
"""















'''
- Clean GenAI response
- Obtain characteristic words from dataset using word scores
- Count
	- Lemmatize tokens
	- 
- Calibration of false positives and false negatives using a relevant dataset
- Validation

'''

# from utils import lemmatizer # local script
# # TODO: iness>'y' ness> ''


# Negative match check / Check documents without matches
# ================================================================================================




# For clinicians to evaluate
# ================================================================================

metadata_df = []
df_i = pd.DataFrame(['Prompt name', 'Definition', 'Definition references', 'Examples', 'Tokens'], columns = ['Construct'])
metadata_df.append(df_i)
for construct in lexicon.constructs.keys():
	df_i = pd.DataFrame([
		[lexicon.constructs[construct]['prompt_name']]+[lexicon.constructs[construct]['definition']]+[lexicon.constructs[construct]['definition_references']]+[lexicon.constructs[construct]['examples']]+lexicon.constructs[construct]['tokens']

	], index=[construct]).T
	metadata_df.append(df_i)
	df_i = pd.DataFrame([], columns = [f'{construct}_include', f'{construct}_add'])
	metadata_df.append(df_i)

metadata_df = pd.concat(metadata_df, axis=1, ignore_index=False)


metadata_df.to_csv(f'data/lexicons/suicide_risk_lexicon_annotate_clinicians_{ts}.csv')
metadata_df





import time
import json
import datetime
import api_keys  # this file should be in the same folder
import litellm
import time


input_dir = './data/'
output_dir = './data/lexicons/'

# construct = ['Suicidal constructs'] *3 + ['Interpersonal issues']*4 + ['Perception of self']* 3 + ['Affective'] * 5 + ['Substance use'] *2 + ['Externalizing']*2

# Generate lexicon
# ========================================================================
# load api keys

os.environ["OPENAI_API_KEY"] = api_keys.open_ai  # string, you need your own key and to put at least $5 in the account
# os.environ["COHERE_API_KEY"] = api_keys.cohere_trial  # string, you can get for free by logging into cohere and going to sandbox
# # pip install -q google-generativeai
# os.environ["GEMINI_API_KEY"] = api_keys.gemini  # here: https://makersuite.google.com/app/apikey
# os.environ["VERTEXAI_PROJECT"] = api_keys.vertex


litellm.set_verbose=False
litellm.drop_params=True

# Config
# ==============================
# domain = None
domain = "mental health"
# model = "command-nightly"
model = "gpt-4-1106-preview"
# model = 'gemini-pro'
seed = 42
# constructs = constructs[39:]
definitions_df = pd.read_csv('./data/lexicons/suicide_risk_constructs_and_definitions.csv', encoding='latin1')



constructs = definitions_df['construct'].values

definitions_df.columns
name = 'suicide_risk_lexicon'

# Initialize lexicon
lexicon = Lexicon()
lexicon.name = 'Suicide Risk Lexicon'
lexicon.description = 'Lexicon for 46 risk factors validated by clinical experts. If you use, please cite publication: Low et al (in prep.). Creating a Suicide Risk Lexicon with Generative AI and word scores.'

import dill
ts = datetime.datetime.utcnow().strftime('%y-%m-%dT%H-%M-%S') # so you don't overwrite, and save timestamp
for i, construct in enumerate(constructs):
	print(i, construct)
# for construct in ['Active suicidal ideation', 'Loneliness & isolation']:# 'Self-injury', 'Sexual abuse & harassment']:#, 'Anxiety', 'Gender & sexual identity']:
	# construct = "sexual_abuse_harassment"

	prompt_name  = definitions_df[definitions_df['construct']==construct]['prompt_name'].values[0]
	examples  = definitions_df[definitions_df['construct']==construct]['examples'].values[0]

	# Definition

	definition = definitions_df[definitions_df['construct']==construct]['definition'].values[0]
	definition_references = definitions_df[definitions_df['construct']==construct]['definition_references'].values[0]
	lexicon.add(construct, section = 'definition', value = definition) # or lexicon.constructs[construct]['definition'] = definition
	lexicon.add(construct, section = 'definition_references', value = definition_references)
	lexicon.add(construct, section = 'prompt_name', value = prompt_name)
	lexicon.add(construct, section = 'examples', value = examples) # lexicon.constructs[construct]['examples'] = examples
	if construct not in ['active_suicidal_ideation', 'passive_suicidal_ideation', 'self-injury']:
		definition = False


	# GENERATE
	prompt = generate_prompt(construct, prompt_name=prompt_name, prompt="default", domain='mental health', definition=definition, examples = examples, output_format="default", remove_parentheses_definition = True)

	print('generating lexicon 1...')
	lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=0,seed=42,max_tokens = 150, top_p=0.9)
	print(len(np.unique(lexicon.constructs[construct]["tokens"])))

	print('generating lexicon 2...')
	lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=0.5, seed=42, max_tokens = 150, top_p=0.9)
	print(len(np.unique(lexicon.constructs[construct]["tokens"])))

	print('generating lexicon 3...')
	lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=1, seed=42, max_tokens = 150, top_p=0.9)
	print(len(np.unique(lexicon.constructs[construct]["tokens"])))

	#
	# print('generating lexicon 3...')
	# lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=1,  max_tokens = 150, top_p=0.9)
	# print(len(np.unique(lexicon.constructs[construct]["tokens"])))

	# print(lexicon.constructs[construct]["tokens"])


	# print('generating lexicon 3...')
	# lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=0.7, seed=42, max_tokens = 150)
	# print(lexicon.constructs[construct]["tokens"])
	# print(len(np.unique(lexicon.constructs[construct]["tokens"])))
	#
	#
	# print('generating lexicon 3...')
	# lexicon.add(construct, value = 'create', prompt = prompt, source = model, temperature=1, seed=42, max_tokens = 150)
	# print(lexicon.constructs[construct]["tokens"])
	# print(len(np.unique(lexicon.constructs[construct]["tokens"])))



	# Save lexicon

	with open(output_dir+f'{name}_{model}_{ts}.json', "w") as fp:
		json.dump(lexicon.constructs, fp, indent=4)



	# Save the file
	dill.dump(lexicon, file = open(output_dir+f'{name}_{model}_{ts}.pickle', "wb"))
	# lexicon_reloaded = dill.load(open(output_dir+f'{name}_{model}_{ts}.pickle', "rb"))


	sleep_n = 5
	print(f'sleeping for {sleep_n} seconds...')
	time.sleep(sleep_n)



	# for n in ['prompt_name', 'definition', 'definition_references', 'examples']:





for construct in lexicon.constructs.keys():
	print()
	print(construct, len(lexicon.constructs[construct]['tokens']))
	for source, metadata in lexicon.constructs[construct]['tokens_metadata'].items():
		tokens = metadata['tokens']
		print(len(tokens), source)
		print(tokens)
		print()




for construct in constructs:
	print(construct)
	tokens = lexicon_reloaded.constructs[construct]['tokens']
	# tokens = [n.replace('i ', 'I ').replace("i'", "I'") for n in tokens]
	print(tokens)
	# lexicon.constructs[construct]['tokens'] = tokens

	# re-assign
	for source in lexicon.constructs[construct]['tokens_metadata'].keys():
		tokens = lexicon.constructs[construct]['tokens_metadata'][source]['tokens']
		# tokens = [n.replace('i ', 'I ').replace("i'", "I'") for n in tokens]
		print(tokens)
		# lexicon.constructs[construct]['tokens_metadata'][source]['tokens'] = tokens
# Save lexicon

with open(output_dir+f'{name}_{model}_{ts}.json', "w") as fp:
	json.dump(lexicon.constructs, fp, indent=4)
import pickle
save_object(lexicon, output_dir+f'{name}_{model}_{ts}.pickle')

import dill

# Save the file
dill.dump(lexicon, file = open(output_dir+f'{name}_{model}_{ts}.pickle', "wb"))
lexicon_reloaded = dill.load(open(output_dir+f'{name}_{model}_{ts}.pickle', "rb"))
lexicon == lexicon_reloaded



lexicon_df = []
for construct in constructs:
	df_i = pd.DataFrame(lexicon.constructs[construct]['tokens'], columns = [construct])
	df_i[construct+'_include'] = [np.nan]*df_i.shape[0]
	df_i[construct+'_add'] = [np.nan]*df_i.shape[0]
	lexicon_df.append(df_i)

lexicon_df = pd.concat(lexicon_df, axis=1)
lexicon_df.to_csv(output_dir+f'lexicon_df_{ts}.csv')


lexicon.constructs.keys()
lexicon_df = lexicon.to_pandas()


"""
	# include, add
	# integrate dataset tokens
	# manually add
	# turn lower case into upper case I

# Compare sources
# construct = 'Loneliness & isolation'
#
# sources = lexicon.constructs[construct]['tokens_metadata'].keys()
# sources_command = [n for n in sources if 'command' in n]
# sources_gpt = [n for n in sources if 'command' not in n]
#
# gpt = lexicon.constructs[construct]['tokens_metadata'][sources_gpt[0]]['tokens'] + lexicon.constructs[construct]['tokens_metadata'][sources_gpt[1]]['tokens']
# command = lexicon.constructs[construct]['tokens_metadata'][sources_command[0]]['tokens'] + lexicon.constructs[construct]['tokens_metadata'][sources_command[1]]['tokens']
# set(command)-set(gpt)
# 'kill myself' in command
# examples  = definitions_df[definitions_df['construct']==construct]['examples'].values[0]
#
# [n for n in gpt if 'segr' in n]
# set(gpt)
# examples

	# ## Remove
	# remove_substrings = ["feelings of","feelings ", "feeling ", "emotional "]  # automate: order from longest to shortest, make sure you add space after because if not it make remove a subword (for "feelings of entrapment", "feeling" would result in "s of entrapment")
	#
	# tokens_less_than_5 = [n for n in lexicon.constructs[construct]["tokens"] if len(n) <5]
	# tokens_less_than_5


	# # Remove tokens that contain other tokens
	# lexicon.remove_tokens_containing_token(construct)
	# print(lexicon.constructs[construct]["tokens"])
	# print(len(lexicon.constructs[construct]["tokens"]))
	# lexicon.constructs[construct]
	#
	# tokens_less_than_5 = [n for n in lexicon.constructs[construct]["tokens"] if len(n) <5]
	# print('tokens less than 5 letters:', tokens_less_than_5)
	# remove_tokens = ['lone', 'maze']
	# lexicon.remove(construct, remove_tokens=None, remove_substrings=remove_substrings)
	#
	# Check words containing jobs.
	#
	#
	#
	# # TODO: remove if beginswith()
	# remove_substrings = ["feelings of","feelings ", "feeling ", "emotional "]  # automate: order from longest to shortest, make sure you add space after because if not it make remove a subword (for "feelings of entrapment", "feeling" would result in "s of entrapment")
	# #  Maybe: ["having "]


# OLD
# ============================================================


#
#
# # lexicon.set_prompt(construct, prompt_name = prompt_name, prompt="default", domain=domain, definition=definition, use_definition=use_definition, examples = examples, output_format="default")
# 	# print(lexicon.prompt)
# 	# lexicon.constructs
# 	# print(lexicon.prompt_name)
# 	# new_prompt = "Provide a list of words and phrases related to loneliness in the mental health domain. Each word or phrase should be separated by a semicolon. Do not provide any explanation or additional text beyond the word and phrases. Include loneliness in the list."
# 	# Lexicon.prompt = new_prompt
#
# 	# response = "debt; salary negotiation; burnout; job security; layoffs; inflation; retirement planning; credit score; office politics; performance review; mortgage; student loans; investment loss; healthcare costs; tax audits; budgeting; deadlines; promotion denial; economic recession; gig economy; financial fraud; unpaid overtime; childcare expenses; job hunting; underemployment; stock market crash; pension cuts; business failure; wage stagnation; rent increase; commuting costs; loan default; financial literacy; career change; workplace discrimination; cost of living; downsizing; bankruptcy; interest rates; savings depletion; contract disputes; market volatility; outsourcing; overwork; union strikes; profit loss; asset depreciation; expense tracking; wage gap; financial crisis; overtime pay; job satisfaction; insurance premiums; workplace safety; severance package; employee turnover; salary freeze; gig work insecurity; financial planning; wage garnishment; career stagnation; workplace stress; dividend cuts; credit card debt; job interview stress; workplace harassment; salary disparity; startup failure; retirement savings shortfall; workplace automation; trade tariffs; furloughs; economic uncertainty; employee benefits cut; property taxes; stock options; job competition; contract termination; financial scams; pension fund risk; workplace bullying; income inequality; project delays; expense reduction; job relocation; market competition; hiring freeze; venture capital loss; workplace culture issues; eviction notice; business regulation; inflation adjustment; economic sanctions; financial aid concerns; performance pressure; workplace conflict; salary benchmarking; business downturn; job market saturation; interest payment; market crash anxiety; layoff rumors; pension plan changes; cost-cutting measures; career development concerns; wage discrimination; rent arrears; stock portfolio loss; workplace diversity issues; tax increase; investment risk; wage theft; market downturn; job offer negotiations; promotion competition; labor market trends; expense mismanagement; financial emergency; workplace morale; debt collection; economic instability; rent control; employee rights; stock market speculation; wage negotiation; financial mismanagement; career burnout; job role change; investment fraud; workplace training costs; minimum wage debate; financial uncertainty; salary comparison; job application stress; mortgage refinancing; cost of living increase; performance bonus cut; tax evasion; business loan rejection; wage underpayment; retirement age increase; workplace ethics issues; financial accountability; job outsourcing; wage competition; economic policy changes; financial distress; job transfer; investment loss recovery; workplace efficiency; salary cut; business insolvency; financial goal setting; expense overruns; labor dispute; career progression; market analysis; salary benchmarking; investment portfolio management; job benefits; workplace innovation; economic growth concerns; business tax issues; job stress; wage disparity; salary negotiation failure; retirement planning stress; wage law changes; investment strategy; workplace wellness programs; bankruptcy proceedings; job market competition; financial management stress; wage gap analysis; expense fraud; career uncertainty; workplace injury; salary increment; job role elimination; economic forecast; financial missteps; workplace productivity; job search anxiety; stock market trends; wage law compliance; career opportunities; business competition; job security concerns; financial goal achievement; workplace flexibility; tax planning; salary review; investment decision stress; market speculation; wage negotiation strategies; job market uncertainty; financial solvency; career planning; wage policy; job satisfaction survey; workplace hierarchy; financial stress management; business expansion; salary increment dispute; economic downturn; job training costs; career advancement; wage cut; financial strategy; business growth; job burnout; wage increase; financial planning challenges; job skill development; workplace communication; tax filing; salary delay; investment portfolio adjustment; job responsibilities; workplace leadership; economic recovery; financial advice; business strategy; job role adjustment; wage adjustment; financial risk management; career goals; workplace relationships; tax deductions; salary gap; investment planning; job promotion; workplace efficiency improvement; economic analysis; financial decision making; business marketing; job stability; wage law changes; financial planning stress; career development; workplace policy; tax return; salary expectation; investment market trends; wage law interpretation; career transition; workplace rules; tax liability; salary scale; investment opportunities; job market trends; wage law understanding; career progression planning; workplace training; tax strategy; salary negotiation techniques; investment risk assessment; job performance; workplace safety standards; economic trends; financial portfolio; business planning; job role expectations; wage policy changes; financial planning advice; career development strategies; workplace culture; tax planning strategies; salary range; investment market analysis; wage law enforcement; career advancement opportunities; workplace strategy; tax compliance; salary increase; investment strategy planning; job promotion prospects; workplace safety policies; economic policy; financial portfolio management; business strategy planning; job role responsibilities; wage policy interpretation; financial risk assessment; career guidance; workplace regulations; tax obligations; salary structure; investment market fluctuations; wage law enforcement strategies; career advancement strategies; workplace strategy planning; tax planning advice; salary level; investment market dynamics; wage policy compliance; career development opportunities; workplace culture development; tax reduction strategies; salary negotiation skills; investment market prediction"
# 	# len(lexicon.clean_response(response))
# 	# len(np.unique(lexicon.clean_response(response)))
# 	lexicon.prompt
# 	try:
# 		print('generating lexicon 1...') # TODO add progress bar
# 		temperature = 0.5
# 		lexicon.create_construct(model=model, temperature=temperature, timeout = 60, num_retries = 2, seed = seed)
#
# 		lexicon.add(construct, 'definition', value = definition)
# 		# print(lexicon.constructs[construct])
# 		print(lexicon.constructs[construct]["tokens"])
# 		print(len(np.unique(lexicon.constructs[construct]["tokens"])))
#
#
# 		# print(len(np.unique(lexicon.constructs[construct]["tokens"])))
# 		# print(len(lexicon.constructs[construct]["tokens"]))
# 		#
# 		# unique_tokens = list(np.unique(lexicon.constructs[construct]["tokens"]))
# 		# lexicon.constructs[construct]["tokens"] = unique_tokens
# 		# lexicon.constructs[construct]["tokens_generated"] = len(unique_tokens)
#
#
# 		 # Add
# 		print('generating lexicon 2...') # TODO add progress bar
# 		temperature = 0.9
# 		lexicon.add(construct, "tokens", model=model, temperature=temperature,timeout = 60,examples=examples, definition=definition, use_definition=use_definition)
#
#
# 		# print(len(np.unique(lexicon.constructs[construct]["tokens"])))
#
# 		# lexicon.constructs[construct]["tokens"]
# 		# lexicon.constructs[construct]
# 		# Save
#
#
#
#
#
#
#
#
# 	except:
# 		print(f'\n\n\n\n\n\n\n\nWARNING: likely time out error for {construct} ===========\n\n\n\n\n')

#
# 	#  gpt-4
# 	model = "gpt-4"
# 	lexicon.add(construct, "tokens", model=model, temperature=0.1,timeout = 40)
# 	print(lexicon.constructs[construct])
# 	print(lexicon.constructs[construct]["tokens"])
# 	print(len(lexicon.constructs[construct]["tokens"]))
#
# 	lexicon.add(construct, "tokens", model=model, temperature=0.9,timeout = 40)
# 	print(lexicon.constructs[construct])
# 	print(lexicon.constructs[construct]["tokens"])
# 	print(len(lexicon.constructs[construct]["tokens"]))
#
#
#
"""