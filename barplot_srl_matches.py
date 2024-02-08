import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('/Users/danielmlow/Dropbox (MIT)/datum/concept-tracker/') # TODO remove
from concept_tracker import lexicon
from concept_tracker.lexicon import * # TODO remove
from concept_tracker.utils import lemmatizer
from srl_constructs import constructs_in_order, categories, colors, colors_list


# load lexicon
# ================================================================================================
timestamp = ''
srl = load_lexicon(f'./data/lexicons/suicide_risk_lexicon_gpt-4-1106-preview_dml_{timestamp}.pickle')

# load training set
with open('./data/input/ctl/ctl_dfs_features.pkl', 'rb') as f:
	dfs = pickle.load(f)

train_df = dfs['train']['df_text'][['text', 'y']]
docs = train_df['text'].tolist()

# Lemmatize and extract
# ================================================================================================
srl = lexicon.lemmatize_tokens(srl) # TODO: integrate this to class: self.lemmatize_tokens() adds tokens_lemmatized

# Extract on l1_docs, l2_docs, l3_docs
feature_vectors, matches_counter_d, matches_per_doc, matches_per_construct  = lexicon.extract(train_df['text'].tolist(),
																					srl.constructs,normalize = False, return_matches=True,
																					add_lemmatized_lexicon=True, lemmatize_docs=False,
																					exact_match_n = srl.exact_match_n,exact_match_tokens = srl.exact_match_tokens)




# Plot counts
# ================================================================================================
colors_list = []
constructs = []
for category in categories.keys():
    category_constructs = categories.get(category)
    constructs.extend(category_constructs)

for construct in constructs:
    for category in categories.keys():
        category_constructs = categories.get(category)
        if construct in category_constructs:
            colors_list.append(category)

colors_list = [colors.get(n) for n in colors_list]


# Counts
train_df_features = pd.concat([train_df, feature_vectors],axis=1)
train_df_features.groupby('y').sum().bar(stacked=True)


train_df_features_binary = train_df_features.copy()
train_df_features_binary[constructs] = train_df_features[constructs].values>0
proportion = True
if proportion:
	counts = train_df_features_binary.groupby('y').sum()
	
else:
	counts = train_df_features.groupby('y').sum()
	
counts.to_csv(f'./data/output/tables/srl_matches_count_proportion-{proportion}.csv', index = False)
counts = counts.reset_index()
counts['Severity'] = counts['y'].map({4: 'Very high', 3: 'High', 2: 'Medium'})

counts = counts.drop(['text', 'word_count', 'y'],axis=1)
counts = counts[['Severity']+constructs]

melted_df = pd.melt(counts, id_vars=['Severity'], var_name='Construct', value_name='Counts')

if proportion:
	melted_df['Proportion'] = melted_df['Counts']/train_df.shape[0]/3

		

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  # Width=10 inches, Height=6 inches

custom_palette = {'Medium': 'gold', 'High': 'orangered', 'Very high':'firebrick'}
if proportion:
	sns.barplot(x='Construct', y='Proportion', hue='Severity', data=melted_df, palette = custom_palette, alpha = 0.8)
	plt.ylabel(f"Proportion of conversations with\nat least one token match")

else:
	sns.barplot(x='Construct', y='Counts', hue='Severity', data=melted_df, palette = custom_palette, alpha = 0.8)
	plt.ylabel(f"Amount of token matches")

plt.xticks(rotation=90, fontsize=10)
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,1, 0]  # This will reverse the order of the legend items

# Step 3: Create a new legend with the specified order
leg = plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

for tick_label, color in zip(plt.gca().get_xticklabels(), colors_list):
    tick_label.set_color(color)

plt.tight_layout()
plt.savefig(f'./data/output/figures/distributions_srl_constructs_ctl_by_severity_proportion-{proportion}.png', bbox_inches='tight', dpi=300)


