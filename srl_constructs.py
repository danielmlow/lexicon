
constructs_in_order = [
	"Passive suicidal ideation",
	"Active suicidal ideation & suicidal planning",
	"Lethal means for suicide",
	"Direct self-injury",
	"Suicide exposure",
	"Other suicidal language",
	'Hospitalization',


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
	"Entrapment & desire to escape",
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

categories = {
    'Suicidal constructs':[
        'Passive suicidal ideation',
        'Active suicidal ideation & suicidal planning',
        'Lethal means for suicide',
        'Direct self-injury',
        'Suicide exposure',
        'Other suicidal language',
        'Entrapment & desire to escape',
		'Hospitalization',
    ],
    'Negative perception of self':[
        'Burdensomeness',
        'Defeat & feeling like a failure',
        'Existential meaninglessness & purposelessness',
        'Shame, self-disgust, & worthlessness',
        'Guilt',
    ],
    'Depressive symptoms':[
        'Depressed mood',
        'Anhedonia & uninterested',
        'Emotional pain & psychache',
        'Grief & bereavement',
        'Emptiness',
        'Hopelessness',
        'Fatigue & tired',
        ],
    
    'Anxious symptoms':[
        
        'Anxiety',
        'Panic',
        'Trauma & PTSD',
        'Agitation',
        'Rumination',
        'Perfectionism',
    ],
    "Interpersonal":[
        'Loneliness & isolation',
        'Social withdrawal',
        'Relationship issues',
        'Relationships & kinship',
        'Bullying',
        'Sexual abuse & harassment',
        'Physical abuse & violence',
    ],
    "Externalizing":[
        'Aggression & irritability',
        'Impulsivity',
        'Alcohol use',
        'Other substance use',
    ],
    "Other disorders": [
        'Psychosis & schizophrenia',
        'Bipolar Disorder',
        'Borderline Personality Disorder',
        'Eating disorders',
        'Sleep issues',
],
    "Social and other determinants":[
        'Physical health issues & disability',
        'Incarceration',
        'Poverty & homelessness',
        'Gender & sexual identity',
        'Discrimination',
        'Finances & work stress',
        'Barriers to treatment',
        'Mental health treatment'
]    
}


# colors = dict(zip(categories.keys(), ['']*len(categories.keys())))

colors_barplot = {'Suicidal constructs': 'orangered',
 'Negative perception of self': 'deepskyblue',#'lightblue',
 'Depressive symptoms': 'dodgerblue',
 'Anxious symptoms': 'springgreen', #'palegreen',
 'Interpersonal': 'mediumslateblue',
 'Externalizing': 'purple',
 'Other disorders': 'violet',
 'Social and other determinants': 'crimson'}


colors = {'Suicidal constructs': 'orangered',
 'Negative perception of self': 'lightblue',
 'Depressive symptoms': 'dodgerblue',
 'Anxious symptoms': 'palegreen',
 'Interpersonal': 'mediumslateblue',
 'Externalizing': 'purple',
 'Other disorders': 'violet',
 'Social and other determinants': 'crimson'}


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


# order for each annotator
constructs_importance = [
	"Passive suicidal ideation",
	"Active suicidal ideation & suicidal planning",
    "Other suicidal language",
	"Direct self-injury",
	"Lethal means for suicide",
	"Suicide exposure",
    'Hospitalization',

	"Depressed mood",
	"Anhedonia & uninterested",
	"Emotional pain & psychache",
	"Grief & bereavement",
	"Fatigue & tired",
	"Emptiness",
	"Hopelessness",
    "Relationship issues",
	"Aggression & irritability",
	"Physical abuse & violence",	
	"Sexual abuse & harassment",
	"Anxiety",
	"Trauma & PTSD",    
    "Mental health treatment",
    
	"Guilt",
	"Borderline Personality Disorder",
	"Panic",
	"Agitation",
	"Entrapment & desire to escape",
    "Burdensomeness",
	"Rumination",
	"Shame, self-disgust, & worthlessness",
	"Loneliness & isolation",
	"Bipolar Disorder",
	"Psychosis & schizophrenia",
    "Eating disorders",
	
    "Impulsivity",
	"Existential meaninglessness & purposelessness",
	"Bullying",
    "Sleep issues",
	"Defeat & feeling like a failure",
	"Perfectionism",
    "Social withdrawal",
    "Barriers to treatment",
    
	"Discrimination",
	"Physical health issues & disability",	
	"Incarceration",
    "Poverty & homelessness",
    "Gender & sexual identity",
    "Finances & work stress",
    "Alcohol use",
	"Other substance use",
    "Relationships & kinship",
    
]


# order for each annotator
constructs_kb = [
	"Passive suicidal ideation",
	"Active suicidal ideation & suicidal planning",
    "Other suicidal language",
	"Direct self-injury",
	"Lethal means for suicide",
	"Suicide exposure",
    'Hospitalization',
    
	"Anxiety",
    "Panic",
	"Agitation",
	"Trauma & PTSD",    
    "Mental health treatment",

	"Depressed mood",
	"Anhedonia & uninterested",
	"Emotional pain & psychache",
	"Grief & bereavement",
	"Fatigue & tired",
	"Emptiness",
	"Hopelessness",
    "Relationship issues",
	"Aggression & irritability",
	"Physical abuse & violence",	
	"Sexual abuse & harassment",
    
	"Borderline Personality Disorder",
    "Shame, self-disgust, & worthlessness",
    "Guilt",
	"Entrapment & desire to escape",
    "Burdensomeness",
	"Rumination",
	
	"Loneliness & isolation",
	"Bipolar Disorder",
	"Psychosis & schizophrenia",
    "Eating disorders",
	
    "Impulsivity",
	"Existential meaninglessness & purposelessness",
	"Bullying",
    "Sleep issues",
	"Defeat & feeling like a failure",
	"Perfectionism",
    "Social withdrawal",
    "Barriers to treatment",
    
	"Discrimination",
	"Physical health issues & disability",	
	"Incarceration",
    "Poverty & homelessness",
    "Gender & sexual identity",
    "Finances & work stress",
    "Alcohol use",
	"Other substance use",
    "Relationships & kinship",
    
]

constructs_or = [
	"Passive suicidal ideation",
	"Active suicidal ideation & suicidal planning",
    "Other suicidal language",
	"Direct self-injury",
	"Lethal means for suicide",
	"Suicide exposure",
    'Hospitalization',
    
	"Depressed mood",
	"Anhedonia & uninterested",
	"Emotional pain & psychache",
	"Grief & bereavement",
	"Fatigue & tired",
	"Emptiness",
	"Hopelessness",
    "Relationship issues",
	"Aggression & irritability",
	"Physical abuse & violence",	
	"Sexual abuse & harassment",
    
	"Anxiety",
    "Panic",
	"Agitation",
	"Trauma & PTSD",    
    "Mental health treatment",
    
	"Borderline Personality Disorder",
    "Shame, self-disgust, & worthlessness",
    "Guilt",
	"Entrapment & desire to escape",
    "Burdensomeness",
	"Rumination",
	"Loneliness & isolation",
	"Bipolar Disorder",
	"Psychosis & schizophrenia",
    "Eating disorders",
	
    "Impulsivity",
	"Existential meaninglessness & purposelessness",
	"Bullying",
    "Sleep issues",
	"Defeat & feeling like a failure",
	"Perfectionism",
    "Social withdrawal",
    "Barriers to treatment",
    
	"Discrimination",
	"Physical health issues & disability",	
	"Incarceration",
    "Poverty & homelessness",
    "Gender & sexual identity",
    "Finances & work stress",
    "Alcohol use",
	"Other substance use",
    "Relationships & kinship",
    
]



constructs_dc = [
	"Passive suicidal ideation",
	"Active suicidal ideation & suicidal planning",
    "Other suicidal language",
	"Direct self-injury",
	"Lethal means for suicide",
	"Suicide exposure",
    'Hospitalization',
    
	"Borderline Personality Disorder",
    "Shame, self-disgust, & worthlessness",
    "Guilt",
	"Entrapment & desire to escape",
    "Burdensomeness",
	"Rumination",
	"Loneliness & isolation",
	"Bipolar Disorder",
	"Psychosis & schizophrenia",
    "Eating disorders",
    
	"Depressed mood",
	"Anhedonia & uninterested",
	"Emotional pain & psychache",
	"Grief & bereavement",
	"Fatigue & tired",
	"Emptiness",
	"Hopelessness",

    "Relationship issues",
	"Aggression & irritability",
	"Physical abuse & violence",	
	"Sexual abuse & harassment",
    
	"Anxiety",
    "Panic",
	"Agitation",
	"Trauma & PTSD",    
    "Mental health treatment",
    
    
	"Impulsivity",
	"Existential meaninglessness & purposelessness",
	"Bullying",
    "Sleep issues",
	"Defeat & feeling like a failure",
	"Perfectionism",
    "Social withdrawal",
    "Barriers to treatment",
    
	"Discrimination",
	"Physical health issues & disability",	
	"Incarceration",
    "Poverty & homelessness",
    "Gender & sexual identity",
    "Finances & work stress",
    "Alcohol use",
	"Other substance use",
    "Relationships & kinship",
    
]




