import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from preprocess import clean_text, stop
from scipy.stats import chi2_contingency
# nltk.download('stopwords')


RACIAL_KW = ['chinese', 'chinese-american', 'american-born chinese', 'cuban', 'cuban-american', 'dominican', \
    'dominican-american', 'salvadoran', 'salvadoran-american', 'guatemalan', 'guatemalan-american', 'indian', \
    'indian-american', 'mexican', 'mexican-american', 'chicana', 'chicano', 'filipina', 'filipina-american', \
    'filipino', 'filipino-american', 'korean', 'korean-american', 'vietnamese', 'vietnamese-american', 'asian', 'asian-american', 'desi', 'east asian', 'oriental', 'south asian', 'southeast asian', 'african', 'african-american', 'black', 'hispanic', 'latinx', 'latine', 'latina', 'latino', 'latin american', 'pacific islander', 'aapi', 'bipoc', 'person of color', 'people of color', 'man of color', 'men of color', 'woman of color', 'women of color', 'ashkenazi jewish', "bahá'í", 'buddhist', 'cheondoist', 'confucianist', 'conservative jewish', 'druze', 'hasidic', 'hindu', 'jain', 'jewish', 'muslim', 'orthodox jewish', 'rasta', 'rastafari', 'rastafarian', 'reform jewish', 'sephardic jewish', 'shia', 'shintoist', 'sikh', 'sunni', 'taoist', 'zoroastrian', 'jewish american princess', 'jewish american princesses', 'jap', 'japs', 'dreadlocked', 'curly-haired', 'frizzy-haired', 'coily-haired', 'afro', 'afros', 'jewfro', 'jewfros', 'brown-skinned', 'dark-skinned', 'olive-skinned', 'yellow', 'asylum seeker', 'asylum seekers', 'refugee', 'refugees', 'immigrant', 'immigrants', 'daca', 'dreamer', 'dreamers']

HATE_SPEECH_KW = [
    "asian drive", "islam terrorism", "arab terror", "jsil", "racecard", "race card", "refugeesnotwelcome",\
    "DeportallMuslims", "banislam", "banmuslims", "destroyislam", "norefugees", "nomuslims", "border jumper", \
    "border nigger", "boojie", "chinaman", "whigger", "white nigger", "wigger", "wigerette", "bitter clinger", \
    "nigger", "coonass", "raghead", "house nigger", "white nigger", "camel lover", "moon cricket", "wetback" "bamboo coon", \
    "camel lover", "chinaman", "whigger", "white nigger", "nigga", "wigger", "zionazi", "camel lover", "zionazi", \
    "#BuildTheWall", "#DeportThemALL", "#RefugeesNOTwelcome", "#BanSharia", "#BanIslam"]

RACIAL_IND = RACIAL_KW + HATE_SPEECH_KW

def protected_attributes(text, racial_indicators=RACIAL_IND):
    tweet_racial_indicators = [indicator for indicator in racial_indicators if indicator in text]

    if tweet_racial_indicators:
        return 1
    else:
        return 0

#0. Read data
df = pd.DataFrame('labeled_data.csv')

#1. Select label, text columns
label_col = 'class'
text_col = 'tweet'

# Replace original column names with standardized names
df.rename({label_col: 'Label', text_col: 'Text'}, axis=1, inplace=True)

#2. Extract racial predictor
#2.1. Clean text
df['clean_text'] = df['Text'].apply(lambda x: stop(clean_text(x).replace('rt', '')))
df['protected_attribute'] = df.clean_tweet.apply(lambda tweet: protected_attributes(tweet))


############################### BIAS MITIGATION STARTS HERE ##################################
from bias_detection import *
from pulse import *

previous_metrics = pd.read_csv('metrics.csv')


# Calculate AUC bias metrics before mitigation
# metrics = pd.DataFrame()
# 
# overall_auc, subgroup_auc, bpsn_auc, bnsp_auc, privileged_group_acc, subgroup_accuracy, acc_delta = detect_bias_subgroup(df, 'clean_text', 'protected_attribute', multi_class=True)
# row = {
#     'subgroup': df['protected_attribute'].unique()[1:].copy(),
#     'bspn_auc': bnsp_auc,
#     'bpsn_auc': bpsn_auc,
#     'subgroup_auc': subgroup_auc,
#     'method': 'Original'
# }
# metrics = pd.concat([metrics, row], axis=0)

# Calculate AUC bias metrics after mitigation
# Option 1: Bias weights
# overall_auc, subgroup_auc, bpsn_auc, bnsp_auc = correct_bias_subgroup_weights(df, 'clean_text', 'protected_attribute', multi_class=True)
# row = {
#     'subgroup': df['protected_attribute'].unique()[1:].copy(),
#     'bspn_auc': bnsp_auc,
#     'bpsn_auc': bpsn_auc,
#     'subgroup_auc': subgroup_auc,
#     'method': 'Correcting Bias Weights'
# }
# metrics = pd.concat([metrics, row], axis=0)

# Option 2: Adversarial debugging
# This algorithm takes too long to run so I'm replacing with hard-coded values 
# overall_auc, subgroup_auc, bpsn_auc, bnsp_auc = correct_bias_subgroup_ad(df, 'clean_text', 'protected_attribute', multi_class=True)
# row = {
#     'subgroup': df['protected_attribute'].unique()[1:].copy(),
#     'bspn_auc': bnsp_auc,
#     'bpsn_auc': bpsn_auc,
#     'subgroup_auc': subgroup_auc,
#     'method': 'Adversarial Debiasing'
# }
# metrics = pd.concat([metrics, row], axis=0)




### Explainers

'''
Subgroup AUC: Restrict the test set to examples that mention the identity. A low value means that the model does poorly at distinguishing abusive and non-abusive examples that mention this identity.
BPSN (Background Positive, Subgroup Negative) AUC: Restrict the test set to non-abusive examples that mention the identity and abusive examples that do not. A low value suggests that the model’s scores skew higher than they should for examples mentioning this identity.
BNSP (Background Negative, Subgroup Positive) AUC: Restrict the test set to abusive examples that mention the identity and non-abusive examples that do not. A low value suggests that the model’s scores skew lower than they should for examples mentioning this identity.
'''