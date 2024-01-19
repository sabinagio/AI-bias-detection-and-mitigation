import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk
from preprocess import clean_text, stop
from scipy.stats import chi2_contingency
#nltk.download('stopwords')


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
#df = pd.DataFrame('labeled_data.csv')
df = pd.read_csv('labeled_data.csv')
# print(df['class'].value_counts())
# labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
# df['class'] = df['class'].map(labels)
# df.to_csv('labeled_data.csv', index=False)


#1. Select label, text columns
label_col = 'class'
text_col = 'tweet'

# Replace original column names with standardized names
df.rename({label_col: 'Label', text_col: 'Text'}, axis=1, inplace=True)

#2. Extract racial predictor
#2.1. Clean text
df['clean_text'] = df['Text'].apply(lambda x: stop(clean_text(x).replace('rt', '')))
df['protected_attribute'] = df.clean_text.apply(lambda tweet: protected_attributes(tweet))

#3. Extract dataframe, group by selector and protector
occurences = df.groupby(['protected_attribute', 'Label'])['Text'].count().unstack()
print(occurences)
occurences = occurences.reset_index(drop=True).rename({'index': 'protected_attribute'}, axis=1)
occurences.index = ['Without Attribute', 'With Attribute']
print(occurences)
occurences['Overall'] = occurences['Hate Speech'] + occurences['Offensive Language'] + occurences['Neither']
pcts = occurences.iloc[1, :] * 100 / occurences.iloc[0, :]
pcts = pd.DataFrame(pcts).reset_index().rename({0: 'Dataset Proportion'}, axis=1)

#4. Review %s of protector group per label
# Create plotly chart to visualize dataset
overall = pcts[pcts['Label'] == 'Overall']
rest = pcts[pcts['Label'] != 'Overall']

fig = px.bar(rest, x='Label', y='Dataset Proportion', color_discrete_sequence=['black'])
fig.add_trace(go.Bar(x=['Overall'], y=overall['Dataset Proportion'], name='Overall', marker=dict(color='grey')))
fig.add_hline(y=overall['Dataset Proportion'].iloc[0], line_width=2, line_dash="dash", line_color="black")
fig.update_layout(template='simple_white', width=400)


fig.show()
#5. Perform chi2-squared test for the selection
def review_influence(occurences, significance=0.05):
    if chi2_contingency(occurences).pvalue < significance:
        print('The protected attribute (racial) has an influence over the predicted label')
    else:
        print('''There is no statistical evidence regarding racial bias from the co-occurrences alone. \n
              More investigation is needed.''')
