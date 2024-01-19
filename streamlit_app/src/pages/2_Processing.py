import streamlit as st
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import nltk
from preprocess import clean_text, stop
from scipy.stats import chi2_contingency
import streamlit as st

lorem = """
"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
"""
st.session_state["df"] = pd.DataFrame()


st.title(st.session_state["apptitle"])
#st.subheader(f"Selected Dataset: {st.session_state["dataset"]}")
st.header("",divider="rainbow")

#file = st.file_uploader("Pick a file")

#add_statefield_menunavigation = st.sidebar.header("xxxx")
#st.sidebar.header(st.session_state['selected_dataset'])
# add_selectbox_menunavigation = st.sidebar.selectbox(
#     'Where do you want to go',
#     ('Bias Identification', 'Bias Correction', 'Train Model', "#processing-step-4",unsafe_allow_html=True)
# )

def read_local_dataframe(filename):

    return pd.read_csv(filename)


#
# Sessions State
#
df = pd.DataFrame



####################
#st.bar_chart({"data": [1, 5, 2, 6, 2, 1]})




# with st.expander("Twitter Hatespeech"):
#     st.write("DATASET DESCRIPTION")
#     st.image("https://static.streamlit.io/examples/dice.jpg")
#     if(st.button("Choose Dataset", type="primary",key="bt_twitterhatespeech")):
#         st.write('Dataset selected')
#         st.session_state['selected_dataset'] = 'Twitter Hatespeech'
#         df = read_local_dataframe("twitterhatespeech.csv")


# with st.expander("Jigsaw Racial"):
#     st.write("DATASET DESCRIPTION")
#     st.image("https://static.streamlit.io/examples/dice.jpg")
#     if(st.button("Choose Dataset", type="primary",key="bt_jigsaw_racial")):
#         st.write('Dataset selected')
#         st.session_state['selected_dataset'] = 'Jigsaw Racial'
#         df = read_local_dataframe("jigsawracial.csv")


######
# selected dataset
######




#st.header("Dataset: " + st.session_state['selected_dataset'])
#for row in df.itertuples():
    #st.write(f"{row.Owner} has a :{row.Pet}:")
    #st.write(row)
#st.header("Dataset: " + st.session_state['dataset'])
if(not df.empty):
    st.write(df)

# with st.expander("Twitter Hatespeech"):
#     next

# st.header("Dataset Processing")


###############################
# Processing Steps 1: Extracting Racial Predictor
################################
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

stepname = "Extracting Racial Attributes"
# st.subheader(stepname)
#st.write("Preprocessing of the dataset, cleaning up text, removing unnecessary object (stop words)")
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
with st.spinner(stepname):
    df['clean_text'] = df['Text'].apply(lambda x: stop(clean_text(x).replace('rt', '')))
    df['protected_attribute'] = df.clean_text.apply(lambda tweet: protected_attributes(tweet))        
st.success(f'Done: {stepname}', icon="✔️")
    #st.write(df.value_counts("protected_attribute"))
#st.session_state["df"] = df
###############################
# Processing Steps 2: Dataframe Processing
################################
#df = st.session_state["df"]
stepname = "Dataframe Processing"
#st.subheader(stepname)
#st.write("Processing the dataframe, calculating occurences of protected attributes")
#if(st.button(f"{stepname}", type="primary",key=f"bt_processing_step_{stepname}")):
    #df = st.session_state["df"]
with st.spinner(stepname):
    #df = st.session_state["df"]
    #3. Extract dataframe, group by selector and protector
    occurences = df.groupby(['protected_attribute', 'Label'])['Text'].count().unstack()
    print(occurences)
    occurences = occurences.reset_index(drop=True).rename({'index': 'protected_attribute'}, axis=1)
    occurences.index = ['Without Attribute', 'With Attribute']
    print(occurences)
    occurences['Overall'] = occurences['Hate Speech'] + occurences['Offensive Language'] + occurences['Neither']
    pcts = occurences.iloc[1, :] * 100 / occurences.iloc[0, :]
    pcts = pd.DataFrame(pcts).reset_index().rename({0: 'Dataset Proportion'}, axis=1)
st.success(f'Done: {stepname}', icon="✔️")

###############################
# Processing Steps 3: Visualization
################################
stepname = "Bias detection"
st.subheader(stepname)
#st.write("Creating visualization of the underlying data for a quick overview of potential biases")
#if(st.button(f"{stepname}", type="primary",key=f"bt_processing_step_{stepname}")):
with st.spinner(stepname):
    #4. Review %s of protector group per label
    # Create plotly chart to visualize dataset
    overall = pcts[pcts['Label'] == 'Overall']
    rest = pcts[pcts['Label'] != 'Overall']

    fig = px.bar(rest, x='Label', y='Dataset Proportion', color_discrete_sequence=['black'])
    fig.add_trace(go.Bar(x=['Overall'], y=overall['Dataset Proportion'], name='Overall', marker=dict(color='grey')))
    fig.add_hline(y=overall['Dataset Proportion'].iloc[0], line_width=2, line_dash="dash", line_color="black")
    fig.update_layout(width=400, paper_bgcolor='white', plot_bgcolor='white', legend_font_color='black')
    fig.update_xaxes(title_font_color='black', color='black', gridcolor='white', linecolor='black', tickfont_color='black')
    fig.update_yaxes(title_font_color='black', color='black', gridcolor='white', linecolor='black', tickfont_color='black')
    #fig.show()
    st.plotly_chart(fig, use_container_width=True)
#st.success(f'Done: {stepname}', icon="✔️")
###############################
# Processing Steps 4: Chi2-Squared Testing
################################
stepname = "Statistical Testing"
st.subheader(stepname)
#st.write("This processing step will extract protected attributes from the dataset")
@st.cache_data
def review_influence(occurences, significance=0.05):
    if chi2_contingency(occurences).pvalue < significance:
        return(f'The racial attribute has an influence over the predicted label (p-value: {chi2_contingency(occurences).pvalue})')
    else:
        return('''There is no statistical evidence regarding racial bias from the co-occurrences alone. \n
            More investigation is needed.''')
#if(st.button(f"{stepname}", type="primary",key=f"bt_processing_step_{stepname}")):
with st.spinner(stepname):
#5. Perform chi2-squared test for the selection
    st.success(f'Done: {stepname}', icon="✔️")
st.error(f'{review_influence(occurences)}', icon="🚨")



###############################
# Processing Steps 5: Show Metrics
################################
stepname = "Model Bias Accuracy Metrics"
st.subheader(stepname)
#st.write("This processing step will extract protected attributes from the dataset")


metrics_df = pd.read_csv("metrics.csv")
#st.write(metrics_df)


df_raw = pd.DataFrame(columns=["Metric","Scoring"])

def calc_delta(baseline,comparison):
    return ((comparison/baseline)-1)*100

st.markdown("#### Original Bias")
spacer01, col1, spacer12,  col2, spacer23, col3, spacer30 = st.columns([0.1, 2, 0.1, 2, 0.1, 2, 0.1])

with col1:
    original_bspn_auc = metrics_df.iloc[0]["bspn_auc"]
    st.metric(label="BSPN_AUC", value="{:.4f}".format(0.9221678678494207))
with col2:
    original_bpsn_auc = metrics_df.iloc[0]["bpsn_auc"]
    st.metric(label="BPSN_AUC", value="{:.4f}".format(0.8173775950907339))
with col3:
    original_subgroup_auc = metrics_df.iloc[0]["subgroup_auc"]
    st.metric(label="Subgroup_AUC", value="{:.4f}".format(0.9221678678494207))


st.markdown("#### Correct Bias Weights")
spacer01, col1, spacer12,  col2, spacer23, col3, spacer30 = st.columns([0.1, 2, 0.1, 2, 0.1, 2, 0.1])
with col1:
    correctedbias_bspn_auc = metrics_df.iloc[1]["bspn_auc"]
    st.metric(label="BSPN_AUC", value="{:.4f}".format(correctedbias_bspn_auc), delta="{:.2f}%".format(calc_delta(original_bspn_auc,correctedbias_bspn_auc)))
with col2:
    correctedbias_bpsn_auc = metrics_df.iloc[1]["bpsn_auc"]
    st.metric(label="BPSN_AUC", value="{:.4f}".format(correctedbias_bpsn_auc), delta="{:.2f}%".format(calc_delta(original_bpsn_auc,correctedbias_bpsn_auc)))
with col3:
    correctedbias_subgroup_auc = metrics_df.iloc[1]["subgroup_auc"]
    st.metric(label="Subgroup_AUC", value="{:.4f}".format(correctedbias_subgroup_auc), delta="{:.2f}%".format(calc_delta(original_subgroup_auc,correctedbias_subgroup_auc)))

st.markdown("#### Adversarial Debiasing")
spacer01, col1, spacer12,  col2, spacer23, col3, spacer30 = st.columns([0.1, 2, 0.1, 2, 0.1, 2, 0.1])
with col1:
    adv_bspn_auc = metrics_df.iloc[2]["bspn_auc"]
    st.metric(label="BSPN_AUC", value="{:.4f}".format(0.865586639571199), delta="{:.2f}%".format(calc_delta(original_bspn_auc,adv_bspn_auc)))
with col2:
    adv_bpsn_auc = metrics_df.iloc[2]["bpsn_auc"]
    st.metric(label="BPSN_AUC", value="{:.4f}".format(0.8827878406326195), delta="{:.2f}%".format(calc_delta(original_bpsn_auc,adv_bpsn_auc)))
with col3:
    adv_subgroup_auc = metrics_df.iloc[2]["subgroup_auc"]
    st.metric(label="Subgroup_AUC", value="{:.4f}".format(0.865586639571199), delta="{:.2f}%".format(calc_delta(original_subgroup_auc,adv_subgroup_auc)))


# st.metric(label="Awesomeness", value="95%", delta="2%",
#     delta_color="normal")
###############################
# Processing Steps 6: Download Data
################################
# d = {
#     "Metric" : ["M1","M2","M3","M4"],
#     "Scoring_Raw": [1,2,3,4],
#     "Scoring_Corrected": [2,3,4,5],
#     "Improvement": ["3.7%","4.5%","7.0%","2.0%"]
    
# }
# df_comparison = pd.DataFrame(data=d).style.hide()
# st.write(df_comparison)



import time

# st.metric(label="Bias Score Improvement", value="65%", delta="17.5%",
#     delta_color="normal")
text_contents = """This is some text"""

st.download_button('Download Evaluation Summary', text_contents)


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')


row = 5
cols = 5

Random = np.random.randint(low=0, high=100, size=(row, cols))

df3 = pd.DataFrame(data=Random,columns=["Bias Metric A","Bias Metric B","Bias Metric C","Bias Metric D","Bias Metric E"])

print(df)
csv = convert_df(df3)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',
)
#st.line_chart(df)
# x = st.slider('x')  # 👈 this is a widget
# st.write(x, 'squared is', x * x)



#################################



