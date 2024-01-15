import streamlit as st
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import nltk
from preprocess import clean_text, stop
from scipy.stats import chi2_contingency

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

st.header("Dataset Processing")


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

stepname = "Extract Protected Attributes"
st.subheader(stepname)
st.write("This processing step will extract protected attributes from the dataset")
if(st.button(f"{stepname}", type="primary",key=f"bt_processing_step_{stepname}")):
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
stepname = "Creating"
st.subheader(stepname)
st.write("This processing step will extract protected attributes from the dataset")
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
stepname = "Visualization"
st.subheader(stepname)
st.write("This processing step will extract protected attributes from the dataset")
#if(st.button(f"{stepname}", type="primary",key=f"bt_processing_step_{stepname}")):
with st.spinner(stepname):
    #4. Review %s of protector group per label
    # Create plotly chart to visualize dataset
    overall = pcts[pcts['Label'] == 'Overall']
    rest = pcts[pcts['Label'] != 'Overall']

    fig = px.bar(rest, x='Label', y='Dataset Proportion', color_discrete_sequence=['black'])
    fig.add_trace(go.Bar(x=['Overall'], y=overall['Dataset Proportion'], name='Overall', marker=dict(color='grey')))
    fig.add_hline(y=overall['Dataset Proportion'].iloc[0], line_width=2, line_dash="dash", line_color="black")
    fig.update_layout(template='simple_white', width=400)
    #fig.show()
    st.plotly_chart(fig, use_container_width=True)
st.success(f'Done: {stepname}', icon="✔️")
###############################
# Processing Steps 4: Chi2-Squared Testing
################################
stepname = "Chi2-Squared Testing"
st.subheader(stepname)
st.write("This processing step will extract protected attributes from the dataset")
@st.cache_data
def review_influence(occurences, significance=0.05):
    if chi2_contingency(occurences).pvalue < significance:
        return('The protected attribute (racial) has an influence over the predicted label')
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
stepname = "Show Metrics"
st.subheader(stepname)
st.write("This processing step will extract protected attributes from the dataset")


metrics_df = pd.read_csv("metrics.csv")
st.write(metrics_df)


df_raw = pd.DataFrame(columns=["Metric","Scoring"])

st.subheader("Original Bias")
st.write(metrics_df.iloc[0])
st.metric(label="BSPN_AUC", value="70 °F", delta="1.2 °F")
st.metric(label="BPSN_AUC", value="70 °F", delta="1.2 °F")
st.metric(label="SUbgroup_AUC", value="70 °F", delta="1.2 °F")

st.subheader("Correct Bias Weights")
#st.metric(label="Correct Bias Weights", value="70 °F", delta="1.2 °F")
st.metric(label="BSPN_AUC", value="70 °F", delta="1.2 °F")
st.metric(label="BPSN_AUC", value="70 °F", delta="1.2 °F")
st.metric(label="SUbgroup_AUC", value="70 °F", delta="1.2 °F")
st.write(metrics_df.iloc[1])

st.subheader("Adversarial Debiasing")
#st.metric(label="Adversarial Debiasing", value="70 °F", delta="1.2 °F")
st.metric(label="BSPN_AUC", value="70 °F", delta="1.2 °F")
st.metric(label="BPSN_AUC", value="70 °F", delta="1.2 °F")
st.metric(label="SUbgroup_AUC", value="70 °F", delta="1.2 °F")
st.write(metrics_df.iloc[2])



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
if(st.button("Upload corrected Dataset", type="primary",key="bt_uploadmitigateddataset")):
    #TODO: spinner while uploading
    with st.spinner('Uploading Dataset...'):
        time.sleep(5)
    st.success('Done!')
    st.error("ohoh")



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



