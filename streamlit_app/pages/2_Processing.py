import streamlit as st
import numpy as np
import pandas as pd

lorem = """
"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
"""



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

pipeline_steps = [
    "Step 1",
    "Step 2",
    "Step 3",
    "Step 4"
]


mock_biasdetection = [
    "Racial Prejudice",
    "Hating Poor People",
    "Not being a cool dawg"
]

for i,step in enumerate(pipeline_steps):
    st.subheader(step)
    st.write(lorem)
    if(st.button(f"Execute {step}", type="primary",key=f"bt_processing_step_{i}")):
        st.write("Executed")


st.header("Bias Identification")
for bias in mock_biasdetection:
    st.info(f'Found Bias: {bias}', icon="🚨")
    st.error(f'Found Bias: {bias}', icon="🚨")

st.write(lorem)


st.header("Bias Correction")
for bias in mock_biasdetection:
    st.info(f'Corrected/Mitigated Bias: {bias}', icon="✔️")
    st.success(f'Corrected/Mitigated Bias: {bias}', icon="✔️")
st.write(lorem)


st.header("Model Training")
st.write(lorem)

#
# EVALUATION
#
st.header("Evaluation")
st.write(lorem)

# TODO: Show metrics of initial dataset
# TODO: Show metrics of corrected dataset

df_raw = pd.DataFrame(columns=["Metric","Scoring"])


d = {
    "Metric" : ["M1","M2","M3","M4"],
    "Scoring_Raw": [1,2,3,4],
    "Scoring_Corrected": [2,3,4,5],
    "Improvement": ["3.7%","4.5%","7.0%","2.0%"]
    
}
df_comparison = pd.DataFrame(data=d).style.hide()
st.write(df_comparison)



import time
if(st.button("Upload corrected Dataset", type="primary",key="bt_uploadmitigateddataset")):
    #TODO: spinner while uploading
    with st.spinner('Uploading Dataset...'):
        time.sleep(5)
    st.success('Done!')
    st.error("ohoh")
    st.session_state["Multipage_var1"] = "Upload Done"


text_contents = """This is some text"""
st.metric(label="Bias Score Improvement", value="65%", delta="17.5%",
    delta_color="normal")
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
