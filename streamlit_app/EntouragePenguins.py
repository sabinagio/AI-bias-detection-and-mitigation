import streamlit as st
import numpy as np
import pandas as pd

#import frequency_review

st.session_state["apptitle"] = "🐧 PULSE - AI Bias Identification and Correction Tool"
st.session_state["dataset"] = ""

st.title(st.session_state["apptitle"])
#st.subheader(f"Selected Dataset: {st.session_state["dataset"]}")
st.header("",divider="rainbow")
st.write("Welcome to the Entourage Penguins' Bias Identification and Correction Tool")


# st.plotly_chart(frequency_review.fig, use_container_width=True)
# s = frequency_review.review_influence(frequency_review.occurences)
# st.write("hi")
