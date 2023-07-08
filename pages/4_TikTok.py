import streamlit as st
import pandas as pd
import numpy as np
from utils import logout
from streamlit_extras.switch_page_button import switch_page
import os
import json
from datetime import datetime
import asyncio
# Import subprocess to run tiktok script from command line
from subprocess import call
# Import plotly for viz
import plotly.express as px

st.set_page_config(page_title="Picanalitika | TikTok Analysis", layout="wide")

hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div.vis-configuration-wrapper{display: block; width: 400px;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)





####################LOGOUT####################
with st. sidebar:
    if st.button("Logout"):
        logout()
        switch_page('home')

#################STARTPAGE###################

a, b = st.columns([1, 10])

with a:
    st.text("")
    st.image("img/tiktoklogo.png", width=100)
with b:
    st.title("TikTok Analysis")
###############################################
listTabs = [
    "👨‍💼 Key Persons Analysis",
    "🦈 Issue Analysis",
    "📈 Data Mining",
    
]

font_css = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 24px;
}
</style>
"""
st.write(font_css, unsafe_allow_html=True)
whitespace = 30
tab1, tab2, tab3 = st.tabs([s.center(whitespace,"\u2001") for s in listTabs])

with tab1:
    st.header("KEY PERSONS")

with tab2:
    st.header("ISSUE")

with tab3:
    st.header("DATA")



# Input 
    hashtag = st.text_input('Search for a hashtag here', value="")

    # Button
    if st.button('Get Data'):
        # Run get data function here
        call(['python', 'tiktokback.py', hashtag])
        # Load in existing data to test it out
        df = pd.read_csv('tiktokdata.csv')

        # Plotly viz here
        fig = px.histogram(df, x='desc', hover_data=['desc'], y='stats_diggCount', height=300) 
        st.plotly_chart(fig, use_container_width=True)

        # Split columns
        left_col, right_col = st.columns(2)

        # First Chart - video stats
        scatter1 = px.scatter(df, x='stats_shareCount', y='stats_commentCount', hover_data=['desc'], size='stats_playCount', color='stats_playCount')
        left_col.plotly_chart(scatter1, use_container_width=True)

        # Second Chart
        scatter2 = px.scatter(df, x='authorStats_videoCount', y='authorStats_heartCount', hover_data=['author_nickname'], size='authorStats_followerCount', color='authorStats_followerCount')
        right_col.plotly_chart(scatter2, use_container_width=True)


        # Show tabular dataframe in streamlit
        df