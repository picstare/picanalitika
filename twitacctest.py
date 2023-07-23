import streamlit as st
import pandas as pd
import numpy as np
from utils import logout
from streamlit_extras.switch_page_button import switch_page
import os
import json
import datetime as dt 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pyvis.network import Network as net
import tweepy
from streamlit_tags import st_tags
from json import JSONEncoder
import PIL
import time
import nltk
import gensim
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pyLDAvis.gensim
from matplotlib import font_manager, mathtext
from pathlib import Path 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from sklearn.metrics import accuracy_score
import plotly.express as px
from streamlit_extras.app_logo import add_logo
import base64
from PIL import Image

consumer_key = "SH94lFcG7ADN1fxrSS57bYzRK"
consumer_secret = "8GnVjcQgpHVG1oATTcJer4ezmgxlUCLWCoNNyW6kV3L19urgPA"
access_token = "1617763543448956929-Ox9uLKXVuLKP00P1ufqqjvfwio9zPI"
access_token_secret = "gw6ZMDnvxY5d7q8l9GaiMKe0mue05ZqCxOQj1m1lSEqGv"
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAN2QoAEAAAAAVJDVMnIlp5SOzQ7sDF9whqRk5zk%3DYJR7rwXSh1utXkl6thLwejvqlJh1kiyNpP3izvP5l3F93iJVfo'

accounts = []
auth = tweepy.OAuth1UserHandler(consumer_key=consumer_key, consumer_secret=consumer_secret, access_token=access_token, access_token_secret=access_token_secret)
# auth.set_access_token(, 'ph47iEpbfD4USmwynPCL1LLNtl7f9seLovjIHOUqwTuQq')
api = tweepy.API(auth)
client = tweepy.Client(bearer_token=bearer_token)

# with st.form(key="taccountform"):
#                 accounts = st_tags(
#                 label='# Enter Account:',
#                 text='Press enter to add more',
#                 value=[],
#                 suggestions=[],
#                 maxtags=4,
#                 key='1')

#                 submit = st.form_submit_button(label="Submit")
#                 if submit:
#                     for account in accounts:
#                         user = api.get_user(screen_name=account)
#                         name = user.name
#                         description = user.description


#                         # get the list of followers for the user
#                         followers = api.get_followers(screen_name=account)
#                         follower_list = [follower.screen_name for follower in followers]

#                         # get the list of users that the user follows
#                         following = api.get_friends(screen_name=account)
#                         following_list = [friend.screen_name for friend in following]

#                         # find friends that do not follow back
#                         not_followed_back = [friend for friend in following_list if friend not in follower_list]

#                         # find followers that have not been followed back
#                         not_following_back = [follower for follower in follower_list if follower not in following_list]

#                         # find friends that follow back
#                         followed_back = [friend for friend in following_list if friend in follower_list]

#                         # find followers that are also friends
#                         following_back = [follower for follower in follower_list if follower in following_list]


#                         # get the user's tweets
#                         tweets = api.user_timeline(screen_name=account, count=5, tweet_mode='extended')
#                         tweets_list = [tweet._json for tweet in tweets]

#                         # search for tweets mentioning the user
#                         mention_tweets = api.search(q=f"@{account}", count=5, tweet_mode='extended')
#                         mention_tweets_list = [tweet._json for tweet in mention_tweets]

#                         # create a dictionary to store the user's information, tweets, friends, and followers
#                         user_data = {
#                             'name': name,
#                             'description': description,
#                             'followers': follower_list,
#                             'following': following_list,
#                             'not_followed_back': not_followed_back,
#                             'not_following_back': not_following_back,
#                             'followed_back': followed_back,
#                             'following_back': following_back,
#                             'tweets': tweets_list,
#                             'mention_tweets': mention_tweets_list
#                         }

#                        # Create a directory if it doesn't exist
#                         os.makedirs("twittl", exist_ok=True)

#                         file_path = f"twittl/{account}_data.json"
#                         if os.path.exists(file_path):
#                             # Load existing data from the file
#                             with open(file_path, 'r') as json_file:
#                                 existing_data = json.load(json_file)

#                             # Update the existing data with new data
#                             existing_data['name'] = name
#                             existing_data['description'] = description
#                             existing_data['followers'] = follower_list
#                             existing_data['following'] = following_list
#                             # Update other fields as needed

#                             # Write the updated data back to the file
#                             with open(file_path, 'w') as json_file:
#                                 json.dump(existing_data, json_file)
#                         else:
#                             # Create a new file and write the data to it
#                             user_data = {
#                                 'name': name,
#                                 'description': description,
#                                 'followers': follower_list,
#                                 'following': following_list,
#                                 'tweets': tweets_list
#                             }
#                             # Add other fields as needed

#                             with open(file_path, 'w') as json_file:
#                                 json.dump(user_data, json_file)



with st.form(key="taccountform"):
    accounts = st_tags(
        label='# Enter Account:',
        text='Press enter to add more',
        value=[],
        suggestions=[],
        maxtags=4,
        key='1'
    )

    submit = st.form_submit_button(label="Submit")
    if submit:
        data_rows = []
        for account in accounts:
            user = api.get_user(screen_name=account)
            name = user.name
            description = user.description

            # get the list of followers for the user
            followers = api.get_followers(screen_name=account)
            follower_list = [follower.screen_name for follower in followers]

            # get the list of users that the user follows
            following = api.get_friends(screen_name=account)
            following_list = [friend.screen_name for friend in following]

            # find friends that do not follow back
            not_followed_back = [friend for friend in following_list if friend not in follower_list]

            # find followers that have not been followed back
            not_following_back = [follower for follower in follower_list if follower not in following_list]

            # find friends that follow back
            followed_back = [friend for friend in following_list if friend in follower_list]

            # find followers that are also friends
            following_back = [follower for follower in follower_list if follower in following_list]

            # get the user's tweets
            tweets = api.user_timeline(screen_name=account, count=5, tweet_mode='extended')
            tweets_list = [tweet._json for tweet in tweets]

            # search for tweets mentioning the user
            mention_tweets = api.search_tweets(q=f"@{account}", count=5, tweet_mode='extended')
            mention_tweets_list = [tweet._json for tweet in mention_tweets]

            # create a dictionary to store the user's information, tweets, friends, and followers
            user_data = {
                'name': name,
                'description': description,
                'followers': follower_list,
                'following': following_list,
                'not_followed_back': not_followed_back,
                'not_following_back': not_following_back,
                'followed_back': followed_back,
                'following_back': following_back,
                'tweets': tweets_list,
                'mention_tweets': mention_tweets_list
            }

            # Append the user data to the list of rows
            data_rows.append(user_data)

            # Create a directory if it doesn't exist
            os.makedirs("twittl", exist_ok=True)

            file_path = f"twittl/{account}_data.json"
            if os.path.exists(file_path):
                # Load existing data from the file
                with open(file_path, 'r') as json_file:
                    existing_data = json.load(json_file)

                # Update the existing data with new data
                existing_data['name'] = name
                existing_data['description'] = description
                existing_data['followers'] = follower_list
                existing_data['following'] = following_list
                # Update other fields as needed

                # Write the updated data back to the file
                with open(file_path, 'w') as json_file:
                    json.dump(existing_data, json_file)
            else:
                # Create a new file and write the data to it
                user_data = {
                    'name': name,
                    'description': description,
                    'followers': follower_list,
                    'following': following_list,
                    'tweets': tweets_list
                }
                # Add other fields as needed

                with open(file_path, 'w') as json_file:
                    json.dump(user_data, json_file)

import glob
import pandas as pd

# ...

data_files = glob.glob("twittl/*_data.json")  # Get a list of all data files

data_rows = []
for file_path in data_files:
    with open(file_path, 'r') as json_file:
        user_data = json.load(json_file)
        flattened_data = {
            'name': user_data['name'],
            'description': user_data['description'],
            'followers': ', '.join(user_data['followers']),
            'following': ', '.join(user_data['following']),
            'not_followed_back': ', '.join(user_data.get('not_followed_back', [])),
            'not_following_back': ', '.join(user_data.get('not_following_back', [])),
            'followed_back': ', '.join(user_data.get('followed_back', [])),
            'following_back': ', '.join(user_data.get('following_back', [])),
            'tweets': ', '.join([tweet['full_text'] for tweet in user_data.get('tweets', [])]),
            'mention_tweets': ', '.join([tweet['full_text'] for tweet in user_data.get('mention_tweets', [])])
        }
        data_rows.append(flattened_data)

df = pd.DataFrame(data_rows)
st.dataframe(df)



