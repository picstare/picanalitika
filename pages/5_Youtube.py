import os
import googleapiclient.discovery
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json
import streamlit as st
import pandas as pd
import numpy as np
from utils import logout
from streamlit_extras.switch_page_button import switch_page
from datetime import datetime, timedelta
import requests
import base64
from PIL import Image
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt


st.set_page_config(page_title="Picanalitika | Youtube Analysis", layout="wide")

###################LOGOUT####################
with st. sidebar:
    if st.button("Logout"):
        logout()
        switch_page('home')

#################STARTPAGE###################

a, b = st.columns([1, 10])

with a:
    st.text("")
    st.image("img/youtubelogo.png", width=100)
with b:
    st.title("Youtube Analysis")
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

    folder_path = "ytaccscrap"
    files = os.listdir(folder_path)

    # Get the modification times of the files
    file_times = [(f, os.path.getmtime(os.path.join(folder_path, f))) for f in files]

    # Sort the files based on modification time in descending order
    sorted_files = sorted(file_times, key=lambda x: x[1], reverse=True)

    # Select the four newest files
    num_files = min(3, len(sorted_files))
    newest_files = [f[0] for f in sorted_files[:num_files]]

    # Update the 'files' variable with the names of the newest files
    files = newest_files

    if len(files) > 0:
        # Create a Streamlit column for each file
        cols = st.columns(num_files)

        for i, col in enumerate(cols):
            # Check if the file is in JSON format
            if i < num_files and files[i].endswith('.json'):
                # Open the file and read its contents as a JSON object
                with open(os.path.join(folder_path, files[i]), 'r') as f:
                    channel_details = json.load(f)

                    # Extract relevant information from the channel details
                    # Extract relevant information from the channel details
                # Extract relevant information from the channel details
                title = channel_details['channel_details']['snippet']['localized']['title']
                description = channel_details['channel_details']['snippet']['localized']['description']
                thumbnail_url = channel_details['channel_details']['snippet']['thumbnails']['medium']['url']
                subscriber_count = int(channel_details['channel_details']['statistics']['subscriberCount'])
                view_count = int(channel_details['channel_details']['statistics']['viewCount'])
                video_count = int(channel_details['channel_details']['statistics']['videoCount'])

                # Display the channel details using Streamlit
                

                # Download the thumbnail image from the URL
                response = requests.get(thumbnail_url)
                thumbnail_image = Image.open(BytesIO(response.content))

                col.image(thumbnail_image, caption='', width=200)
                col.subheader(title)
                col.markdown(description)

                # Display statistics
            
                col.metric("Subscribers", f"{subscriber_count:,}")
                col.metric("Views", f"{view_count:,}")
                col.metric("Videos", f"{video_count:,}")

                
                    
########################TIME SERIES ANALYSIS##############################
    st.header('Time Series Analysis of The Key Persons')
    
    folder_path = "ytaccscrap"
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
        # data = []
    df1 = None

    if files:
        data = []
        for file in files:
            with open(file, 'r') as f:
                file_data = json.load(f)

                channel_details = file_data['channel_details']
                channel = channel_details['snippet']['title']


                for video in file_data['videos']:
                    data.append({
                        'channel': channel,
                        'date': pd.to_datetime(video['published_at'])
                    })
        df1 = pd.DataFrame(data)

    # Create a list of available screen names
    if df1 is not None:
        channels = list(df1['channel'].unique())
    else:
        channels = []

    # Set the default selected names to the first 4 channelsselected_channels
    default_channels = channels[:4]

    # Set the default time range to one month from the current date
    end_date = pd.to_datetime(datetime.today(), utc=True)
    start_date = end_date - timedelta(days=30)

    # Create widgets for selecting the screen name and time range
    selected_channels = st.multiselect('Select channel to compare', channels, default=default_channels, key='yta((t;')
    cols_ta, cols_tb = st.columns([1, 1])
    start_date = pd.to_datetime(cols_ta.date_input('Start date', value=start_date), utc=True)
    end_date = pd.to_datetime(cols_tb.date_input('End date', value=end_date), utc=True)

    # Filter the data based on the selected names and time range
    if df1 is not None:
        mask = (df1['channel'].isin(selected_channels)) & (df1['date'] >= start_date) & (df1['date'] <= end_date)
        df1_filtered = df1.loc[mask]
    else:
        df1_filtered = pd.DataFrame()

    if len(df1_filtered) > 0:
        df1_grouped = df1_filtered.groupby(['date', 'channel']).size().reset_index(name='count')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df1_grouped, x='date', y='count', hue='channel', ax=ax)
        ax.set_title(f"Video Post per Day for {', '.join(selected_channels)}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Tweets")
        st.pyplot(fig)
    else:
        st.write("No data available for the selected time range and users.")

    st.markdown("---")



#########################################SNA ACCOUNT YOUTUBE #############################################






######################################TOPIC MODELING #####################################
    

    from nltk.sentiment import SentimentIntensityAnalyzer
    import numpy as np
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from datetime import datetime
    import json
    import os
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import matplotlib.pyplot as plt
    import pandas as pd
    import streamlit as st
    from gensim import corpora
    from gensim.models import LdaModel
    import pyLDAvis.gensim_models
    import pyLDAvis
    from wordcloud import WordCloud

    
    def load_and_preprocess_data(file_data, start_date, end_date):
        preprocessed_text_data = []
        stop_words = set(stopwords.words('indonesian'))
        stemmer = StemmerFactory().create_stemmer()

        if 'videos' in file_data:
            for video in file_data['videos']:
                if 'comments' in video:
                    for comment in video['comments']:
                        published_at = comment['published_at']
                        created_at_date = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=None)

                        if start_date <= created_at_date <= end_date:
                            text = comment['comment']
                            text = text.lower()
                            tokens = word_tokenize(text)
                            processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
                            preprocessed_text_data.append(processed_tokens)

        return preprocessed_text_data


    def perform_topic_modeling(text_data):
        dictionary = corpora.Dictionary(text_data)
        corpus = [dictionary.doc2bow(tokens) for tokens in text_data]

        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)

        vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

        # Get the top keywords for each topic
        top_keywords = []
        for i in range(lda_model.num_topics):
            keywords = [word for word, prob in lda_model.show_topic(i, topn=10)]
            top_keywords.append(keywords)

        return vis_data, top_keywords


    st.title("Topic Modeling of Comments")
    folder_path = "ytaccscrap"
    account_list = [file_name.split("_data.json")[0] for file_name in os.listdir(folder_path) if file_name.endswith('_data.json')]
    selected_channels = st.multiselect("Select channel", account_list, default=account_list[:4], key="c4ntpmtf")
    start_date = st.date_input("Start Date", key="ch4nytst")
    end_date = st.date_input("End Date", key='chanyt3dt')

    if len(selected_channels) == 0:
        st.warning("No channel selected. Please choose at least one account.")
    elif not start_date or not end_date:
        st.warning("Please select a start date and end date.")
    else:
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
        num_columns = len(selected_channels)
        columns = st.columns(num_columns)

        for i, account in enumerate(selected_channels):
            file_name = f"{account}_data.json"
            file_path = os.path.join(folder_path, file_name)
            file_data = None

            with open(file_path, 'r') as f:
                file_data = json.load(f)

            text_data = load_and_preprocess_data(file_data, start_date, end_date)

            if len(text_data) == 0:
                st.warning(f"There were no posts from {account} on {start_date} to {end_date}, so sentiment analysis cannot be performed.")
                continue

            vis_data, top_keywords = perform_topic_modeling(text_data)
            st.subheader(f"Topic Model of Comments in {account}")

            # Save pyLDAvis HTML to a temporary file
            with open(f"{account}_pyldavis.html", "w") as f:
                pyLDAvis.save_html(vis_data, f)

            # Display pyLDAvis using st.components.v1.html
            with open(f"{account}_pyldavis.html", "r") as f:
                st.components.v1.html(f.read(), width=1000, height=800)

            # Display word cloud interactively
            st.subheader("Word Cloud for Selected Topic")
            selected_topic = st.selectbox("Select a topic", [f"Topic {i + 1}" for i in range(len(top_keywords))])

            topic_index = int(selected_topic.split()[-1]) - 1
            keywords = top_keywords[topic_index]
            st.write(f"**{selected_topic} Keywords:** {' | '.join(keywords)}")
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))
            plt.figure(figsize=(8, 4))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"{selected_topic} Keywords Word Cloud")
            st.pyplot(plt)

            # Display horizontal rule to separate topics
            st.markdown("---")

            # Close the file to release resources
            f.close()



##################################SENTIMENT ANALYSIS########################################
    
    
    from nltk.sentiment import SentimentIntensityAnalyzer
    import numpy as np
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from datetime import datetime
    import json
    import os
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import matplotlib.pyplot as plt
    import pandas as pd
    import streamlit as st

    @st.cache_data
    def load_and_preprocess_data(file_data, start_date, end_date):
        preprocessed_text_data = []
        stop_words = set(stopwords.words('indonesian'))
        stemmer = StemmerFactory().create_stemmer()

        if 'videos' in file_data:
            for video in file_data['videos']:
                if 'comments' in video:
                    for comment in video['comments']:
                        published_at = comment['published_at']
                        created_at_date = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=None)

                        if start_date <= created_at_date <= end_date:
                            text = comment['comment']
                            text = text.lower()
                            tokens = word_tokenize(text)
                            processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
                            preprocessed_text_data.append(processed_tokens)

        return preprocessed_text_data

 
    def perform_sentiment_analysis(text_data):
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = []

        for text in text_data:
            sentiment_score = sia.polarity_scores(' '.join(text))
            sentiment_scores.append(sentiment_score)

        df_sentiment = pd.DataFrame(sentiment_scores)
        return df_sentiment


    st.title("Sentiment Analysis")
    folder_path = "ytaccscrap"
    account_list = [file_name.split("_data.json")[0] for file_name in os.listdir(folder_path) if file_name.endswith('_data.json')]
    selected_channels = st.multiselect("Select channel", account_list, default=account_list[:4], key="c4nsentf")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    if len(selected_channels) == 0:
        st.warning("No channel selected. Please choose at least one account.")
    elif not start_date or not end_date:
        st.warning("Please select a start date and end date.")
    else:
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
        num_columns = len(selected_channels)
        columns = st.columns(num_columns)

        for i, account in enumerate(selected_channels):
            file_name = f"{account}_data.json"
            file_path = os.path.join(folder_path, file_name)
            file_data = None

            with open(file_path, 'r') as f:
                file_data = json.load(f)

            text_data = load_and_preprocess_data(file_data, start_date, end_date)

            if len(text_data) == 0:
                st.warning(f"There were no posts from {account} on {start_date} to {end_date}, so sentiment analysis cannot be performed.")
                continue

            df_sentiment = perform_sentiment_analysis(text_data)
            sentiment_distribution = df_sentiment.mean().drop("compound")

            with columns[i]:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.pie(sentiment_distribution.values, labels=sentiment_distribution.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                ax.set_title(f"Sentiment Distribution: {account}")

                st.pyplot(fig)
                with st.expander(""):
                    st.dataframe(df_sentiment)

    

#########################SENTIMENT ANALYSIS PER USER#############################
    from nltk.sentiment import SentimentIntensityAnalyzer
    import numpy as np
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from datetime import datetime
    import json
    import os
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import matplotlib.pyplot as plt
    import pandas as pd
    import streamlit as st


    @st.cache_data
    def load_and_preprocess_data(file_path):
        with open(file_path, 'r') as f:
            channel_data = json.load(f)

        preprocessed_text_data = []
        stop_words = set(stopwords.words('indonesian'))
        stemmer = StemmerFactory().create_stemmer()

        if 'videos' in channel_data:
            for video in channel_data['videos']:
                if 'comments' in video:
                    for comment in video['comments']:
                        # published_at = comment['published_at']
                        text = comment['comment']
                        user = comment['username']  # Get the user name
                        text = text.lower()
                        tokens = word_tokenize(text)
                        processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

                        preprocessed_text_data.append((processed_tokens, user))

        return preprocessed_text_data

        
    @st.cache_data
    def perform_sentiment_analysis_per_user(text_data):
        sia = SentimentIntensityAnalyzer()
        user_sentiment_scores = {}

        for text, user in text_data:
            sentiment_score = sia.polarity_scores(' '.join(text))

            if user not in user_sentiment_scores:
                user_sentiment_scores[user] = {
                    'positive': [],
                    'negative': [],
                    'neutral': [],
                    'compound': []
                }

            user_sentiment_scores[user]['positive'].append(sentiment_score['pos'])
            user_sentiment_scores[user]['negative'].append(sentiment_score['neg'])
            user_sentiment_scores[user]['neutral'].append(sentiment_score['neu'])
            user_sentiment_scores[user]['compound'].append(sentiment_score['compound'])

        user_sentiment_scores_avg = {}
        for user, scores in user_sentiment_scores.items():
            user_sentiment_scores_avg[user] = {
                'positive': np.mean(scores['positive']),
                'negative': np.mean(scores['negative']),
                'neutral': np.mean(scores['neutral']),
                'compound': np.mean(scores['compound'])
            }

        df_sentiment_per_user = pd.DataFrame.from_dict(user_sentiment_scores_avg, orient='index')
        return df_sentiment_per_user

    st.title("Sentiment Analysis per user")

    folder_path = "ytaccscrap"
    account_list = [file_name.split("_data.json")[0] for file_name in os.listdir(folder_path) if file_name.endswith('_data.json')]
    selected_accounts = st.multiselect("Select channels", account_list, default=account_list[:1], key="chan5entselfil")
    start_date = st.date_input("Start Date", key="chan5entu4(t")
    end_date = st.date_input("End Date", key="ch4n3dinp)z")

    if len(selected_accounts) == 0:
        st.warning("No channels selected. Please choose at least one channel.")
    elif not start_date or not end_date:
        st.warning("Please select a start date and end date.")
    else:
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
        num_columns = len(selected_accounts)
        columns = st.columns(num_columns)

        for i, account in enumerate(selected_accounts):
            file_name = f"{account}_data.json"
            file_path = os.path.join(folder_path, file_name)

            if not os.path.exists(file_path):
                st.warning(f"No data available for {account}.")
                continue

            text_data = load_and_preprocess_data(file_path)

            if len(text_data) == 0:
                st.warning(f"There were no comments for {account}'s videos on {start_date} to {end_date}, so sentiment analysis cannot be performed.")
                continue

            df_sentiment_per_user = perform_sentiment_analysis_per_user(text_data)

            st.subheader(f"Sentiment Analysis per User: {account}")

            with st.expander(""):
                st.dataframe(df_sentiment_per_user)

            ax = df_sentiment_per_user.plot(kind='bar', rot=0, fontsize=5)
            plt.xlabel('User', fontsize=7)
            plt.ylabel('Sentiment Score', fontsize=7)
            plt.title(f"Sentiment Analysis per User: {account}")
            plt.xticks(rotation='vertical', fontsize=5)
            plt.legend(fontsize=5)
            plt.tight_layout()

            for p in ax.patches:
                ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=5)

            st.pyplot(plt)


##################################LOCATION#############################

    # import json
    # import os
    # import pandas as pd
    # from datetime import datetime
    # import folium
    # from streamlit_folium import folium_static, st_folium
    # from geopy.geocoders import Nominatim
    # from geopy.exc import GeocoderUnavailable
    # import streamlit as st

    # def load_tweet_data(file_path):
    #     with open(file_path, 'r') as f:
    #         channel_data = json.load(f)
    #     return channel_data['videos']

    # def filter_comments_by_date(tweets, start_date, end_date):
    #     filtered_tweets = []
    #     for tweet in tweets:
    #         created_at = datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S %z %Y').replace(tzinfo=None)
    #         if start_date <= created_at <= end_date:
    #             filtered_tweets.append(tweet)
    #     return filtered_tweets

    # def get_tweet_user_locations(tweets):
    #     user_locations = {}
    #     for tweet in tweets:
    #         user = tweet['user']
    #         user_name = user['screen_name']
    #         location = user['location']
    #         if location:
    #             if user_name not in user_locations:
    #                 user_locations[user_name] = {
    #                     'location': location,
    #                     'created_at': tweet['created_at']
    #                 }
    #     return user_locations

    # st.title("Tweet User Locations")

    # folder_path = "twittl"
    # account_list = [file_name.split("_data.json")[0] for file_name in os.listdir(folder_path) if file_name.endswith('_data.json')]
    # selected_accounts = st.multiselect("Select Accounts", account_list, default=account_list[:1], key="accselloc")
    # start_date = st.date_input("Start Date", key="acl0c)sd")
    # end_date = st.date_input("End Date", key='a(loc3d')

    # if len(selected_accounts) == 0:
    #     st.warning("No accounts selected. Please choose at least one account.")
    # elif not start_date or not end_date:
    #     st.warning("Please select a start date and end date.")
    # else:
    #     start_date = datetime.combine(start_date, datetime.min.time())
    #     end_date = datetime.combine(end_date, datetime.max.time())
    #     user_locations = {}

    #     for account in selected_accounts:
    #         file_name = f"{account}_data.json"
    #         file_path = os.path.join(folder_path, file_name)

    #         if not os.path.exists(file_path):
    #             st.warning(f"No data available for {account}.")
    #             continue

    #         tweets = load_tweet_data(file_path)
    #         filtered_tweets = filter_tweets_by_date(tweets, start_date, end_date)
    #         account_user_locations = get_tweet_user_locations(filtered_tweets)
    #         user_locations.update(account_user_locations)

    #     if not user_locations:
    #         st.warning(f"No tweets available for the selected accounts and date range.")
    #     else:
    #         df_user_locations = pd.DataFrame.from_dict(user_locations, orient='index')
    #         df_user_locations['User'] = df_user_locations.index
    #         df_user_locations = df_user_locations[['User', 'location', 'created_at']]

    #         st.subheader("Tweet User Locations")

    #         with st.expander("User Location Data"):
    #             st.dataframe(df_user_locations)

    #         location_counts = df_user_locations['location'].value_counts()

        
    #         # Create a Folium map centered around the first user location
    #         geolocator = Nominatim(user_agent="tweet_location_geocoder")
    #         first_location = df_user_locations['location'].iloc[0]
    #         location = geolocator.geocode(first_location)
    #         if location:
    #             lat, lon = location.latitude, location.longitude
    #             tweet_map = folium.Map(location=[lat, lon], zoom_start=6)
    #         else:
    #             tweet_map = folium.Map(location=[0, 0], zoom_start=2)

    #         # Add markers for each user location
    #         for _, row in df_user_locations.iterrows():
    #             location = row['location']
    #             user = row['User']
    #             tweet_time = row['created_at']
    #             geocode_result = geolocator.geocode(location, timeout=10)
    #             if geocode_result:
    #                 lat, lon = geocode_result.latitude, geocode_result.longitude
    #                 folium.Marker(
    #                     location=[lat, lon],
    #                     popup=f"User: {user}<br>Location: {location}<br>Time: {tweet_time}"
    #                 ).add_to(tweet_map)

    #         # Display the map
    #         folium_static(tweet_map, width=1300, height=600)



############################################

with tab2:
    st.header("ISSUE")

################################time series#################
    import json
    import glob
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    from datetime import datetime, timedelta

    # Function to load video data from files
    def load_video_data(file_paths):
        video_data_list = []
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                video_data_list.extend(json_data)
        return video_data_list

    def filter_video_data_by_date(video_data_list, start_date, end_date):
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)

        filtered_data = [data for data in video_data_list if start_datetime <= datetime.fromisoformat(data['date_posted']) < end_datetime]
        return filtered_data

    def create_multiline_time_series_plot(video_data_lists, file_names):
        if not video_data_lists:
            st.warning("No video data found.")
            return

        fig = px.line(title=f'Video Posts Time Series on {file_names_str}')

        for i, video_data_list in enumerate(video_data_lists):
            if not video_data_list:
                st.info(f"No video data available for {file_names[i]}.")
            else:
                # Create DataFrame and group by date
                df = pd.DataFrame(video_data_list)
                df['date_posted'] = pd.to_datetime(df['date_posted'])
                df_grouped = df.groupby([df['date_posted'].dt.date]).size().reset_index(name='count')

                # Add line to the chart
                fig.add_scatter(x=df_grouped['date_posted'], y=df_grouped['count'], name=file_names[i])

        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Count of Video Posts')

        # Customize the chart size
        fig.update_layout(
            autosize=False,
            width=1000,   # Set the width in pixels
            height=600,  # Set the height in pixels
        )

        # Customize the font size
        fig.update_layout(
            font=dict(
                size=12,  # Set the font size for the entire chart
            )
        )

        st.plotly_chart(fig)

    # Get file paths in the "ytvideo" folder
    folder_path = 'ytvideo'
    file_paths = glob.glob(f"{folder_path}/*.json")
    file_names = [file_path.split('/')[-1].replace('_videos.json', '') for file_path in file_paths]

    # User interface
    st.title("YouTube Video Posts Time Series Analysis")
    selected_files = st.multiselect("Select issues", file_names, default=file_names[:2])
    start_date = st.date_input("Start Date", value=(datetime.today() - timedelta(days=30)), key='start_date_key')
    end_date = st.date_input("End Date", value=datetime.today(), key='end_date_key')

        # Combine the selected file names into a single string
    file_names_str = ', '.join(selected_files)

    # Load video data from selected files
    video_data_lists = [load_video_data([file_path]) for file_path in file_paths if file_path.split('/')[-1].replace('_videos.json', '') in selected_files]

    # Perform time series analysis and display results
    if not selected_files:
        st.warning("Please select at least one file.")
    elif start_date > end_date:
        st.warning("Start Date should be before End Date.")
    else:
        create_multiline_time_series_plot(video_data_lists, selected_files)


###############################################SNA ISSUE YOUTUBE###############
    
    
        import os
        import json
        import pandas as pd
        import streamlit as st
        import networkx as nx
        from pyvis.network import Network
        from datetime import datetime, timedelta
        import community

        # Set up the Streamlit app
        st.title("YouTube Video Social Network Analysis")
        
        # Get the file list from the ytvideo folder
        file_list = os.listdir("ytvideo")

        # Allow the user to select multiple files
        col1, col2, col3=st.columns([2,1,1])
        with col1:
            selected_files = st.multiselect("Select issues", file_list)

        # Function to load video data from files
        def load_video_data(file_paths):
            video_data_list = []
            for file_path in file_paths:
                with open(file_path, 'r') as file:
                    json_data = json.load(file)
                    for data in json_data:
                        # Check if the 'date_posted' field is present and has a valid format
                        if 'date_posted' in data and isinstance(data['date_posted'], str):
                            try:
                                datetime.fromisoformat(data['date_posted'])
                            except ValueError:
                                # Invalid date format, skip this data point
                                continue
                        else:
                            # 'date_posted' field is missing or not a string, skip this data point
                            continue

                        video_data_list.append(data)

            return video_data_list

        # Function to filter video data by date
        def filter_video_data_by_date(video_data_list, start_date, end_date):
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)

            filtered_data = [data for data in video_data_list if start_datetime <= datetime.fromisoformat(data['date_posted']) < end_datetime]
            return filtered_data

        # Load video data from selected files if any files are selected
        if selected_files:
            video_data_list = load_video_data([os.path.join("ytvideo", file) for file in selected_files])
        else:
            video_data_list = []

        with col2:
            # Allow the user to input start and end dates using Streamlit date inputs
            if video_data_list:
                start_date = st.date_input("Start Date", min([datetime.fromisoformat(data['date_posted']) for data in video_data_list]))
            else:
                # Set some default start date if no video data is available
                start_date = st.date_input("Start Date", datetime.now(), key='stdyti55ue')

        with col3:
            if video_data_list:
                end_date = st.date_input("End Date", max([datetime.fromisoformat(data['date_posted']) for data in video_data_list]))
            else:
                # Set some default end date if no video data is available
                end_date = st.date_input("End Date", datetime.now(), key='3ndtissu3inst')

        
        
        # Filter video data based on the selected date range if there are any files selected
            if selected_files:
                filtered_data = filter_video_data_by_date(video_data_list, start_date, end_date)
            else:
                filtered_data = []

        
        if not filtered_data:
            st.write("")
        else:
            colytissu1, colytissu2, colytissu3, colytissu4=st.tabs(['Social Network','Main Actors', 'Bridging Actors', 'Supporting Actors'])

            with colytissu1:

                
                G = nx.DiGraph()

                # Add nodes for channel titles
                G.add_nodes_from(data['channel_title'] for data in filtered_data)

                # For comments to channel relationship
                for data in filtered_data:
                    G.add_nodes_from(comment['username'] for comment in data['comments'])
                    for comment in data['comments']:
                        # Add the edge in the direction from comment to channel
                        G.add_edge(comment['username'], data['channel_title'], arrows=True, relationship='comment')

                # For replies to comment relationship
                for data in filtered_data:
                    for comment in data['comments']:
                        G.add_nodes_from(reply['username'] for reply in comment['replies'])
                        for reply in comment['replies']:
                            # Add the edge in the direction from reply to comment
                            G.add_edge(reply['username'], comment['username'], arrows=True, relationship='reply')

                # For tags to channel_title relationship
                for data in filtered_data:
                    # Add tags as nodes and create edges from tags to channel_title
                    if 'tags' in data and isinstance(data['tags'], list):
                        for tag in data['tags']:
                            # Add the tag node if it doesn't already exist
                            G.add_node(tag, color="#228B22")  # You can change the color for the tag nodes here

                            # Add the edge in the direction from tag to channel_title with the relationship 'tag'
                            G.add_edge(data['channel_title'], tag, arrows=True, relationship='tag')

                # Function to get clusters (strongly connected components) from the directed graph
                    def get_clusters(digraph):
                        return list(nx.strongly_connected_components(digraph))

                    # Function to get cluster labels (map cluster ID to nodes)
                    def get_cluster_labels(clusters):
                        cluster_labels = {}
                        for cluster_id, cluster in enumerate(clusters):
                            for node in cluster:
                                cluster_labels[node] = f"Cluster {cluster_id}"
                        return cluster_labels

                    # Get the clusters (strongly connected components) from the directed graph
                    clusters = get_clusters(G)

                    # Get cluster labels (map cluster ID to nodes)
                    cluster_labels = get_cluster_labels(clusters)


                # Use Pyvis to visualize the graph with clustered nodes
                nt = Network(height="900px", width="100%", bgcolor="#fff", font_color="darkgrey", directed=True)

                # Function to get the cluster ID for a node
                def get_node_cluster_id(node, clusters):
                    for cluster_id, cluster in enumerate(clusters):
                        if node in cluster:
                            return cluster_id
                    return None

                # Add nodes and edges to the Pyvis network object with cluster information
                for node in G.nodes():
                    cluster_id = get_node_cluster_id(node, clusters)
                    if node in (data['channel_title'] for data in filtered_data):
                        nt.add_node(node, label=node, color="#FF4500", title=f"Cluster: {cluster_id}")  # Set channel_title node color
                    elif node in (comment['username'] for data in filtered_data for comment in data['comments']):
                        nt.add_node(node, label=node, color="#1E90FF", title=f"Cluster: {cluster_id}")  # Set comment username node color
                    elif node in (reply['username'] for data in filtered_data for comment in data['comments'] for reply in comment['replies']):
                        nt.add_node(node, label=node, color="#BAEAFE", title=f"Cluster: {cluster_id}")  # Set reply username node color
                    else:
                        nt.add_node(node, label=node, color="#89CB09", title=f"Cluster: {cluster_id}")  # Set tag node color

                for edge in G.edges():
                    nt.add_edge(edge[0], edge[1], label=G.edges[edge]['relationship'])

                # Save the network graph as an HTML file
                nt.save_graph("html_files/ytissuesna.html")



                
                with open('html_files/ytissuesna.html', 'r') as f:
                    html_string = f.read()
                    st.components.v1.html(html_string, height=910, scrolling=True)



################################################################
            with colytissu2:
                import pandas as pd
                import matplotlib.pyplot as plt

                # Calculate degree centrality of nodes in the original graph
                degree_centrality = nx.degree_centrality(G)

                # Create a new graph for degree centrality visualization
                degree_graph = nx.DiGraph()

                # Add nodes with sizes proportional to degree centrality values
                for node, centrality in degree_centrality.items():
                    size = centrality * 500  # You can adjust the multiplier to control the size of nodes
                    degree_graph.add_node(node, size=size)

                # Add edges from the original graph to the degree graph
                for edge in G.edges():
                    degree_graph.add_edge(edge[0], edge[1])

                # Visualize the new graph with node sizes reflecting degree centrality
                nt_degree = Network(height="900px", width="100%", bgcolor="#fff", font_color="darkgrey", directed=True)

                # Add nodes and edges to the Pyvis network object with cluster information
                for node in degree_graph.nodes():
                    cluster_id = get_node_cluster_id(node, clusters)
                    if node in (data['channel_title'] for data in filtered_data):
                        nt_degree.add_node(node, label=node, color="#FF4500", title=f"Cluster: {cluster_id}", size=degree_graph.nodes[node]['size'])  # Set channel_title node color
                    elif node in (comment['username'] for data in filtered_data for comment in data['comments']):
                        nt_degree.add_node(node, label=node, color="#1E90FF", title=f"Cluster: {cluster_id}", size=degree_graph.nodes[node]['size'])  # Set comment username node color
                    elif node in (reply['username'] for data in filtered_data for comment in data['comments'] for reply in comment['replies']):
                        nt_degree.add_node(node, label=node, color="#BAEAFE", title=f"Cluster: {cluster_id}", size=degree_graph.nodes[node]['size'])  # Set reply username node color
                    else:
                        nt_degree.add_node(node, label=node, color="#89CB09", title=f"Cluster: {cluster_id}", size=degree_graph.nodes[node]['size'])  # Set tag node color

                for edge in degree_graph.edges():
                    nt_degree.add_edge(edge[0], edge[1])

                # Save the degree centrality network graph as an HTML file
                nt_degree.save_graph("html_files/ytdegree_centrality.html")

                # Create a pandas DataFrame of the top ten nodes ranked by degree centrality
                top_ten_degree = pd.DataFrame({"Node": list(degree_centrality.keys()), "Degree Centrality": list(degree_centrality.values())})
                top_ten_degree = top_ten_degree.sort_values(by="Degree Centrality", ascending=False).head(10)

                # Create a bar chart of the top ten nodes' degree centrality values
                plt.figure(figsize=(8, 6))
                plt.bar(top_ten_degree["Node"], top_ten_degree["Degree Centrality"])
                plt.xlabel("Node")
                plt.ylabel("Degree Centrality")
                plt.title("Top 10 Nodes by Degree Centrality")
                plt.xticks(rotation=45)
                plt.tight_layout()

                # Display the degree centrality network graph and the pandas DataFrame with the bar chart side by side in Streamlit
                # Display the degree centrality network graph and the pandas DataFrame with the bar chart side by side in Streamlit
                with st.container():
                    col_graph, col_data = st.columns([2, 1])

                    with col_graph:
                        with st.spinner("Loading Graph..."):
                            with open('html_files/ytdegree_centrality.html', 'r') as f:
                                degree_html_string = f.read()
                                st.components.v1.html(degree_html_string, height=910, scrolling=True)

                    with col_data:
                        with st.spinner("Loading Top 10 Degree Centrality Data..."):
                            st.write("### Top 10 Nodes by Degree Centrality")
                            st.dataframe(top_ten_degree)
                            st.pyplot(plt)



#############################BETWEENESS ############################
            with colytissu3:
                import pandas as pd
                import matplotlib.pyplot as plt

                # Calculate betweenness centrality of nodes in the original graph
                betweenness_centrality = nx.betweenness_centrality(G)

                # Create a new graph for betweenness centrality visualization
                betweenness_graph = nx.DiGraph()

                # Add nodes with sizes proportional to betweenness centrality values
                for node, centrality in betweenness_centrality.items():
                    size = centrality * 500  # You can adjust the multiplier to control the size of nodes
                    betweenness_graph.add_node(node, size=size)

                # Add edges from the original graph to the betweenness graph
                for edge in G.edges():
                    betweenness_graph.add_edge(edge[0], edge[1])

                # Visualize the new graph with node sizes reflecting betweenness centrality
                nt_betweenness = Network(height="900px", width="100%", bgcolor="#fff", font_color="darkgrey", directed=True)

                # Add nodes and edges to the Pyvis network object with cluster information
                for node in betweenness_graph.nodes():
                    cluster_id = get_node_cluster_id(node, clusters)
                    if node in (data['channel_title'] for data in filtered_data):
                        nt_betweenness.add_node(node, label=node, color="#FF4500", title=f"Cluster: {cluster_id}", size=betweenness_graph.nodes[node]['size'])  # Set channel_title node color
                    elif node in (comment['username'] for data in filtered_data for comment in data['comments']):
                        nt_betweenness.add_node(node, label=node, color="#1E90FF", title=f"Cluster: {cluster_id}", size=betweenness_graph.nodes[node]['size'])  # Set comment username node color
                    elif node in (reply['username'] for data in filtered_data for comment in data['comments'] for reply in comment['replies']):
                        nt_betweenness.add_node(node, label=node, color="#BAEAFE", title=f"Cluster: {cluster_id}", size=betweenness_graph.nodes[node]['size'])  # Set reply username node color
                    else:
                        nt_betweenness.add_node(node, label=node, color="#89CB09", title=f"Cluster: {cluster_id}", size=betweenness_graph.nodes[node]['size'])  # Set tag node color

                for edge in betweenness_graph.edges():
                    nt_betweenness.add_edge(edge[0], edge[1])

                # Save the betweenness centrality network graph as an HTML file
                nt_betweenness.save_graph("html_files/ytbetweenness_centrality.html")

                # Calculate betweenness centrality of nodes in the original graph
                betweenness_centrality = nx.betweenness_centrality(G)

                # Create a pandas DataFrame of the top ten nodes ranked by betweenness centrality
                top_ten_betweenness = pd.DataFrame({"Node": list(betweenness_centrality.keys()), "Betweenness Centrality": list(betweenness_centrality.values())})
                top_ten_betweenness = top_ten_betweenness.sort_values(by="Betweenness Centrality", ascending=False).head(10)

                # Create a bar chart of the top ten nodes' betweenness centrality values
                plt.figure(figsize=(8, 6))
                plt.bar(top_ten_betweenness["Node"], top_ten_betweenness["Betweenness Centrality"])
                plt.xlabel("Node")
                plt.ylabel("Betweenness Centrality")
                plt.title("Top 10 Nodes by Betweenness Centrality")
                plt.xticks(rotation=45)
                plt.tight_layout()

                
                # Display the new betweenness centrality network graph, the pandas DataFrame, and the bar chart below it in Streamlit
                with st.container():
                    col_graph, col_data = st.columns([2, 1])

                    with col_graph:
                        with st.spinner("Loading Betweenness Centrality Graph..."):
                            with open('html_files/ytbetweenness_centrality.html', 'r') as f:
                                betweenness_html_string = f.read()
                                st.components.v1.html(betweenness_html_string, height=910, scrolling=True)

                    with col_data:
                        with st.spinner("Loading Top 10 Betweenness Centrality Data..."):
                            st.write("### Top 10 Nodes by Betweenness Centrality")
                            st.dataframe(top_ten_betweenness)
                            st.pyplot(plt)





################################CLOSENESS###########################################


            with colytissu4:
                import pandas as pd
                import matplotlib.pyplot as plt

                # ... (previous code remains the same)

                # Calculate closeness centrality of nodes in the original graph
                closeness_centrality = nx.closeness_centrality(G)

                # Create a new graph for closeness centrality visualization
                closeness_graph = nx.DiGraph()

                # Add nodes with sizes proportional to closeness centrality values
                for node, centrality in closeness_centrality.items():
                    size = centrality * 500  # You can adjust the multiplier to control the size of nodes
                    closeness_graph.add_node(node, size=size)

                # Add edges from the original graph to the closeness graph
                for edge in G.edges():
                    closeness_graph.add_edge(edge[0], edge[1])

                # Visualize the new graph with node sizes reflecting closeness centrality
                nt_closeness = Network(height="900px", width="100%", bgcolor="#fff", font_color="darkgrey", directed=True)

                # Add nodes and edges to the Pyvis network object with cluster information
                for node in closeness_graph.nodes():
                    cluster_id = get_node_cluster_id(node, clusters)
                    if node in (data['channel_title'] for data in filtered_data):
                        nt_closeness.add_node(node, label=node, color="#FF4500", title=f"Cluster: {cluster_id}", size=closeness_graph.nodes[node]['size'])  # Set channel_title node color
                    elif node in (comment['username'] for data in filtered_data for comment in data['comments']):
                        nt_closeness.add_node(node, label=node, color="#1E90FF", title=f"Cluster: {cluster_id}", size=closeness_graph.nodes[node]['size'])  # Set comment username node color
                    elif node in (reply['username'] for data in filtered_data for comment in data['comments'] for reply in comment['replies']):
                        nt_closeness.add_node(node, label=node, color="#BAEAFE", title=f"Cluster: {cluster_id}", size=closeness_graph.nodes[node]['size'])  # Set reply username node color
                    else:
                        nt_closeness.add_node(node, label=node, color="#89CB09", title=f"Cluster: {cluster_id}", size=closeness_graph.nodes[node]['size'])  # Set tag node color

                for edge in closeness_graph.edges():
                    nt_closeness.add_edge(edge[0], edge[1])

                # Save the closeness centrality network graph as an HTML file
                nt_closeness.save_graph("html_files/yt_closeness_centrality.html")

                # Calculate closeness centrality of nodes in the original graph
                closeness_centrality = nx.closeness_centrality(G)

                # Create a pandas DataFrame of the top ten nodes ranked by closeness centrality
                top_ten_closeness = pd.DataFrame({"Node": list(closeness_centrality.keys()), "Closeness Centrality": list(closeness_centrality.values())})
                top_ten_closeness = top_ten_closeness.sort_values(by="Closeness Centrality", ascending=False).head(10)

                # Create a bar chart of the top ten nodes' closeness centrality values
                plt.figure(figsize=(8, 6))
                plt.bar(top_ten_closeness["Node"], top_ten_closeness["Closeness Centrality"])
                plt.xlabel("Node")
                plt.ylabel("Closeness Centrality")
                plt.title("Top 10 Nodes by Closeness Centrality")
                plt.xticks(rotation=45)
                plt.tight_layout()

                # Display the new closeness centrality network graph, the pandas DataFrame, and the bar chart below it in Streamlit
                with st.container():
                    col_graph, col_data = st.columns([2, 1])

                    with col_graph:
                        with st.spinner("Loading Closeness Centrality Graph..."):
                            with open('html_files/yt_closeness_centrality.html', 'r') as f:
                                closeness_html_string = f.read()
                                st.components.v1.html(closeness_html_string, height=910, scrolling=True)

                    with col_data:
                        with st.spinner("Loading Top 10 Closeness Centrality Data..."):
                            st.write("### Top 10 Nodes by Closeness Centrality")
                            st.dataframe(top_ten_closeness)
                            st.pyplot(plt)

            

#############################HASTAG VIDEOS ISSUE ################
    import json
    import glob
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    import datetime as dt

    # Function to extract hashtags from a video's tags
    def extract_hashtags(tags):
        return tags

    # Function to load video data from files
    def load_video_data(file_paths):
        video_data = []
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                video_data.extend(json_data)
        return video_data

    def perform_hashtag_analysis(file_paths, start_date, end_date):
        # Load video data from files
        video_data = load_video_data(file_paths)

        # Filter video data based on date range
        filtered_video_data = filter_data_by_date(video_data, start_date, end_date)

        # Extract hashtags from video data
        hashtags = []
        for video in filtered_video_data:
            hashtags.extend(extract_hashtags(video['tags']))

        # Create a DataFrame with hashtag counts
        hashtag_counts = pd.Series(hashtags).value_counts().reset_index()
        hashtag_counts.columns = ['Hashtag', 'Count']

        return hashtag_counts

    # Get file paths in the "ytvideo" folder
    folder_path = 'ytvideo'
    file_paths = glob.glob(f"{folder_path}/*.json")
    file_names = [file_path.split('/')[-1].replace('_videos.json', '') for file_path in file_paths]

    # User interface
    st.title("Hashtag Analysis of YouTube Videos")
    col1,col2,col3=st.columns([2,1,1])
    with col1:
        selected_files = st.multiselect("Select issues", file_names)
    with col2:
        start_date = st.date_input("Start Date", key='yt_start_date')
    with col3:
        end_date = st.date_input("End Date", key='yt_end_date')

    def filter_data_by_date(video_data, start_date, end_date):
        filtered_data = []
        for video in video_data:
            date_posted = dt.datetime.strptime(video['date_posted'], '%Y-%m-%d').replace(tzinfo=None)
            if start_date <= date_posted <= end_date:
                filtered_data.append(video)
        return filtered_data

    if st.button("Perform Analysis"):
        if not selected_files:
            st.warning("Please select at least one file.")
        elif start_date > end_date:
            st.warning("Start Date should be before End Date.")
        else:
            file_paths_selected = [file_paths[file_names.index(file_name)] for file_name in selected_files]
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)

            # Create columns for displaying charts
            cols = st.columns(len(file_paths_selected))

            for i, file_path in enumerate(file_paths_selected):
                # st.subheader(f"Analysis for keyword: {selected_files[i]}")
                hashtag_counts = perform_hashtag_analysis([file_path], start_date, end_date)

                if len(hashtag_counts) > 0:
                    fig = px.bar(hashtag_counts, x='Hashtag', y='Count', title=f'Hashtag Analysis for keyword: {selected_files[i]}')

                    # Customize the chart size
                    fig.update_layout(
                        autosize=False,
                        width=700,   # Set the width in pixels
                        height=600,  # Set the height in pixels
                    )

                    cols[i].plotly_chart(fig)
                else:
                    cols[i].info("No hashtags found within the specified date range.")

#####################################################

    import json
    import glob
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    from datetime import datetime, timedelta
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk

    nltk.download('stopwords')
    nltk.download('vader_lexicon')

    # Load stopwords for Bahasa Indonesia
    stop_words_indonesian = set(stopwords.words('indonesian'))

    # Function to load video data from files
    def load_video_data(file_paths):
        video_data_list = []
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                video_data_list.extend(json_data)
        return video_data_list

    # Function to filter video data by date
    def filter_video_data_by_date(video_data_list, start_date, end_date):
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
        filtered_data = [data for data in video_data_list if start_datetime <= datetime.fromisoformat(data['date_posted']) < end_datetime]
        return filtered_data

   # Function to perform sentiment analysis on comments or reply texts
    def perform_sentiment_analysis(texts):
        sia = SentimentIntensityAnalyzer()
        sentiments = []
        for text in texts:
            # Remove stopwords for Bahasa Indonesia
            text_words = [word for word in text.lower().split() if word not in stop_words_indonesian]
            processed_text = ' '.join(text_words)
            sentiment_score = sia.polarity_scores(processed_text)['compound']
            if sentiment_score > 0:
                sentiments.append('Positive')
            elif sentiment_score < 0:
                sentiments.append('Negative')
            else:
                sentiments.append('Neutral')
        return sentiments

    # Function to create pie chart for sentiment distribution
    def create_pie_chart(sentiments, title):
        sentiment_counts = pd.Series(sentiments).value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig = px.pie(sentiment_counts, values='Count', names='Sentiment', title=title)
        st.plotly_chart(fig)

    # Streamlit UI
    st.title("YouTube Video Sentiment Analysis")
    folder_path = 'ytvideo'
    file_paths = glob.glob(f"{folder_path}/*.json")
    file_names = [file_path.split('/')[-1].replace('_videos.json', '') for file_path in file_paths]  # Remove the '_videos.json' extension
    selected_files = st.multiselect("Select files", file_names, default=file_names[:2], key='sel5entytvid')
    start_date = st.date_input("Start Date", value=(datetime.today() - timedelta(days=30)), key='stdytv1d')
    end_date = st.date_input("End Date", value=datetime.today(), key='edytv5')

    # Load video data from selected files
    selected_file_paths = [file_path for file_path in file_paths if file_path.split('/')[-1].replace('_videos.json', '') in selected_files]  # Remove the '_videos.json' extension
    video_data_lists = [load_video_data([file_path]) for file_path in selected_file_paths]

    # Perform sentiment analysis on comments and reply snippets for each selected file
    for i, file_name in enumerate(selected_files):
        st.subheader(f"Sentiment Analysis for {file_name}")
        video_data_list = video_data_lists[i]
        
        # Filter video data by date
        video_data_filtered = filter_video_data_by_date(video_data_list, start_date, end_date)
        
        # Perform sentiment analysis on comments and reply snippets
        comments = [data['comments'] for data in video_data_filtered]
        comments = [comment for sublist in comments for comment in sublist]  # Flatten list of comments

        # Extract reply dictionaries from comments
        replies = [reply for comment in comments for reply in comment['replies']]

        # Extract comment and reply texts
        comment_texts = [comment['text'] for comment in comments]
        reply_texts = [reply['text'] for reply in replies]

        comment_sentiments = perform_sentiment_analysis(comment_texts)
        reply_sentiments = perform_sentiment_analysis(reply_texts)

        # Dynamically create columns for each selected file and display pie charts side by side
        cols = st.columns(2)
        with cols[0]:
            create_pie_chart(comment_sentiments, f"Sentiment Analysis for Comments - {file_name}")
        with cols[1]:
            create_pie_chart(reply_sentiments, f"Sentiment Analysis for Reply Snippets - {file_name}")  


#####################################################

with tab3:
    st.header("DATA")

    ytscolta, ytscoltb = st.columns([2, 2])
    with ytscolta:

        # Set up the YouTube Data API client
        def get_youtube_client(api_key):
            return build('youtube', 'v3', developerKey=api_key)

        def search_videos(youtube, query, start_date, end_date):
            video_data_list = []  # List to store the video data

            try:
                search_response = youtube.search().list(
                    q=query,
                    part='id,snippet',
                    type='video',
                    maxResults=10,  # Set the maximum number of results you want to retrieve
                    publishedAfter=start_date.isoformat() + "T00:00:00Z",  # Convert start date to ISO format
                    publishedBefore=end_date.isoformat() + "T23:59:59Z"  # Convert end date to ISO format
                ).execute()

                # Process the search results
                for search_result in search_response.get('items', []):
                    video_id = search_result['id']['videoId']
                    video_title = search_result['snippet']['title']
                    video_description = search_result['snippet']['description']

                    # Retrieve video statistics and recording details
                    video_response = youtube.videos().list(
                        part='statistics,contentDetails,snippet,recordingDetails',
                        id=video_id
                    ).execute()

                    # Extract duration, views, subscriber count, date posted, like count, and dislike count
                    video_duration = video_response['items'][0]['contentDetails']['duration']
                    video_views = video_response['items'][0]['statistics']['viewCount']
                    video_comment_count = video_response['items'][0]['statistics'].get('commentCount', 'N/A')
                    video_like_count = video_response['items'][0]['statistics'].get('likeCount', 'N/A')
                    video_dislike_count = video_response['items'][0]['statistics'].get('dislikeCount', 'N/A')
                    video_tags = video_response['items'][0]['snippet'].get('tags', [])
                    video_date_posted = datetime.strptime(video_response['items'][0]['snippet']['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").date()

                    # Retrieve user/channel information
                    channel_id = search_result['snippet']['channelId']
                    channel_response = youtube.channels().list(
                        part='snippet,statistics',
                        id=channel_id
                    ).execute()

                    # Extract channel title, subscriber count, and username
                    channel_title = channel_response['items'][0]['snippet']['title']
                    channel_subscriber_count = channel_response['items'][0]['statistics'].get('subscriberCount', 'N/A')
                    channel_username = channel_response['items'][0]['snippet'].get('customUrl', 'N/A')

                    # Retrieve channel profile thumbnails
                    channel_thumbnails = channel_response['items'][0]['snippet'].get('thumbnails', {})
                    channel_profile_default = channel_thumbnails.get('default', {}).get('url', 'N/A')
                    channel_profile_medium = channel_thumbnails.get('medium', {}).get('url', 'N/A')
                    channel_profile_high = channel_thumbnails.get('high', {}).get('url', 'N/A')

                    # Retrieve channel location data
                    try:
                        channel_location = channel_response['items'][0]['brandingSettings']['channel'].get('country', 'N/A')
                    except KeyError:
                        channel_location = 'N/A'

                    # Video URL
                    video_url = f"https://www.youtube.com/watch?v={video_id}"

                    # Retrieve comments for the video
                    comments_response = youtube.commentThreads().list(
                        part='snippet',
                        videoId=video_id,
                        maxResults=10  # Set the maximum number of comments you want to retrieve
                    ).execute()

                    # Extract comment data
                    comments = []
                    for comment_result in comments_response.get('items', []):
                        comment_text = comment_result['snippet']['topLevelComment']['snippet']['textDisplay']
                        comment_username = comment_result['snippet']['topLevelComment']['snippet']['authorDisplayName']
                        comment_date = datetime.strptime(comment_result['snippet']['topLevelComment']['snippet']['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").date()
                        comment_profile_picture = comment_result['snippet']['topLevelComment']['snippet']['authorProfileImageUrl']
                        comment_like_count = comment_result['snippet']['topLevelComment']['snippet'].get('likeCount', 'N/A')
                        comment_dislike_count = comment_result['snippet']['topLevelComment']['snippet'].get('dislikeCount', 'N/A')
                        comment_data = {
                            'text': comment_text,
                            'username': comment_username,
                            'date_posted': str(comment_date),
                            'profile_picture': comment_profile_picture,
                            'like_count': comment_like_count,
                            'dislike_count': comment_dislike_count,
                            'replies': []
                        }

                        # Retrieve comment replies
                        reply_response = youtube.comments().list(
                            part='snippet',
                            parentId=comment_result['snippet']['topLevelComment']['id'],
                            maxResults=5  # Set the maximum number of replies you want to retrieve
                        ).execute()

                        # Extract reply data
                        for reply_result in reply_response.get('items', []):
                            reply_text = reply_result['snippet']['textDisplay']
                            reply_username = reply_result['snippet']['authorDisplayName']
                            reply_date = datetime.strptime(reply_result['snippet']['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").date()
                            reply_like_count = reply_result['snippet'].get('likeCount', 'N/A')
                            reply_dislike_count = reply_result['snippet'].get('dislikeCount', 'N/A')
                            reply_profile_picture = reply_result['snippet']['authorProfileImageUrl']
                            reply_data = {
                                'text': reply_text,
                                'username': reply_username,
                                'date_posted': str(reply_date),
                                'like_count': reply_like_count,
                                'dislike_count': reply_dislike_count,
                                'profile_picture': reply_profile_picture
                            }
                            comment_data['replies'].append(reply_data)

                        comments.append(comment_data)

                    # Retrieve video location data
                    video_location = video_response['items'][0]['recordingDetails'].get('location', {})
                    video_latitude = video_location.get('latitude', 'N/A')
                    video_longitude = video_location.get('longitude', 'N/A')

                    # Save video details in the list
                    video_data = {
                        'video_id': video_id,
                        'title': video_title,
                        'description': video_description,
                        'duration': video_duration,
                        'views': video_views,
                        'comment_count': video_comment_count,
                        'like_count': video_like_count,
                        'dislike_count': video_dislike_count,
                        'tags': video_tags,
                        'date_posted': str(video_date_posted),
                        'subscriber_count': channel_subscriber_count,
                        'channel_title': channel_title,
                        'channel_username': channel_username,
                        'channel_profile_default': channel_profile_default,
                        'channel_profile_medium': channel_profile_medium,
                        'channel_profile_high': channel_profile_high,
                        'channel_location': channel_location,
                        'url': video_url,
                        'comments': comments,
                        'latitude': video_latitude,
                        'longitude': video_longitude
                    }
                    video_data_list.append(video_data)

                # Save video data as JSON
                keyword = query.lower().replace(" ", "_")
                filename = f"ytvideo/{keyword}_videos.json"
                os.makedirs("ytvideo", exist_ok=True)

                with open(filename, 'w') as json_file:
                    json.dump(video_data_list, json_file, indent=4)

                # Display the video details and comments
                for video_data in video_data_list:
                    st.write("Video ID:", video_data['video_id'])
                    st.write("Title:", video_data['title'])
                    st.write("Description:", video_data['description'])
                    st.write("Duration:", video_data['duration'])
                    st.write("Views:", video_data['views'])
                    st.write("Comment Count:", video_data['comment_count'])
                    st.write("Like Count:", video_data['like_count'])
                    st.write("Dislike Count:", video_data['dislike_count'])
                    st.write("Tags:", ", ".join(video_data['tags']) if video_data['tags'] else "N/A")
                    st.write("Date Posted:", video_data['date_posted'])
                    st.write("Channel Title:", video_data['channel_title'])
                    st.write("Channel Subscriber Count:", video_data['subscriber_count'])
                    st.write("Channel Username:", video_data['channel_username'])
                    st.write("Channel Location:", video_data['channel_location'])
                    st.image(video_data['channel_profile_default'])
                    # st.image(video_data['channel_profile_medium'], "Medium Profile Thumbnail")
                    # st.image(video_data['channel_profile_high'], "High Profile Thumbnail")
                    st.video(video_data['url'])

                    # Location data
                    st.write("Latitude:", video_data['latitude'])
                    st.write("Longitude:", video_data['longitude'])

                    st.write("Comments:")
                    for comment in video_data['comments']:
                        st.image(comment['profile_picture'])
                        st.write("Text:", comment['text'])
                        st.write("Username:", comment['username'])
                        st.write("Date Posted:", comment['date_posted'])
                        st.write("Likes:", comment['like_count'])
                        st.write("Dislikes:", comment['dislike_count'])
                        st.write("Replies:")
                        for reply in comment['replies']:
                            st.image(reply['profile_picture'])
                            st.write("- Reply:")
                            st.write("  Text:", reply['text'])
                            st.write("  Username:", reply['username'])
                            st.write("  Date Posted:", reply['date_posted'])
                            st.write("  Likes:", reply['like_count'])
                            st.write("  Dislikes:", reply['dislike_count'])
                    st.write("---")

            except HttpError as e:
                st.write(f"An HTTP error {e.resp.status} occurred:\n{json.loads(e.content)['error']['message']}")




        # Set your YouTube Data API key
        api_key = 'AIzaSyDJk1gJGYQmgEJdUxYmzRZ9gerTXW3kxQw'

        # Set up the YouTube client
        youtube_client = get_youtube_client(api_key)

            # Create a Streamlit form
        st.title("YouTube Video Search")
        search_query = st.text_input("Enter your search query:")
        start_date = st.date_input("Start Date", key='yt4ccstd')
        end_date = st.date_input("End Date", key='ytacc3d')

        # Perform the search when the form is submitted
        if st.button("Search"):
            search_videos(youtube_client, search_query, start_date, end_date)




    ######################################
    with ytscoltb:
        

        # Set up the YouTube Data API client
        # api_key = os.environ.get('YOUTUBE_API_KEY')
        youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)

        # Function to fetch channel ID from account/channel name
        def get_channel_id(channel_name):
            response = youtube.search().list(part='id', q=channel_name, maxResults=50, type='channel').execute()
            if response['items']:
                return response['items'][0]['id']['channelId']
            else:
                return None

        # Function to fetch channel details
        def get_channel_details(channel_id):
            response = youtube.channels().list(part='snippet,statistics', id=channel_id).execute()
            channel = response['items'][0]
            return channel

        # Function to fetch video details
        def get_video_details(video_id):
            response = youtube.videos().list(part='snippet,statistics', id=video_id).execute()
            video = response['items'][0]
            return video

        # Function to fetch comments for a video
        def get_comments(video_id):
            response = youtube.commentThreads().list(part='snippet', videoId=video_id).execute()
            comments = response['items']
            return comments

        # Save the result in a JSON file
        def save_result(result, channel_name):
            folder_name = "ytaccscrap"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # Generate the file title
            file_title = f"{channel_name}_data.json"

            # Save the result in the JSON file
            file_path = os.path.join(folder_name, file_title)
            with open(file_path, 'w') as f:
                json.dump(result, f)

            st.success(f"Result saved successfully: {file_path}")

        # Streamlit app
        def youtube_scraper_app(channel_name, max_results, start_date, end_date):
            # Convert account/channel name to channel ID
            channel_id = get_channel_id(channel_name)

            if channel_id:
                # Fetch and display channel details
                channel = get_channel_details(channel_id)
                st.subheader("Channel Details")
                st.write("Name:", channel['snippet']['title'])
                st.image(channel['snippet']['thumbnails']['default']['url'])
                st.write("Joined:", channel['snippet']['publishedAt'])
                st.write("Description:", channel['snippet']['description'])
                st.write("Subscriber Count:", channel['statistics']['subscriberCount'])
                st.write("Video Count:", channel['statistics']['videoCount'])

                # Convert start and end dates to ISO 8601 format with timezone offset
                start_date_iso = (start_date - timedelta(days=1)).isoformat() + "Z"
                end_date_iso = (end_date + timedelta(days=1)).isoformat() + "Z"

                # Fetch videos from the channel within the specified date range
                response = youtube.search().list(
                    part='snippet',
                    channelId=channel_id,
                    maxResults=max_results,
                    order='date',
                    publishedAfter=start_date_iso,
                    publishedBefore=end_date_iso
                ).execute()
                videos = response['items']

                if videos:
                    result = {'channel_details': channel, 'videos': []}

                    st.subheader("Video List")
                    for video in videos:
                        video_id = video['id']['videoId']
                        video_details = get_video_details(video_id)
                        st.write("Title:", video_details['snippet']['title'])
                        st.write("Published At:", video_details['snippet']['publishedAt'])
                        st.write("Description:", video_details['snippet']['description'])
                        st.write("View Count:", video_details['statistics']['viewCount'])
                        st.write("Like Count:", video_details['statistics']['likeCount'])
                        st.write("Comment Count:", video_details['statistics']['commentCount'])

                        video_data = {
                            'title': video_details['snippet']['title'],
                            'published_at': video_details['snippet']['publishedAt'],
                            'description': video_details['snippet']['description'],
                            'view_count': video_details['statistics']['viewCount'],
                            'like_count': video_details['statistics']['likeCount'],
                            'comment_count': video_details['statistics']['commentCount'],
                            'video_url': f"https://www.youtube.com/watch?v={video_id}",
                            'comments': []
                        }

                        # Display video URL
                        st.markdown(f"**Video URL:** [{video_data['video_url']}]({video_data['video_url']})")

                        # Embed YouTube video using iframe
                        iframe = f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>'
                        st.components.v1.html(iframe, width=560, height=315)

                      # Fetch and display comments for the video
                        comments = get_comments(video_id)
                        if comments:
                            st.subheader("Comments")
                            for comment in comments:
                                comment_snippet = comment['snippet']['topLevelComment']['snippet']
                                st.write("Username:", comment_snippet['authorDisplayName'])
                                st.write("Comment:", comment_snippet['textDisplay'])
                                st.write("Published At:", comment_snippet['publishedAt'])

                                comment_data = {
                                    'username': comment_snippet['authorDisplayName'],
                                    'comment': comment_snippet['textDisplay'],
                                    'published_at': comment_snippet['publishedAt'],
                                    'replies': []  # Initialize an empty list for storing replies
                                }

                                # Fetch profile picture
                                profile_picture_url = comment_snippet['authorProfileImageUrl']
                                response = requests.get(profile_picture_url)
                                if response.status_code == 200:
                                    profile_picture = response.content
                                    profile_picture_base64 = base64.b64encode(profile_picture).decode('utf-8')  # Convert to base64-encoded string
                                    st.image(profile_picture, width=50)  # Display the profile picture

                                comment_data['profile_picture'] = profile_picture_base64

                                video_data['comments'].append(comment_data)

                                # Fetch and display replies for the comment
                                reply_response = youtube.comments().list(
                                    part='snippet',
                                    parentId=comment['id'],
                                    maxResults=5  # Set the maximum number of replies you want to retrieve
                                ).execute()

                                # Extract reply data
                                for reply_result in reply_response.get('items', []):
                                    reply_text = reply_result['snippet']['textDisplay']
                                    reply_username = reply_result['snippet']['authorDisplayName']
                                    reply_date = datetime.strptime(reply_result['snippet']['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").date()
                                    reply_like_count = reply_result['snippet'].get('likeCount', 'N/A')
                                    reply_dislike_count = reply_result['snippet'].get('dislikeCount', 'N/A')
                                    reply_profile_picture = reply_result['snippet']['authorProfileImageUrl']
                                    
                                    # Fetch profile picture for reply owner
                                    response = requests.get(reply_profile_picture)
                                    if response.status_code == 200:
                                        reply_owner_profile_picture = response.content
                                        st.image(reply_owner_profile_picture, width=50)  # Display the profile picture

                                    reply_data = {
                                        'text': reply_text,
                                        'username': reply_username,
                                        'date_posted': str(reply_date),
                                        'like_count': reply_like_count,
                                        'dislike_count': reply_dislike_count,
                                        'profile_picture': reply_owner_profile_picture
                                    }

                                    comment_data['replies'].append(reply_data)

                        result['videos'].append(video_data)

                    # Save the result in a JSON file
                    save_result(result, channel_name, start_date, end_date)

                else:
                    st.write("No videos found for the channel")
            else:
                st.write("Channel not found")

        st.title("YouTube Account Search")

        # Text input for YouTube account/channel name
        channel_name = st.text_input("Enter YouTube account/channel name", "Example: OpenAI")

        # Slider for specifying the number of videos to fetch
        max_results = 5

        # Date inputs for specifying the video published date range
        start_date = st.date_input("Start Date", key='ytaccsd')
        end_date = st.date_input("End Date", key='ytacced')

        # Convert start and end dates to datetime objects
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())

        # Call the scraper function when an account/channel name is provided
        if st.button("Search", key="accyts"):
            youtube_scraper_app(channel_name, max_results, start_date, end_date)





