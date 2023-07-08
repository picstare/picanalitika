import streamlit as st
from play_app import run_scraper
from datetime import datetime, timedelta
import asyncio
import dateparser
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

st.set_page_config(page_title="Picanalitika | Facebook Analysis", layout="wide")
hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div.vis-configuration-wrapper{display: block; width: 400px;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

###########################################################

a, b = st.columns([1, 10])

with a:
    st.text("")
    st.image("img/fblog.png", width=100)
with b:
    st.title("Facebook Analysis")



#############################################################
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

    def load_posts_data(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data


    def filter_posts_by_date(posts_data, start_date, end_date):
        filtered_data = []
        for post in posts_data:
            try:
                post_date = pd.to_datetime(post["date post"]).date()  # Convert to datetime.date
                if start_date <= post_date and post_date <= end_date:  # Use separate comparison statements
                    filtered_data.append(post)
            except (ValueError, KeyError):
                # Skip invalid or missing date values
                continue
        return filtered_data

    # Function to perform time series analysis
    def perform_time_series_analysis(posts_data):
        df = pd.DataFrame(posts_data)
        
        if "date post" not in df.columns:
            st.error("No posts found within the selected date range.")
            return None
        
        df["date post"] = pd.to_datetime(df["date post"])  # Update column name here
        df.set_index("date post", inplace=True)  # Update column name here
        return df

    # Get list of JSON files in the "fbperson" folder
    folder_name = "fbperson"
    file_names = os.listdir(folder_name)
    file_paths = [os.path.join(folder_name, file_name) for file_name in file_names]

    # Multiselect form to choose files
    selected_files = st.multiselect("Select Files", file_names, default=file_names[:4])

    # Set default start date as one month before today
    default_start_date = datetime.now().date() - timedelta(days=30)

    # Calculate default end date as one month after the default start date
    default_end_date = default_start_date + timedelta(days=30)

    # Set the default start and end dates in the date_input widgets
    start_date = st.date_input("Start Date", value=default_start_date)
    end_date = st.date_input("End Date", value=default_end_date)

    # Perform time series analysis for selected files
    dataframes = []
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        if file_name in selected_files:
            # Load posts data from JSON file
            posts_data = load_posts_data(file_path)

            # Filter posts data based on start and end dates
            filtered_data = filter_posts_by_date(posts_data, start_date, end_date)

            # Perform time series analysis
            df = perform_time_series_analysis(filtered_data)

            # Check if DataFrame is not None
            if df is not None:
                # Append DataFrame to the list
                account_name = os.path.splitext(file_name)[0]  # Extract account name from the file title
                df["Account"] = account_name
                dataframes.append(df)

    # Concatenate DataFrames from selected files
    if dataframes:
        combined_df = pd.concat(dataframes)

        if not combined_df.empty:  # Check if the DataFrame is not empty
            # Group by date and account, and calculate the count of posts
            grouped_df = combined_df.groupby(["date post", "Account"]).size().unstack(fill_value=0)

            fig, ax = plt.subplots(figsize=(12, 6))  # Specify the figure size (width, height)

            for column in grouped_df.columns:
                ax.plot(grouped_df.index, grouped_df[column], label=column)

            ax.set_xlabel("Date", fontsize=8)  # Set the font size for x-axis label
            ax.set_ylabel("Post Count", fontsize=8)  # Set the font size for y-axis label
            ax.set_title("Time Series Analysis - Post Count by Account", fontsize=14)  # Set the font size for the title
            ax.legend()

            plt.xticks(fontsize=7)  # Set the font size for x-axis tick labels
            plt.yticks(fontsize=7)  # Set the font size for y-axis tick labels

            st.pyplot(fig)
        else:
            st.info("No data available for the selected date range.")

    # # Function to load posts data from a JSON file
    # def load_posts_data(file_path):
    #     with open(file_path, "r") as f:
    #         data = json.load(f)
    #     return data


    # def filter_posts_by_date(posts_data, start_date, end_date):
    #     filtered_data = []
    #     for post in posts_data:
    #         post_date = pd.to_datetime(post["date post"]).date()  # Convert to datetime.date
    #         if start_date <= post_date and post_date <= end_date:  # Use separate comparison statements
    #             filtered_data.append(post)
    #     return filtered_data

    # # Function to perform time series analysis
    # def perform_time_series_analysis(posts_data):
    #     df = pd.DataFrame(posts_data)
        
    #     if "date post" not in df.columns:
    #         st.error("No posts found within the selected date range.")
    #         return None
        
    #     df["date post"] = pd.to_datetime(df["date post"])  # Update column name here
    #     df.set_index("date post", inplace=True)  # Update column name here
    #     return df

    # # Get list of JSON files in the "fbperson" folder
    # folder_name = "fbperson"
    # file_names = os.listdir(folder_name)
    # file_paths = [os.path.join(folder_name, file_name) for file_name in file_names]

    # # Multiselect form to choose files
    # selected_files = st.multiselect("Select Files", file_names, default=file_names[:4])

    # # Start and end date inputs
    # start_date = st.date_input("Start Date")
    # end_date = st.date_input("End Date", value=pd.to_datetime("today"))

    # # Perform time series analysis for selected files
    # dataframes = []
    # for file_path in file_paths:
    #     file_name = os.path.basename(file_path)
    #     if file_name in selected_files:
    #         # Load posts data from JSON file
    #         posts_data = load_posts_data(file_path)

    #         # Filter posts data based on start and end dates
    #         filtered_data = filter_posts_by_date(posts_data, start_date, end_date)

    #         # Perform time series analysis
    #         df = perform_time_series_analysis(filtered_data)

    #         # Check if DataFrame is not None
    #         if df is not None:
    #             # Append DataFrame to the list
    #             account_name = os.path.splitext(file_name)[0]  # Extract account name from the file title
    #             df["Account"] = account_name
    #             dataframes.append(df)

    # # Concatenate DataFrames from selected files
    # if dataframes:
    #     combined_df = pd.concat(dataframes)

    #     if not combined_df.empty:  # Check if the DataFrame is not empty
    #         # Group by date and account, and calculate the count of posts
    #         grouped_df = combined_df.groupby(["date post", "Account"]).size().unstack(fill_value=0)

    #         fig, ax = plt.subplots(figsize=(12, 6))  # Specify the figure size (width, height)

    #         for column in grouped_df.columns:
    #             ax.plot(grouped_df.index, grouped_df[column], label=column)

    #         ax.set_xlabel("Date", fontsize=8)  # Set the font size for x-axis label
    #         ax.set_ylabel("Post Count", fontsize=8)  # Set the font size for y-axis label
    #         ax.set_title("Time Series Analysis - Post Count by Account", fontsize=14)  # Set the font size for the title
    #         ax.legend()

    #         plt.xticks(fontsize=7)  # Set the font size for x-axis tick labels
    #         plt.yticks(fontsize=7)  # Set the font size for y-axis tick labels

    #         st.pyplot(fig)
    #     else:
    #         st.info("No data available for the selected date range.")

##########################SENTIMENT ANALYSIS #########################################
    
   # Load posts and comments data from JSON file
    def load_posts_data(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    # Get list of JSON files in the "fbperson" folder
    folder_name = "fbperson"
    file_names = os.listdir(folder_name)
    file_paths = [os.path.join(folder_name, file_name) for file_name in file_names]

    selected_files = st.multiselect("Select Files", file_names, default=file_names[:4], key="file_selector")

    # Calculate default end date as one month after the start date
    if start_date:
        default_end_date = start_date + timedelta(days=30)
    else:
        default_end_date = datetime.now().date()

    # Set the default end date if it's greater than the current date
    if default_end_date > datetime.now().date():
        default_end_date = datetime.now().date()

    # Set the default start date as one month before the default end date
    default_start_date = default_end_date - timedelta(days=30)

    # Set the default start and end dates in the date_input widgets
    start_date = st.date_input("Start Date", value=default_start_date, key="start_date")
    end_date = st.date_input("End Date", value=default_end_date, key="end_date")

    # Initialize the sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Initialize data for time series chart
    sentiment_data = {'Date': [], 'Sentiment Score': [], 'Account': []}

    # Perform sentiment analysis for each selected file
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        if file_name in selected_files:
            # Extract the account name from the file name
            account_name = os.path.splitext(file_name)[0]

            # Load posts and comments data from JSON file
            data = load_posts_data(file_path)

            # Perform sentiment analysis for each comment
            for post in data:
                try:
                    post_date = datetime.strptime(post['date post'], "%Y-%m-%d %H:%M:%S").date()
                    if start_date <= post_date <= end_date:
                        for comment in post['comments']:
                            comment_text = comment['comment_text']
                            sentiment_scores = sia.polarity_scores(comment_text)
                            sentiment_score = sentiment_scores['compound']
                            sentiment_data['Date'].append(post_date)
                            sentiment_data['Sentiment Score'].append(sentiment_score)
                            sentiment_data['Account'].append(account_name)
                except ValueError:
                    # Skip invalid date format
                    continue

    # Create a DataFrame from the sentiment data
    df_sentiment = pd.DataFrame(sentiment_data)

    # Group the sentiment scores by date and account, and calculate the average sentiment score for each date and account
    df_sentiment_avg = df_sentiment.groupby(['Date', 'Account'])['Sentiment Score'].mean().reset_index()

    # Set the date column as the index
    df_sentiment_avg.set_index('Date', inplace=True)

    # Plot the time series sentiment analysis chart
    fig = plt.figure(figsize=(12, 6))

    for account_name in df_sentiment_avg['Account'].unique():
        account_data = df_sentiment_avg[df_sentiment_avg['Account'] == account_name]
        sentiment_scores = account_data['Sentiment Score']

        sentiment_labels = []
        for score in sentiment_scores:
            if score > 0:
                sentiment_labels.append("Positive")
            elif score < 0:
                sentiment_labels.append("Negative")
            else:
                sentiment_labels.append("Neutral")

        plt.plot(account_data.index, sentiment_scores, label=account_name)

        # Add sentiment labels to the plot
        for i in range(len(account_data.index)):
            plt.text(account_data.index[i], sentiment_scores[i], sentiment_labels[i], ha='center', va='bottom')

    plt.title('Time Series Sentiment Analysis')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()

    # Display the chart in Streamlit app
    st.pyplot(fig)
            

##################################################





with tab2:
    st.header("ISSUE")

with tab3:
    import requests

    st.header("DATA")
    # Calculate default start date and end date
    default_end_date = datetime.now().date()
    default_start_date = default_end_date - timedelta(days=30)

    def check_ip_blocked():
        url = "https://www.facebook.com"  # Replace with the URL of a website to test
        response = requests.get(url)
        if response.status_code == 200:
            return False  # IP is not blocked
        else:
            return True  # IP is blocked

    with st.form(key="fbaccform"):
        st.header("Facebook Account Scraper")
        account_name = st.text_input("Account Name", value="jemberkab")
        start_date = st.date_input("Start Date", value=default_start_date)
        end_date = st.date_input("End Date", value=default_end_date)

        if st.form_submit_button("Scrape"):
            if check_ip_blocked():
                st.error("IP is blocked. Please use a different IP.")
            else: 
                posts_data = asyncio.run(run_scraper(account_name, start_date, end_date))
                folder_name = "fbperson"
                file_name = f"{account_name}.json"
                file_path = os.path.join(folder_name, file_name)

                # Create the folder if it doesn't exist
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                # Save the scraping result as JSON
                with open(file_path, "w") as f:
                    json.dump(posts_data, f, indent=4)

                st.success(f"Scraping result saved successfully as {file_path}")
                

                st.write(posts_data)
