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

    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import timedelta, datetime
    import streamlit as st
    import os
    import json
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # Function to load posts and comments data from JSON file
    def load_posts_data(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    # Function to filter posts data by date range
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

        df["date post"] = pd.to_datetime(df["date post"])
        df.set_index("date post", inplace=True)
        return df

    # Get list of JSON files in the "fbperson" folder
    folder_name = "fbperson"
    file_names = os.listdir(folder_name)
    file_paths = [os.path.join(folder_name, file_name) for file_name in file_names]

    # Remove the ".json" extension from file names for display in the multiselect
    file_names_without_extension = [os.path.splitext(file_name)[0] for file_name in file_names]

    col1, col2, col3 = st.columns([2,1,1])
    # Multiselect form to choose files
    with col1:
        selected_files = st.multiselect("Select account", file_names_without_extension, default=file_names_without_extension[:4])

    # Set default start date as one month before today
    default_start_date = datetime.now().date() - timedelta(days=30)

    # Calculate default end date as one month after the default start date
    default_end_date = default_start_date + timedelta(days=30)
    with col2:
    # Set the default start and end dates in the date_input widgets
        start_date = st.date_input("Start Date", value=default_start_date)
    with col3:
        end_date = st.date_input("End Date", value=default_end_date)

    # Perform time series analysis for selected files
    dataframes = []
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        if os.path.splitext(file_name)[0] in selected_files:
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

            # Join the selected files without square brackets and single quotes for the title
            selected_files_str = ", ".join(selected_files)

            ax.set_title(f"Time Series Analysis - Post Count of {selected_files_str}", fontsize=14)  # Set the font size for the title
            ax.legend()

            plt.xticks(fontsize=7)  # Set the font size for x-axis tick labels
            plt.yticks(fontsize=7)  # Set the font size for y-axis tick labels

            st.pyplot(fig)
        else:
            st.info("No data available for the selected date range.")


    
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

    st.header('Sentiment Analysis')

    selected_files = st.multiselect("Select account", file_names, default=file_names[:4], key="file_selector")

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
            

########################SOCIAL NETWORK ANALYSIS ACCOUNT #########################

    import os
    import json
    import networkx as nx
    from pyvis.network import Network
    import streamlit as st

    # Function to read data from a JSON file
    def read_json(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    # Function to filter data by date range
    def filter_by_date(posts, start_date, end_date):
        return [post for post in posts if start_date <= post["date post"] <= end_date]

    def create_social_network_graph(posts):
        graph = nx.DiGraph()

        for post in posts:
            node_data = {
                "color": "#FFCC00" if ("hashtags" in post and len(post["hashtags"]) > 0) else "#FFFFFF",
                "relationship": "Tag" if ("hashtags" in post and len(post["hashtags"]) > 0) else "Unknown"
            }
            graph.add_node(str(post["username"]), **node_data)

            for hashtag in post.get("hashtags", []):
                if not graph.has_node(hashtag):
                    graph.add_node(str(hashtag), color="#FFCC00", relationship="Tag")
                graph.add_edge(str(post["username"]), str(hashtag), relationship="Tag")  # Add relationship attribute here

            for comment in post["comments"]:
                commenter = comment["username"]
                if not graph.has_node(commenter):
                    graph.add_node(str(commenter), color="#FF6666", relationship="Commenter")
                graph.add_edge(str(commenter), str(post["username"]), relationship="Commenter")

        return graph

    def visualize_graph(graph):
        nt = Network("800px", "100%", directed=True)

        # Define colors for nodes and edges based on the relationship type
        node_colors = {
            "Tag": "#FFCC00",        # Yellow
            "Commenter": "#FF6666",  # Light Red
        }

        edge_colors = {
            "Tag": "#99CCFF",        # Light Blue
            "Commenter": "#FF9999",  # Light Pink
        }

        for node in graph.nodes:
            relationship = graph.nodes[node].get("relationship", "Unknown")
            color = node_colors.get(relationship, "#FFFFFF")  # Default color: White
            nt.add_node(node, color=color)

        for edge in graph.edges(data=True):
            source, target, data = edge
            relationship = data.get("relationship", "Unknown")
            color = edge_colors.get(relationship, "#000000")  # Default color: Black
            nt.add_edge(source, target, label=relationship, color=color)

        return nt

    # Function to save the social network graph as an HTML file
    def save_graph(graph):
        nt = visualize_graph(graph)
        nt.save_graph("html_files/fbgraph.html")

    st.title("Social Network Analysis of Key Persons")

    # Load JSON files from "fbperso" folder
    folder_path = "fbperson"
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    file_names_dict = {f: f.replace(".json", "") for f in json_files}
    
    col1, col2, col3 =st.columns([2,1,1])
    with col1:
        selected_files = st.multiselect("Select account", list(file_names_dict.values()))

    if not selected_files:
        st.warning("Please select at least one account.")

    # Convert the selected file names back to full file names with ".json" extension
    selected_files_with_extension = [file_name + ".json" for file_name in selected_files]

    # Read selected JSON files
    posts = []
    for file_name in selected_files_with_extension:
        file_path = os.path.join(folder_path, file_name)
        posts.extend(read_json(file_path))

    # Filter posts by date
    with col2:
        start_date = st.date_input("Start Date", key="sdatefb5na")
    with col3:
        end_date = st.date_input("End Date", key="endatesnaf6")
    filtered_posts = filter_by_date(posts, str(start_date), str(end_date))

    # Create social network graph
    graph = create_social_network_graph(filtered_posts)

    # Save the graph as an HTML file
    save_graph(graph)


  

    #############################DEGRE CENTRALITY ############################
    import pandas as pd
    import matplotlib.pyplot as plt

    # Calculate degree centrality of nodes in the original graph
    degree_centrality = nx.degree_centrality(graph)

    # Create a new graph for degree centrality visualization
    degree_graph = nx.DiGraph()

    # Add nodes with sizes proportional to degree centrality values
    for node, centrality in degree_centrality.items():
        size = centrality * 500  # You can adjust the multiplier to control the size of nodes
        degree_graph.add_node(node, size=size)

    # Add edges from the original graph to the degree graph
    for edge in graph.edges():
        degree_graph.add_edge(edge[0], edge[1])

    # Visualize the new graph with node sizes reflecting degree centrality
    nt_degree = Network(height="900px", width="100%", bgcolor="#fff", font_color="darkgrey", directed=True)

    node_colors = {
            "Tag": "#FFCC00",        # Yellow
            "Commenter": "#FF6666",  # Light Red
        }

    edge_colors = {
            "Tag": "#99CCFF",        # Light Blue
            "Commenter": "#FF9999",  # Light Pink
        }

    # Add nodes and edges to the Pyvis network object with cluster information
    for node in degree_graph.nodes():
            relationship = graph.nodes[node].get("relationship", "Unknown")
            color = node_colors.get(relationship, "#FFFFFF")  # Default color: White
            nt_degree.add_node(node, color=color)
        

    for edge in degree_graph.edges():
        nt_degree.add_edge(edge[0], edge[1])

    # Save the degree centrality network graph as an HTML file
    nt_degree.save_graph("html_files/fbdegree_centrality.html")

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
    

#####################################BETWEENESS CENTRALITY ###################

# Calculate degree centrality of nodes in the original graph
    betweenness_centrality = nx.betweenness_centrality(graph)

    # Create a new graph for degree centrality visualization
    betweeness_graph = nx.DiGraph()

    # Add nodes with sizes proportional to degree centrality values
    for node, centrality in betweenness_centrality.items():
        size = centrality * 500  # You can adjust the multiplier to control the size of nodes
        betweeness_graph.add_node(node, size=size)

    # Add edges from the original graph to the degree graph
    for edge in graph.edges():
        betweeness_graph.add_edge(edge[0], edge[1])

    # Visualize the new graph with node sizes reflecting degree centrality
    nt_betweeness = Network(height="900px", width="100%", bgcolor="#fff", font_color="darkgrey", directed=True)

    node_colors = {
            "Tag": "#FFCC00",        # Yellow
            "Commenter": "#FF6666",  # Light Red
        }

    edge_colors = {
            "Tag": "#99CCFF",        # Light Blue
            "Commenter": "#FF9999",  # Light Pink
        }

    # Add nodes and edges to the Pyvis network object with cluster information
    for node in betweeness_graph.nodes():
            relationship = graph.nodes[node].get("relationship", "Unknown")
            color = node_colors.get(relationship, "#FFFFFF")  # Default color: White
            nt_betweeness.add_node(node, color=color)
        

    for edge in betweeness_graph.edges():
        nt_betweeness.add_edge(edge[0], edge[1])

    # Save the degree centrality network graph as an HTML file
    nt_betweeness.save_graph("html_files/fbbetween_centrality.html")

    # # Create a pandas DataFrame of the top ten nodes ranked by degree centrality
    # top_ten_betweeness = pd.DataFrame({"Node": list(betweenness_centrality.keys()), "Betweeness Centrality": list(betweenness_centrality.values())})
    # top_ten_betweeness = top_ten_betweeness.sort_values(by="Betweeness Centrality", ascending=False).head(10)

    # # Create a bar chart of the top ten nodes' degree centrality values
    # plt.figure(figsize=(8, 6))
    # plt.bar(top_ten_betweeness["Node"], top_ten_betweeness["Betweeness Centrality"])
    # plt.xlabel("Node")
    # plt.ylabel("Betweeness Centrality")
    # plt.title("Top 10 Nodes by Betweeness Centrality")
    # plt.xticks(rotation=45)
    # plt.tight_layout()

    
    
############################CLOSENESS ########################################


    # Calculate closeness centrality of nodes in the original graph
    closeness_centrality = nx.closeness_centrality(graph)

    # Create a new graph for closeness centrality visualization
    closeness_graph = nx.DiGraph()

    # Add nodes with sizes proportional to closeness centrality values
    for node, centrality in closeness_centrality.items():
        size = centrality * 500  # You can adjust the multiplier to control the size of nodes
        closeness_graph.add_node(node, size=size)

    # Add edges from the original graph to the closeness graph
    for edge in graph.edges():
        closeness_graph.add_edge(edge[0], edge[1])

    # Visualize the new graph with node sizes reflecting closeness centrality
    nt_closeness = Network(height="900px", width="100%", bgcolor="#fff", font_color="darkgrey", directed=True)

    # Use the same node_colors and edge_colors dictionaries as before

    # Add nodes and edges to the Pyvis network object with cluster information
    for node in closeness_graph.nodes():
        relationship = graph.nodes[node].get("relationship", "Unknown")
        color = node_colors.get(relationship, "#FFFFFF")  # Default color: White
        nt_closeness.add_node(node, color=color)

    for edge in closeness_graph.edges():
        nt_closeness.add_edge(edge[0], edge[1])

    # Save the closeness centrality network graph as an HTML file
    nt_closeness.save_graph("html_files/fbcloseness_centrality.html")

    # # Create a pandas DataFrame of the top ten nodes ranked by closeness centrality
    # top_ten_closeness = pd.DataFrame({"Node": list(closeness_centrality.keys()), "Closeness Centrality": list(closeness_centrality.values())})
    # top_ten_closeness = top_ten_closeness.sort_values(by="Closeness Centrality", ascending=False).head(10)

    # # Create a bar chart of the top ten nodes' closeness centrality values
    # plt.figure(figsize=(8, 6))
    # plt.bar(top_ten_closeness["Node"], top_ten_closeness["Closeness Centrality"])
    # plt.xlabel("Node")
    # plt.ylabel("Closeness Centrality")
    # plt.title("Top 10 Nodes by Closeness Centrality")
    # plt.xticks(rotation=45)
    # plt.tight_layout()


    ###################################visualsna#################################

    tabfbpksna1, tabfbpksna2, tabfbpksna3, tabfbpksna4=st.tabs(['Social Network','Main Actors', 'Bridging Actors', 'Supporting Actors'])

    with tabfbpksna1:
        # Display the graph using the HTML component
        with open("html_files/fbgraph.html", "r") as f:
            st.components.v1.html(f.read(), height=810)



    with tabfbpksna2:
        # Display the degree centrality network graph and the pandas DataFrame with the bar chart side by side in Streamlit
        with st.container():
            col_graph, col_data = st.columns([2, 1])

            with col_graph:
                with st.spinner("Loading Graph..."):
                    with open('html_files/fbdegree_centrality.html', 'r') as f:
                        degree_html_string = f.read()
                        st.components.v1.html(degree_html_string, height=910, scrolling=True)

            with col_data:
                with st.spinner("Loading Top 10 Main Actors..."):
                    st.write("### Top 10 Main Actors")
                    st.dataframe(top_ten_degree)
                    st.pyplot(plt)

    with tabfbpksna3:

        # Display the degree centrality network graph and the pandas DataFrame with the bar chart side by side in Streamlit
        with st.container():
            col_graph, col_data = st.columns([2, 1])

            with col_graph:
                with st.spinner("Loading Graph..."):
                    with open('html_files/fbbetween_centrality.html', 'r') as f:
                        betweeness_html_string = f.read()
                        st.components.v1.html(betweeness_html_string, height=910, scrolling=True)

            with col_data:
                with st.spinner("Loading Top 10 Bridging Actors..."):
                    st.write("### Top 10 Bridging Actors")
                    # Create a pandas DataFrame of the top ten nodes ranked by degree centrality
                    betweenness_centrality = nx.betweenness_centrality(graph)

                    top_ten_betweeness = pd.DataFrame({"Node": list(betweenness_centrality.keys()), "Betweeness Centrality": list(betweenness_centrality.values())})
                    top_ten_betweeness = top_ten_betweeness.sort_values(by="Betweeness Centrality", ascending=False).head(10)

                    st.dataframe(top_ten_betweeness)

                    # Create a bar chart of the top ten nodes' degree centrality values
                    plt.figure(figsize=(8, 6))
                    plt.bar(top_ten_betweeness["Node"], top_ten_betweeness["Betweeness Centrality"])
                    plt.xlabel("Node")
                    plt.ylabel("Betweeness Centrality")
                    plt.title("Top 10 Bridging Actors")
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    st.pyplot(plt)




    with tabfbpksna4:
    # Display the closeness centrality network graph and the pandas DataFrame with the bar chart side by side in Streamlit
        with st.container():
            col_graph, col_data = st.columns([2, 1])

            with col_graph:
                with st.spinner("Loading Graph..."):
                    with open('html_files/fbcloseness_centrality.html', 'r') as f:
                        closeness_html_string = f.read()
                        st.components.v1.html(closeness_html_string, height=910, scrolling=True)

            with col_data:
                with st.spinner("Loading Top 10 Supporting Actors..."):
                    st.write("### Top 10 Supporting Actors")

                    # Update the top_ten_closeness DataFrame based on the filtered posts
                    closeness_centrality = nx.closeness_centrality(graph)
                    

                    top_ten_closeness = pd.DataFrame({"Node": list(closeness_centrality.keys()), "Closeness Centrality": list(closeness_centrality.values())})
                    top_ten_closeness = top_ten_closeness.sort_values(by="Closeness Centrality", ascending=False).head(10)

                    st.dataframe(top_ten_closeness)

                    # Create a bar chart of the top ten nodes' closeness centrality values
                    plt.figure(figsize=(8, 6))
                    plt.bar(top_ten_closeness["Node"], top_ten_closeness["Closeness Centrality"])
                    plt.xlabel("Node")
                    plt.ylabel("Closeness Centrality")
                    plt.title("Top 10 Supporting Actors")  # Set the title here
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    st.pyplot(plt)


###########################################TOPIC MODEL LDA#################################

    import os
    import json
    import gensim
    import nltk
    import pyLDAvis
    import streamlit as st
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from streamlit import components
    import string
    import re
    from datetime import timedelta, datetime
    import time
    import nltk

    nltk.download('punkt')
    nltk.download('stopwords')

    # Function to preprocess the "post text" data in Bahasa Indonesia
    def preprocess_text(text):
        # Tokenize the text
        words = word_tokenize(text)

        # Define Indonesian stopwords
        indonesian_stopwords = set(stopwords.words('indonesian'))

        # Remove punctuation and stopwords
        words = [word for word in words if word.isalpha() and word not in indonesian_stopwords]

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        # Join the words back to a string
        processed_text = ' '.join(words)

        return processed_text.lower()

    def load_and_preprocess_data(folder_path):
        data = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                with open(os.path.join(folder_path, filename), "r") as file:
                    posts_data = json.load(file)
                    for post_data in posts_data:
                        post_text = preprocess_text(post_data.get("post text", ""))
                        post_date = post_data.get("date post", "")
                        data.append((post_text, post_date))
        return data

    # Function to perform topic modeling using Gensim's LDA model
    def perform_topic_modeling(data, num_topics=5):
        # Preprocess text and tokenize
        texts = [word_tokenize(text) for text, _ in data]

        # Create a dictionary and corpus
        dictionary = gensim.corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
        return lda_model, corpus, dictionary  # Return the corpus and dictionary


   # Function to filter data based on start date and end date
    def filter_data(data, start_date, end_date):
        filtered_data = []
        for text, date_str in data:
            # Convert the date_str to a datetime.date object
            date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").date()
            if start_date <= date <= end_date:
                filtered_data.append((text, date))
        return filtered_data

    # Function to create a word cloud with keywords from the LDA model
    def create_word_cloud_from_lda(lda_model):
        topics = lda_model.show_topics(num_topics=10, num_words=10, formatted=False)
        keywords = [word for topic in topics for word, _ in topic[1]]
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))

        # Get the colormap
        colormap = 'viridis'

        # Set the colormap for the word cloud
        wordcloud.colormap = colormap

        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Keywords Word Cloud')
        st.pyplot(plt)
        plt.close()

  
    # Function to load and preprocess a single JSON file
    def load_and_preprocess_json(file_path):
        with open(file_path, "r") as file:
            posts_data = json.load(file)
        data = []
        for post_data in posts_data:
            post_text = preprocess_text(post_data.get("post text", ""))
            post_date = post_data.get("date post", "")
            data.append((post_text, post_date))
        return data

    # The data loading and preprocessing part
    st.header("Facebook Post Topic Modeling")

    # Get list of JSON files in the "fbperson" folder
    folder_name = "fbperson"
    file_names = os.listdir(folder_name)
    file_paths = [os.path.join(folder_name, file_name) for file_name in file_names]

    # Extract file names without ".json" extension for select box
    file_names_without_extension = [file_name[:-5] for file_name in file_names]
    col1, col2, col3=st.columns([2,1,1])
    # Display a select box to choose a JSON file for topic modeling
    # Display a select box to choose a JSON file for topic modeling
    with col1:
        selected_file_name = st.selectbox("Select account", file_names_without_extension)

    # Load and preprocess data from the selected JSON file
    selected_file_path = [file_path for file_name, file_path in zip(file_names, file_paths) if file_name == selected_file_name + ".json"][0]
    data = load_and_preprocess_json(selected_file_path)

    # Set default start date as one month before today
    default_start_date = datetime.now().date() - timedelta(days=30)

    # Calculate default end date as one month after the default start date
    default_end_date = default_start_date + timedelta(days=30)

    # Set the default start and end dates in the date_input widgets
    with col2:
        start_date = st.date_input("Start Date", value=default_start_date, key='stf6kptp')
    with col3:
        end_date = st.date_input("End Date", value=default_end_date, key='kp3dfb')

    # Filter data based on date range
    filtered_data = filter_data(data, start_date, end_date)

    # Create the subheader "Topic Modeling of {selected file name}"
    st.subheader(f"Topic Modeling of {selected_file_name}")

    # Check if filtered_data is not empty before performing topic modeling
    if not filtered_data:
        st.error("No posts found within the selected date range. Please choose a different date range.")
    else:
        # Perform topic modeling and display results using pyLDAvis
        num_topics = 10
        lda_model, corpus, dictionary = perform_topic_modeling(filtered_data, num_topics=num_topics)  # Receive corpus and dictionary

        # Generate the pyLDAvis visualization
        lda_display = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word, sort_topics=False)

        # Save the pyLDAvis visualization as an HTML file
        html_string = pyLDAvis.prepared_data_to_html(lda_display)

        # Display the HTML in Streamlit
        st.components.v1.html(html_string, height=800, scrolling=False) 

        # Create the word cloud with keywords from the LDA model
        create_word_cloud_from_lda(lda_model)

##################################SENTIMENT ANALYSIS###########################


    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import timedelta, datetime
    import streamlit as st
    import json
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    


    # Function to load posts and comments data from JSON file
    def load_posts_data(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    # Function to perform sentiment analysis on the comment_text
    def analyze_comment_sentiment(comment_text):
        sentiment_scores = sia.polarity_scores(comment_text)
        sentiment_score = sentiment_scores['compound']
        if sentiment_score > 0:
            return 'Positive'
        elif sentiment_score < 0:
            return 'Negative'
        else:
            return 'Neutral'

    # Get list of JSON files in the "fbperson" folder
    folder_name = "fbperson"
    file_names = os.listdir(folder_name)
    file_paths = [os.path.join(folder_name, file_name) for file_name in file_names]

    # Extract file names without the ".json" extension
    file_names_without_extension = [os.path.splitext(file_name)[0] for file_name in file_names]

    st.header('Sentiment Analysis')

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_files = st.multiselect("Select account", file_names_without_extension, default=file_names_without_extension[:4], key="mul53lfkfb")

    # Calculate default end date as one month after the start date
    default_end_date = datetime.now().date()
    default_start_date = default_end_date - timedelta(days=30)

    # Set the default start and end dates in the date_input widgets
    with col2:
        start_date = st.date_input("Start Date", value=default_start_date, key="stdf8kp")
    with col3:
        end_date = st.date_input("End Date", value=default_end_date, key="3dtfbkp")

    # Initialize the sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Initialize data for pie charts for posts and comments
    post_pie_charts_data = []
    comment_pie_charts_data = []

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        if os.path.splitext(file_name)[0] in selected_files:  # Check if the file name without extension is in selected_files
            # Extract the account name from the file name
            account_name = os.path.splitext(file_name)[0]

            # Load posts and comments data from JSON file
            data = load_posts_data(file_path)

            # Initialize data for the current file's pie charts
            post_pie_chart_data = {'Sentiment': [], 'Count': []}
            comment_pie_chart_data = {'Sentiment': [], 'Count': []}

            # Perform sentiment analysis for each post and comment
            for post in data:
                try:
                    post_date = datetime.strptime(post['date post'], "%Y-%m-%d %H:%M:%S").date()
                    if start_date <= post_date <= end_date:
                        post_text = post['post text']
                        post_sentiment = analyze_comment_sentiment(post_text)

                        # Add the post sentiment label and count to the current file's pie chart data
                        if post_sentiment not in post_pie_chart_data['Sentiment']:
                            post_pie_chart_data['Sentiment'].append(post_sentiment)
                            post_pie_chart_data['Count'].append(1)
                        else:
                            idx = post_pie_chart_data['Sentiment'].index(post_sentiment)
                            post_pie_chart_data['Count'][idx] += 1

                        for comment in post['comments']:
                            comment_text = comment['comment_text']
                            comment_sentiment = analyze_comment_sentiment(comment_text)

                            # Add the comment sentiment label and count to the current file's pie chart data
                            if comment_sentiment not in comment_pie_chart_data['Sentiment']:
                                comment_pie_chart_data['Sentiment'].append(comment_sentiment)
                                comment_pie_chart_data['Count'].append(1)
                            else:
                                idx = comment_pie_chart_data['Sentiment'].index(comment_sentiment)
                                comment_pie_chart_data['Count'][idx] += 1
                except ValueError:
                    # Skip invalid date format
                    continue

            # Create DataFrames from the current file's pie chart data for posts and comments
            df_post_pie_chart = pd.DataFrame(post_pie_chart_data)
            df_comment_pie_chart = pd.DataFrame(comment_pie_chart_data)

            # Add the current file's pie chart data to the list of pie chart data for posts and comments
            post_pie_charts_data.append((account_name, df_post_pie_chart))
            comment_pie_charts_data.append((account_name, df_comment_pie_chart))

    # Display the pie charts for posts and comments side by side
    num_columns = len(selected_files)  # Number of columns equals the number of selected files
    num_pie_charts = len(post_pie_charts_data)

    for i in range(num_pie_charts):
        account_name, df_post_pie_chart = post_pie_charts_data[i]
        _, df_comment_pie_chart = comment_pie_charts_data[i]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 6))

        ax1.pie(df_post_pie_chart['Count'], labels=df_post_pie_chart['Sentiment'], autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'lightcoral', 'lightblue'])
        ax1.axis('equal')
        ax1.set_title(f'Post Sentiment Distribution - {account_name}')

        ax2.pie(df_comment_pie_chart['Count'], labels=df_comment_pie_chart['Sentiment'], autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'lightcoral', 'lightblue'])
        ax2.axis('equal')
        ax2.set_title(f'Comment Sentiment Distribution - {account_name}')

        st.pyplot(fig)




#############################SENTIMENT PER USER ################################
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from datetime import timedelta, datetime
    import streamlit as st
    import json
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('vader_lexicon')

    # Function to load posts and comments data from JSON file
    def load_posts_data(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    # Function to perform sentiment analysis on the comment_text
    def analyze_comment_sentiment(comment_text):
        sentiment_scores = sia.polarity_scores(comment_text)
        sentiment_score = sentiment_scores['compound']
        if sentiment_score > 0:
            return 'Positive'
        elif sentiment_score < 0:
            return 'Negative'
        else:
            return 'Neutral'

    # Get list of JSON files in the "fbperson" folder
    folder_name = "fbperson"
    file_names = os.listdir(folder_name)
    file_paths = [os.path.join(folder_name, file_name) for file_name in file_names]

    # Extract file names without the ".json" extension
    file_names_without_extension = [os.path.splitext(file_name)[0] for file_name in file_names]

    st.header('Sentiment Analysis per User based on Comments')

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_files = st.multiselect("Select account", file_names_without_extension, default=file_names_without_extension[:4], key="usent1mul53lfkfb")

    # Calculate default end date as one month after the start date
    default_end_date = datetime.now().date()
    default_start_date = default_end_date - timedelta(days=30)

    # Set the default start and end dates in the date_input widgets
    with col2:
        start_date = st.date_input("Start Date", value=default_start_date, key="stdf8kpuser53nti")
    with col3: 
        end_date = st.date_input("End Date", value=default_end_date, key="3dtfbkpuser5ent1")

    # Initialize the sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Initialize data for bar charts for comments
    comment_bar_charts_data = []

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        if os.path.splitext(file_name)[0] in selected_files:  # Check if the file name without extension is in selected_files
            # Extract the account name from the file name
            account_name = os.path.splitext(file_name)[0]

            # Load posts and comments data from JSON file
            data = load_posts_data(file_path)

            # Initialize data for the current file's bar chart
            comment_bar_chart_data = {'User': [], 'Positive': [], 'Negative': [], 'Neutral': []}

            # Perform sentiment analysis for each comment
            for post in data:
                try:
                    post_date = datetime.strptime(post['date post'], "%Y-%m-%d %H:%M:%S").date()
                    if start_date <= post_date <= end_date:
                        for comment in post['comments']:
                            comment_text = comment['comment_text']
                            comment_owner = comment['username']
                            comment_sentiment = analyze_comment_sentiment(comment_text)

                            # Add the comment sentiment to the data
                            if comment_owner not in comment_bar_chart_data['User']:
                                comment_bar_chart_data['User'].append(comment_owner)
                                comment_bar_chart_data['Positive'].append(0)
                                comment_bar_chart_data['Negative'].append(0)
                                comment_bar_chart_data['Neutral'].append(0)

                            idx = comment_bar_chart_data['User'].index(comment_owner)
                            comment_bar_chart_data[comment_sentiment][idx] += 1
                except ValueError:
                    # Skip invalid date format
                    continue

            # Create DataFrame from the current file's bar chart data for comments
            df_comment_bar_chart = pd.DataFrame(comment_bar_chart_data)

            # Add the current file's bar chart data to the list of bar chart data for comments
            comment_bar_charts_data.append((account_name, df_comment_bar_chart))

    # Display the bar charts for comments side by side
    num_columns = len(selected_files)  # Number of columns equals the number of selected files
    num_bar_charts = len(comment_bar_charts_data)

    for i in range(num_bar_charts):
        account_name, df_comment_bar_chart = comment_bar_charts_data[i]

        fig, ax = plt.subplots(figsize=(20, 6))

        ax.bar(df_comment_bar_chart['User'], df_comment_bar_chart['Positive'], label='Positive', color='lightgreen')
        ax.bar(df_comment_bar_chart['User'], df_comment_bar_chart['Negative'], label='Negative', color='lightcoral', bottom=df_comment_bar_chart['Positive'])
        ax.bar(df_comment_bar_chart['User'], df_comment_bar_chart['Neutral'], label='Neutral', color='lightblue', bottom=df_comment_bar_chart['Positive'] + df_comment_bar_chart['Negative'])
        ax.set_xlabel('User')
        ax.set_ylabel('Count')
        ax.set_title(f'Comment Sentiment Distribution per User - {account_name}')
        ax.legend()

        # Rotate the x-axis tick labels diagonally
        plt.xticks(rotation=45, ha='right')

        st.pyplot(fig)




#######################################TAB2#####################################
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
