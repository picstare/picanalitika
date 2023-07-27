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
import joblib
from sklearn.metrics import accuracy_score
import plotly.express as px
from streamlit_extras.app_logo import add_logo
import base64
from PIL import Image


nltk.download('vader_lexicon')


consumer_key = "1Hov50UKDBETZmY1wR9zkE3Q7"
consumer_secret = "lAAcJVSDE1Oyc1BuOmxJNN4D575NkHQcg3hEa5zeurrGwCpXH0"
access_token = "16645853-jRxQql8XCzcaWsBSTeA3eutXPA5xRcHxqRHDgx6m9"
access_token_secret = "STmNDeF9BX33PTYuE18vPq7yndA4okKroeq9LXX6FV2gk"
bearer_token = 'AAAAAAAAAAAAAAAAAAAAALMe9wAAAAAAN%2BggvuMVDLKLIEX3Kk%2B8nOSxH88%3DiBMoMLAjE4JuPUKzRyZjYbs5zRZ82uZk9T89YCBgKkeXmgbKY5'

accounts = []
auth = tweepy.OAuth1UserHandler(consumer_key=consumer_key, consumer_secret=consumer_secret, access_token=access_token, access_token_secret=access_token_secret)
# auth.set_access_token(, 'ph47iEpbfD4USmwynPCL1LLNtl7f9seLovjIHOUqwTuQq')
api = tweepy.API(auth)
client = tweepy.Client(bearer_token=bearer_token)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


st.set_page_config(page_title="Picanalitika | Twitter Analysis", layout="wide")

hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div.vis-configuration-wrapper{display: block; width: 400px;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

class DateTimeEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)
    


####################LOGOUT####################
with st. sidebar:
    if st.button("Logout"):
        logout()
        switch_page('home')

#################STARTPAGE###################

a, b = st.columns([1, 10])

with a:
    st.text("")
    st.image("img/twitterlkogo.png", width=100)
with b:
    st.title("Twitter Analysis")
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

    container1=st.container()
    with container1:
        folder_path = "twittl"
        files = os.listdir(folder_path)

        # Get the modification times of the files
        file_times = [(f, os.path.getmtime(os.path.join(folder_path, f))) for f in files]

        # Sort the files based on modification time in descending order
        sorted_files = sorted(file_times, key=lambda x: x[1], reverse=True)

        # Select the four newest files
        num_files = min(4, len(sorted_files))
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
                        user_data = json.load(f)
                        
                    # Access the follower count
                    followers_count = user_data["tweets"][0]["user"]["followers_count"]
                    profilepic=user_data["tweets"][0]["user"]["profile_image_url_https"]

                    friend_count=user_data["tweets"][0]["user"]["friends_count"]
                    listed_count=user_data["tweets"][0]["user"]["listed_count"]
                    status=user_data["tweets"][0]["user"]["statuses_count"]

                    
                    # Display the user data in the column
                    col.image(profilepic, width=100)
                    # col.write(f"Account: {files[i].replace('_data.json', '')}")
                    col.write(f"{user_data['name']}")
                    col.write(f"{user_data['description']}")
                    col.write(f"Tweets: {status}")
                    col.write(f"Followers: {followers_count}")
                    col.write(f"Friend: {friend_count}")
                    col.write(f"Listed: {listed_count}")

###################################### CHART TIME SERIES #######################

        st.header("TIME SERIES ANALYSIS OF THE KEY PERSONS")
        

        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
        # data = []
        df1 = None

        if files:
            data = []
            for file in files:
                with open(file, 'r') as f:
                    file_data = json.load(f)
                    for tweet in file_data['tweets']:
                        data.append({
                            'name': tweet['user']['name'],
                            'date': pd.to_datetime(tweet['created_at'])
                        })
            df1 = pd.DataFrame(data)

        # Create a list of available screen names
        if df1 is not None:
            names = list(df1['name'].unique())
        else:
            names = []

        # Set the default selected names to the first 4 accounts
        default_names = names[:4]

        # Set the default time range to one month from the current date
        end_date = pd.to_datetime(datetime.today(), utc=True)
        start_date = end_date - timedelta(days=30)

        # Create widgets for selecting the screen name and time range
        cols_t0, cols_ta, cols_tb = st.columns([2, 1, 1])
         
        selected_names = cols_t0.multiselect('Select names to compare', names, default=default_names, key='selper')
        start_date = pd.to_datetime(cols_ta.date_input('Start date', value=start_date), utc=True)
        end_date = pd.to_datetime(cols_tb.date_input('End date', value=end_date), utc=True)

        # Filter the data based on the selected names and time range
        if df1 is not None:
            mask = (df1['name'].isin(selected_names)) & (df1['date'] >= start_date) & (df1['date'] <= end_date)
            df1_filtered = df1.loc[mask]
        else:
            df1_filtered = pd.DataFrame()

        if len(df1_filtered) > 0:
            # Calculate total posts and average posts per day for each account
            df1_summary = df1_filtered.groupby('name').agg(
                total_posts=('date', 'count'),
                average_posts_per_day=('date', lambda x: x.count() / (end_date - start_date).days)
            ).reset_index()

            # Display the matrix with total posts and average posts per day
            st.write(df1_summary)

            # Plot the time series line chart as before
            df1_grouped = df1_filtered.groupby(['date', 'name']).size().reset_index(name='count')
            fig, ax = plt.subplots(figsize=(20, 6))
            sns.lineplot(data=df1_grouped, x='date', y='count', hue='name', ax=ax)
            ax.set_title(f"Tweets per Day of {', '.join(selected_names)}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Tweets")
            st.pyplot(fig)
        else:
            st.write("No data available for the selected time range and users.")
    
        st.markdown("---")

    #####################SNA########################
        import json
        import os
        import networkx as nx
        from pyvis.network import Network
        import streamlit as st
        from datetime import datetime, timedelta
        import pandas as pd

        st.header('TWITTER NETWORK ANALYSIS OF KEYPERSONS')

        # Function to create social network analysis graph
        def create_social_network_analysis(json_data):
            # Create an undirected graph
            graph = nx.Graph()

            # Add edges based on tweet relationships
            for tweet in json_data["tweets"]:
                tweet_user = tweet["user"]["screen_name"]

                if "quoted_status_user" in tweet:
                    quoted_user = tweet["quoted_status_user"]["screen_name"]
                    graph.add_edge(tweet_user, quoted_user, relationship="quoted")

                for mention in tweet["entities"]["user_mentions"]:
                    mention_user = mention["screen_name"]
                    graph.add_edge(tweet_user, mention_user, relationship="mention")

                if "in_reply_to_screen_name" in tweet:
                    reply_user = tweet["in_reply_to_screen_name"]
                    if reply_user is not None:
                        graph.add_edge(tweet_user, reply_user, relationship="reply")

            # Add nodes for followers and following
            for follower in json_data["followers"]:
                graph.add_edge(follower, tweet_user, relationship="follower")

            for following in json_data["following"]:
                graph.add_edge(tweet_user, following, relationship="following")

            # Compute the layout using spring layout
            layout = nx.spring_layout(graph)

            # Set the layout as a node attribute
            nx.set_node_attributes(graph, layout, name="pos")

            return graph

        # Load JSON data from selected files
        def load_json_files(json_files):
            json_data_list = []
            for json_file in json_files:
                with open(json_file) as f:
                    json_data = json.load(f)
                    json_data_list.append(json_data)
            return json_data_list

        # Get list of JSON files from the "twittl" folder
        folder_path = "twittl"
        json_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".json")]

        snakpcol1, snakpcol2, snakpcol3 = st.columns([2, 1, 1])

        # Initialize st.session_state
        if "selected_files" not in st.session_state:
            st.session_state.selected_files = [os.path.basename(json_files[0]).split(".")[0]]

        if "start_date" not in st.session_state:
            default_start_date = datetime.now().date() - timedelta(days=30)
            st.session_state.start_date = default_start_date

        if "end_date" not in st.session_state:
            default_end_date = datetime.now().date()
            st.session_state.end_date = default_end_date

        # Select JSON files
        with snakpcol1:
            selected_files = st.multiselect(
                "Select JSON files",
                [os.path.basename(file).split(".")[0] for file in json_files],
                default=st.session_state.selected_files,
            )

        # Select start and end date
        with snakpcol2:
            start_date = st.date_input("Select start date", value=st.session_state.start_date)
        with snakpcol3:
            end_date = st.date_input("Select end date", value=st.session_state.end_date)

        # Filter JSON files based on selected dates
        filtered_files = []
        for file in selected_files:
            file_name = f"{file}.json"  # Remove the additional "_data" part
            file_path = os.path.join(folder_path, file_name)
            with open(file_path) as f:
                json_data = json.load(f)
                created_at = json_data["tweets"][0]["created_at"]
                created_date = datetime.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y").date()
                if start_date <= created_date <= end_date:
                    filtered_files.append(file_path)

        # Load selected JSON files
        json_data_list = load_json_files(filtered_files)

        # Create social network analysis graphs
        graphs = [create_social_network_analysis(json_data) for json_data in json_data_list]

        # Define group values for each relationship
        group_by_relationship = {
            "follower": 1,
            "following": 2,
            "quoted": 3,
            "mention": 4,
            "reply": 5
        }

        # Define color values for each group
        group_colors = {
            1: "#ff0000",  # Red
            2: "#00ff00",  # Green
            3: "#0000ff",  # Blue
            4: "#ffff00",  # Yellow
            5: "#ff00ff"   # Magenta
        }

        # Function to compute network statistics
        def compute_network_statistics(graph):
            num_nodes = len(graph.nodes())
            num_edges = len(graph.edges())
            density = nx.density(graph)
            connectivity = nx.is_connected(graph)
            avg_clustering = nx.average_clustering(graph)

            # Compute counts for each relationship type
            follower_count = sum(1 for (source, target, data) in graph.edges(data=True) if data.get('relationship', '') == "follower")
            following_count = sum(1 for (source, target, data) in graph.edges(data=True) if data.get('relationship', '') == "following")
            mention_count = sum(1 for (source, target, data) in graph.edges(data=True) if data.get('relationship', '') == "mention")
            reply_count = sum(1 for (source, target, data) in graph.edges(data=True) if data.get('relationship', '') == "reply")
            quote_count = sum(1 for (source, target, data) in graph.edges(data=True) if data.get('relationship', '') == "quoted")

            # Compute connectivity score
            if connectivity:
                connectivity_score = 1.0
            else:
                largest_component = max(nx.connected_components(graph), key=len)
                connectivity_score = len(largest_component) / num_nodes

            return {
                "Number of Nodes": num_nodes,
                "Number of Edges": num_edges,
                "Density": density,
                "Connectivity": connectivity_score,
                "Average Clustering": avg_clustering,
                "Follower Count": follower_count,
                "Following Count": following_count,
                "Mention Count": mention_count,
                "Reply Count": reply_count,
                "Quote Count": quote_count
            }
        
        

       # Compute network statistics for each graph
        network_statistics = [compute_network_statistics(graph) for graph in graphs]

        # Create a pandas DataFrame from the network statistics
        df_network_statistics = pd.DataFrame(network_statistics)

        # Transpose the DataFrame to display it vertically
        df_network_statistics = df_network_statistics.T

        # Define the threshold for virality analysis
        virality_threshold = 0.5

        # Function to analyze network implications
        def analyze_network_implications(df_network_statistics):
            density = df_network_statistics.loc["Density"].values[0]
            virality_threshold = 0.5
            
            if density > virality_threshold:
                virality = "Tinggi"
            else:
                virality = "Rendah"
                
            return virality

        # Function to analyze network engagement
        def analyze_network_engagement(df_network_statistics):
            avg_clustering = df_network_statistics.loc["Average Clustering"].values[0]
            num_edges = df_network_statistics.loc["Number of Edges"].values[0]
            
            clustering_threshold = 0.2  # Adjust this threshold as needed
            edges_threshold = 50  # Adjust this threshold as needed
            
            if avg_clustering > clustering_threshold and num_edges > edges_threshold:
                engagement = "Tinggi"
            else:
                engagement = "Rendah"
                
            return engagement

        # Function to analyze network reach
        def analyze_network_reach(df_network_statistics):
            follower_count = df_network_statistics.loc["Follower Count"].values[0]
            following_count = df_network_statistics.loc["Following Count"].values[0]
            
            if follower_count > following_count:
                reach = "Luas"
            else:
                reach = "Sempit"
                
            return reach

        # Function to analyze network influence
        def analyze_network_influence(df_network_statistics):
            mention_count = df_network_statistics.loc["Mention Count"].values[0]
            reply_count = df_network_statistics.loc["Reply Count"].values[0]
            quote_count = df_network_statistics.loc["Quote Count"].values[0]
            
            total_interactions = mention_count + reply_count + quote_count
            
            if total_interactions > 100:  # Adjust this threshold as needed
                influence = "Tinggi"
            else:
                influence = "Rendah"
                
            return influence

        # Analyze network implications
        network_implications = analyze_network_implications(df_network_statistics)

        # Analyze network engagement
        network_engagement = analyze_network_engagement(df_network_statistics)

        # Analyze network reach
        network_reach = analyze_network_reach(df_network_statistics)

        # Analyze network influence
        network_influence = analyze_network_influence(df_network_statistics)


        # Function untuk memberikan rekomendasi otomatis berdasarkan analisis jaringan
        def generate_recommendations(network_implications, network_engagement, network_reach, network_influence):
            recommendations = []

            if network_implications == "High":
                recommendations.append("Pertimbangkan untuk memanfaatkan viralitas tinggi jaringan ini dalam kampanye yang berdampak.")
            else:
                recommendations.append("Fokus pada keterlibatan terarah untuk meningkatkan viralitas jaringan.")

            if network_engagement == "High":
                recommendations.append("Terlibatlah aktif dengan jaringan ini untuk memperkuat hubungan.")
            else:
                recommendations.append("Cari cara untuk meningkatkan keterlibatan dan interaksi dalam jaringan ini.")

            if network_reach == "Wide":
                recommendations.append("Jaringan Anda memiliki jangkauan yang luas. Pertimbangkan untuk mencapai audiens baru.")
            else:
                recommendations.append("Manfaatkan jaringan yang erat untuk menguatkan pesan kepada kelompok tertentu.")

            if network_influence == "High":
                recommendations.append("Jaringan Anda memiliki pengaruh yang signifikan. Gunakan pengaruh ini untuk mendorong perubahan positif.")
            else:
                recommendations.append("Usahakan untuk meningkatkan interaksi dan sebutan untuk meningkatkan pengaruh jaringan Anda.")

            return recommendations

        # Generate automatic recommendations based on the network analysis
        network_recommendations = generate_recommendations(network_implications, network_engagement, network_reach, network_influence)

       
        # Visualize the graph using Pyvis
        def visualize_social_network(graphs):
            # Visualize the graph using Pyvis
            nt = Network(height="910px", width="100%", bgcolor="#fff", font_color="grey")
            for i, graph in enumerate(graphs):
                for edge in graph.edges(data=True):
                    source, target, data = edge
                    relationship = data.get('relationship', '')
                    group = group_by_relationship.get(relationship, 0)
                    nt.add_node(source, label=source, group=group)
                    nt.add_node(target, label=target, group=group)
                    nt.add_edge(source, target, label=relationship)

            # Save the graph to an HTML file
            html_path = "html_files/social_network.html"
            nt.save_graph(html_path)

            # Display the network visualization
            st.components.v1.html(open(html_path, 'r').read(), height=920, scrolling=False)
        

############################################DEGREE CENTRALITY #########################################

       
    def visualize_degree_centrality_network(graphs):
        nt = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='#333333', directed=True)
        degree_centrality_values = []

        for graph in graphs:
            # Calculate degree centrality
            degree_centrality = nx.degree_centrality(graph)
            degree_subgraph = graph.subgraph([node for node, centrality in degree_centrality.items() if centrality > 0])
            degree_centrality_values.append(degree_centrality)

            # Add nodes to the network with size based on degree centrality
            for node in degree_subgraph.nodes:
                centrality = degree_centrality[node]
                node_size = centrality * 20  # Adjust the scaling factor as needed
                nt.add_node(node, label=node, size=node_size)

            # Add edges to the network
            for edge in degree_subgraph.edges:
                source = edge[0]
                target = edge[1]
                relationship = degree_subgraph.edges[edge]['relationship']
                nt.add_edge(source, target, label=relationship)

        nt.save_graph('html_files/degree_centrality_network.html')

        col1, col2 = st.columns([2, 1])

        with col1:
            with open('html_files/degree_centrality_network.html', 'r') as f:
                html_string = f.read()
                st.components.v1.html(html_string, height=960, scrolling=True)

        top_actors = []
        centrality_scores = []
        for degree_centrality in degree_centrality_values:
            top_actors += [actor for actor in degree_centrality if actor in degree_subgraph.nodes]
            centrality_scores += [degree_centrality[actor] for actor in top_actors if actor in degree_centrality]

        # Create a DataFrame of the top 10 degree centrality nodes
        df_top_actors = pd.DataFrame({'Actor': top_actors, 'Degree Centrality': centrality_scores})
        df_top_actors = df_top_actors.sort_values(by='Degree Centrality', ascending=False).head(10)
        with col2:
            # Display the top 10 degree centrality nodes DataFrame
            st.subheader("Top 10 Degree Centrality Nodes")
            st.dataframe(df_top_actors)

            x_pos = list(range(len(df_top_actors)))

            # Set the font size
            plt.rc('font', size=8)

            # Set the figure size
            fig, ax = plt.subplots(figsize=(10, 6))  # Set the width to 10 inches and height to 6 inches

            ax.bar(x_pos, df_top_actors['Degree Centrality'])
            ax.set_ylabel('Degree Centrality')
            ax.set_xlabel('Top Actors')
            ax.set_title('Top 10 Main Actors by Degree Centrality')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_top_actors['Actor'], rotation=45, ha='right')  # Display user names on the x-axis
            plt.tight_layout()

            # Display the plot in Streamlit
            st.pyplot(fig) 

################## BETWEENESS CENTRALITY ##########################

   

    def visualize_betweenness_centrality_network(graphs):
        nt = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='#333333', directed=True)
        betweenness_centrality_values = []

        for graph in graphs:
            # Calculate betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(graph)
            betweenness_subgraph = graph.subgraph([node for node, centrality in betweenness_centrality.items() if centrality > 0])
            betweenness_centrality_values.append(betweenness_centrality)

            # Add nodes to the network with size based on betweenness centrality
            for node in betweenness_subgraph.nodes:
                centrality = betweenness_centrality[node]
                node_size = centrality * 20  # Adjust the scaling factor as needed
                nt.add_node(node, label=node, size=node_size)

            # Add edges to the network
            for edge in betweenness_subgraph.edges:
                source = edge[0]
                target = edge[1]
                relationship = betweenness_subgraph.edges[edge]['relationship']
                nt.add_edge(source, target, label=relationship)

        nt.save_graph('html_files/betweenness_centrality_network.html')

        col1, col2 = st.columns([3, 1])

        with col1:
            with open('html_files/betweenness_centrality_network.html', 'r') as f:
                html_string = f.read()
                st.components.v1.html(html_string, height=960, scrolling=True)

        top_actors = []
        centrality_scores = []
        for betweenness_centrality in betweenness_centrality_values:
            top_actors += [actor for actor in betweenness_centrality if actor in betweenness_subgraph.nodes]
            centrality_scores += [betweenness_centrality[actor] for actor in top_actors if actor in betweenness_centrality]

        # Create a DataFrame of the top 10 actors based on betweenness centrality
        df_top_actors = pd.DataFrame({'Actor': top_actors, 'Betweenness Centrality': centrality_scores})
        df_top_actors = df_top_actors.sort_values(by='Betweenness Centrality', ascending=False).head(10)

        # Display the top 10 actors DataFrame
        with col2:
            st.subheader("Top 10 Actors based on Betweenness Centrality")
            st.dataframe(df_top_actors)

            x_pos = list(range(len(df_top_actors)))

            fig, ax = plt.subplots()
            ax.bar(x_pos, df_top_actors['Betweenness Centrality'])
            ax.set_xlabel('Actors')
            ax.set_ylabel('Betweenness Centrality')
            ax.set_title('Top 10 Actors based on Betweenness Centrality')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_top_actors['Actor'], rotation=45, ha='right')  # Display user names on the x-axis
            plt.tight_layout()

            # Display the plot in Streamlit
            st.pyplot(fig)




############################### CLOSENESS CENTRALITY ################################

    def visualize_closeness_centrality_network(graphs, tweet_user):
        nt = Network(height='790px', width='100%', bgcolor='#ffffff', font_color='#333333', directed=True)
        sorted_centralities = []

        for graph in graphs:
            closeness_centrality = nx.closeness_centrality(graph)
            sorted_centrality = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)
            sorted_centralities.append(sorted_centrality)

            # Add nodes to the network with size based on closeness centrality
            for node, centrality in sorted_centrality:
                node_size = centrality * 20  # Adjust the scaling factor as needed
                nt.add_node(node, label=node, size=node_size)

            # Add edges to the network
            for edge in graph.edges:
                source = edge[0]
                target = edge[1]
                relationship = graph.edges[edge]['relationship']
                nt.add_edge(source, target, label=relationship)

        nt.save_graph('html_files/closeness_centrality_network.html')

        top_actors = []
        centrality_scores = []
        for sorted_centrality in sorted_centralities:
            top_actors += [node for node, _ in sorted_centrality[:10]]
            centrality_scores += [centrality for _, centrality in sorted_centrality[:10]]

        # Create a DataFrame of the top 10 closeness centrality nodes
        df_top_actors = pd.DataFrame({'Actor': top_actors, 'Closeness Centrality': centrality_scores})

        message_closeness = "Di atas adalah representasi visual dari subgraf dengan sepuluh simpul teratas berdasarkan sentralitas kedekatan (closeness centrality).\n"
        # message_closeness += f"Sepuluh simpul teratas adalah: {', '.join(top_actors)}.\n"
        # message_closeness += f"Mereka memiliki nilai sentralitas kedekatan sebagai berikut: {', '.join(map(str, centrality_scores))}.\n"
        message_closeness += "Sentralitas kedekatan adalah ukuran seberapa cepat simpul dapat mencapai simpul lain dalam jaringan.\n"
        message_closeness += "Sentralitas kedekatan yang lebih tinggi menandakan bahwa simpul tersebut lebih sentral dan terhubung dengan baik dalam jaringan.\n"
        message_closeness += "Simpul-simpul dengan closeness centrality yang tinggi berperan sebagai aktor pendukung dengan menyokong dan menjaga soliditas lingkaran sosial mereka, daripada sebagai aktor penjembatan (hub)."
        
        

        col1, col2 = st.columns([2, 1])

        with col1:
            
            with open('html_files/closeness_centrality_network.html', 'r') as f:
                html_string = f.read()
            st.components.v1.html(html_string, height=800, scrolling=True)

            # Display the dynamic message for closeness centrality
            st.subheader("Detail Analisis Jaringan Sosial")
            st.caption(message_closeness)

        

        with col2:
            # Display the top 10 closeness centrality nodes DataFrame
            st.subheader("Top 10 Closeness Centrality Nodes")
            st.dataframe(df_top_actors)
            
            # Display the chart in Streamlit
            plt.figure(figsize=(10, 6))
            plt.bar(top_actors, centrality_scores)
            plt.ylabel('Closeness Centrality')
            plt.xlabel('Actors')
            plt.title('Top Actors Based on Closeness Centrality', fontsize=16)
            plt.tight_layout()
            st.pyplot(plt) 




##############################################################
    colvics1, colvics2, colvics3, colvics4 = st.tabs(['Social Network','Main Actors', 'Bridging Actors', 'Supporting Actors'])

    with colvics1:
        
        col1, col2=st.columns ([3,1])
        with col1:
            visualize_social_network(graphs)
            
            

        with col2:
           # Display the pandas dataframe with network statistics and implications
            st.subheader("Network Statistics:")
            st.dataframe(df_network_statistics)

                        # Tampilkan pembacaan analisis jaringan sosial
            st.subheader("Pembacaan Analisis Jaringan Sosial")
            st.write(f"Impikasi Jaringan: {network_implications}")
            st.write(f"Keterlibatan Jaringan: {network_engagement}")
            st.write(f"Jangkauan Jaringan: {network_reach}")
            st.write(f"Pengaruh Jaringan: {network_influence}")

            # Tampilkan rekomendasi otomatis
            st.subheader("Rekomendasi")
            for recommendation in network_recommendations:
                st.write(recommendation)


    with colvics2:
        # Call the visualize_degree_centrality_network function
        visualize_degree_centrality_network(graphs)

    with colvics3:

        visualize_betweenness_centrality_network(graphs)
    with colvics4:

        tweet_user = tweet["user"]["screen_name"]

        visualize_closeness_centrality_network(graphs, tweet_user)
        # Display the dynamic message for closeness centrality



#########################################################################
    import json
    import os
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

    # Function to preprocess the text data
    def preprocess_text_data(user_data):
        preprocessed_text_data = []
        stop_words = set(stopwords.words('indonesian'))
        stop_words.update(["and", "of", "the"])  # Add "and" and "of" to the set of stopwords
        lemmatizer = WordNetLemmatizer()

        for tweet in user_data['tweets']:
            text = tweet['full_text']
            # Convert text to lowercase
            text = text.lower()

            # Remove URLs
            text = re.sub(r'http\S+', '', text)

            # Remove double quotes
            text = text.replace('"', '')

            # Tokenize the text
            tokens = word_tokenize(text)

            # Remove stopwords, punctuation, and perform lemmatization
            processed_tokens = [
                lemmatizer.lemmatize(token)
                for token in tokens
                if token not in stop_words and token not in string.punctuation
            ]

            # Append the processed tokens to the preprocessed text data
            preprocessed_text_data.append(processed_tokens)

        return preprocessed_text_data


        # Function to perform topic modeling
    def perform_topic_modeling(text_data):
        # Create a dictionary from the text data
        dictionary = gensim.corpora.Dictionary(text_data)

        # Create a corpus (Bag of Words representation)
        corpus = [dictionary.doc2bow(text) for text in text_data]

        # Perform topic modeling using LDA
        lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)

        # Return the LDA model, corpus, and dictionary
        return lda_model, corpus, dictionary



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


    # Streamlit app
    st.title("Topic Modeling of Key Persons")

    # Folder path containing the JSON files
    folder_path = "twittl"

    # Check if the folder exists
    if not os.path.exists(folder_path):
        st.error("Folder 'twittl' does not exist.")
    else:
        # Get the list of JSON files in the folder
        file_list = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.json')]

        # Extract file names without the extension and "_data" suffix
        file_names_without_data = [file_name.replace('_data.json', '') for file_name in file_list]

        # Select a file
        selected_file = st.selectbox("Select an account", file_names_without_data)

        # Append '_data.json' to the selected file to get the actual file name
        selected_file_with_data = selected_file + '_data.json'

        # Load and preprocess the text data from the selected file
        file_path = os.path.join(folder_path, selected_file_with_data)
        with open(file_path, 'r') as f:
            user_data = json.load(f)

        text_data = preprocess_text_data(user_data)

        # Perform topic modeling on the preprocessed text data
        lda_model, corpus, dictionary = perform_topic_modeling(text_data)

        # Generate the pyLDAvis visualization
        lda_display = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word, sort_topics=False)

        # Save the pyLDAvis visualization as an HTML file
        html_string = pyLDAvis.prepared_data_to_html(lda_display)

        # Display the HTML in Streamlit
        st.components.v1.html(html_string, height=800, scrolling=False) 

        # Create the word cloud with keywords from the LDA model
        create_word_cloud_from_lda(lda_model)  # Use the correct function name 

        # # Create a word cloud
        # topics = lda_model.show_topics(num_topics=10, num_words=10, formatted=False)
        # keywords = [word for topic in topics for word, _ in topic[1]]
        # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))



######################## SENTIMENT ANALYSIS   ############################

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
    def load_and_preprocess_data(user_data, start_date, end_date):
        preprocessed_text_data = []
        stop_words = set(stopwords.words('indonesian'))
        stemmer = StemmerFactory().create_stemmer()

        for tweet in user_data['tweets']:
            created_at = tweet['created_at']
            created_at_date = datetime.strptime(created_at, '%a %b %d %H:%M:%S %z %Y').replace(tzinfo=None)

            if start_date <= created_at_date <= end_date:
                text = tweet['full_text']
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
    folder_path = "twittl"
    account_list = [file_name.split("_data.json")[0] for file_name in os.listdir(folder_path) if file_name.endswith('_data.json')]
    selected_accounts = st.multiselect("Select Accounts", account_list, default=account_list[:4], key="accsentf")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    if len(selected_accounts) == 0:
        st.warning("No accounts selected. Please choose at least one account.")
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



 ############################SENTIMENT ANALYSIS PER USER FOR ACCOUNT #############################
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
            user_data = json.load(f)

        preprocessed_text_data = []
        stop_words = set(stopwords.words('indonesian'))
        stemmer = StemmerFactory().create_stemmer()

        for tweet in user_data['tweets']:
            text = tweet['full_text']
            user = tweet['user']['name']  # Get the user name

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

    folder_path = "twittl"
    account_list = [file_name.split("_data.json")[0] for file_name in os.listdir(folder_path) if file_name.endswith('_data.json')]
    selected_accounts = st.multiselect("Select Accounts", account_list, default=account_list[:1], key="acsentselfil")
    start_date = st.date_input("Start Date", key="sentu4(t")
    end_date = st.date_input("End Date", key="3dinp)z")

    if len(selected_accounts) == 0:
        st.warning("No accounts selected. Please choose at least one account.")
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
                st.warning(f"There were no posts from {account} on {start_date} to {end_date}, so sentiment analysis cannot be performed.")
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
          

############################LOCATION ACCOUNT ################################

    import json
    import os
    import pandas as pd
    from datetime import datetime
    import folium
    from streamlit_folium import folium_static, st_folium
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderUnavailable
    import streamlit as st

    def load_tweet_data(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data['tweets']

    def filter_tweets_by_date(tweets, start_date, end_date):
        filtered_tweets = []
        for tweet in tweets:
            created_at = datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S %z %Y').replace(tzinfo=None)
            if start_date <= created_at <= end_date:
                filtered_tweets.append(tweet)
        return filtered_tweets

    def get_tweet_user_locations(tweets):
        user_locations = {}
        for tweet in tweets:
            user = tweet['user']
            user_name = user['screen_name']
            location = user['location']
            if location:
                if user_name not in user_locations:
                    user_locations[user_name] = {
                        'location': location,
                        'created_at': tweet['created_at']
                    }
        return user_locations

    st.title("Tweet User Locations")

    folder_path = "twittl"
    account_list = [file_name.split("_data.json")[0] for file_name in os.listdir(folder_path) if file_name.endswith('_data.json')]
    selected_accounts = st.multiselect("Select Accounts", account_list, default=account_list[:1], key="accselloc")
    start_date = st.date_input("Start Date", key="acl0c)sd")
    end_date = st.date_input("End Date", key='a(loc3d')

    # Join the account names using "and" as the separator
    formatted_accounts = " and ".join(selected_accounts)

    

    if len(selected_accounts) == 0:
        st.warning("No accounts selected. Please choose at least one account.")
    elif not start_date or not end_date:
        st.warning("Please select a start date and end date.")
    else:
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
        user_locations = {}

        for account in selected_accounts:
            file_name = f"{account}_data.json"
            file_path = os.path.join(folder_path, file_name)

            if not os.path.exists(file_path):
                st.warning(f"No data available for {account}.")
                continue

            tweets = load_tweet_data(file_path)
            filtered_tweets = filter_tweets_by_date(tweets, start_date, end_date)
            account_user_locations = get_tweet_user_locations(filtered_tweets)
            user_locations.update(account_user_locations)

        if not user_locations:
            st.warning(f"No tweets available for the selected accounts and date range.")
        else:
            df_user_locations = pd.DataFrame.from_dict(user_locations, orient='index')
            df_user_locations['User'] = df_user_locations.index
            df_user_locations = df_user_locations[['User', 'location', 'created_at']]

            # Display the formatted result using st.subheader
            st.subheader(f'Tweet User Locations of {formatted_accounts}')

            with st.expander("User Location Data"):
                st.dataframe(df_user_locations)

            location_counts = df_user_locations['location'].value_counts()

        
            # Create a Folium map centered around the first user location
            geolocator = Nominatim(user_agent="tweet_location_geocoder")
            first_location = df_user_locations['location'].iloc[0]
            location = geolocator.geocode(first_location)
            if location:
                lat, lon = location.latitude, location.longitude
                tweet_map = folium.Map(location=[lat, lon], zoom_start=6)
            else:
                tweet_map = folium.Map(location=[0, 0], zoom_start=2)

            # Add markers for each user location
            for _, row in df_user_locations.iterrows():
                location = row['location']
                user = row['User']
                tweet_time = row['created_at']
                geocode_result = geolocator.geocode(location, timeout=10)
                if geocode_result:
                    lat, lon = geocode_result.latitude, geocode_result.longitude
                    folium.Marker(
                        location=[lat, lon],
                        popup=f"User: {user}<br>Location: {location}<br>Time: {tweet_time}"
                    ).add_to(tweet_map)

            # Display the map
            folium_static(tweet_map, width=1300, height=600)



#######################HASHTAG ACCOUNT################################
    import json
    import glob
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    import datetime as dt

    # Function to extract hashtags from a tweet
    def extract_hashtags(tweet):
        if 'entities' in tweet and 'hashtags' in tweet['entities']:
            return [tag['text'] for tag in tweet['entities']['hashtags']]
        else:
            return []

    # Function to load tweets from files
    def load_tweets(file_paths):
        tweets = []
        for file_path in file_paths:
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                tweets.extend(json_data['tweets'])
        return tweets

    

    def perform_hashtag_analysis(file_paths, start_date, end_date):
        # Load tweets from files
        tweets = load_tweets(file_paths)

        # Filter tweets based on date range
        filtered_tweets = filter_tweets_by_date(tweets, start_date, end_date)

        # Extract hashtags from tweets
        hashtags = []
        for tweet in filtered_tweets:
            hashtags.extend(extract_hashtags(tweet))

        # Create a DataFrame with hashtag counts
        hashtag_counts = pd.Series(hashtags).value_counts().reset_index()
        hashtag_counts.columns = ['Hashtag', 'Count']

        return hashtag_counts

    # Get file paths in the "twittl" folder
    folder_path = 'twittl'
    file_paths = glob.glob(f"{folder_path}/*.json")
    file_names = [file_path.split('/')[-1] for file_path in file_paths]

    st.title("Hashtag Analysis of Tweets")

    # Get file paths in the "twittl" folder
    folder_path = 'twittl'
    file_paths = glob.glob(f"{folder_path}/*.json")
    file_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in file_paths]

    hashkpcol1, hashkpcol2, hashkpcol3 = st.columns([2, 1, 1])
    with hashkpcol1:
        selected_files = st.multiselect("Select account", file_names)
    with hashkpcol2:
        start_date = st.date_input("Start Date", key='hst5td')
    with hashkpcol3:
        end_date = st.date_input("End Date", key='hst3dt')

    def filter_tweets_by_date(tweets, start_date, end_date):
        filtered_tweets = []
        for tweet in tweets:
            created_at = dt.datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S %z %Y').replace(tzinfo=None)
            if start_date <= created_at <= end_date:
                filtered_tweets.append(tweet)
        return filtered_tweets
    
    # Perform hashtag analysis and display results
    if st.button("Perform Analysis"):
        if not selected_files:
            st.warning("Please select at least one file.")
        elif start_date > end_date:
            st.warning("Start Date should be before End Date.")
        else:
            for file_name in selected_files:
                file_path_selected = file_paths[file_names.index(file_name)]
                start_date = pd.Timestamp(start_date)
                end_date = pd.to_datetime(dt.datetime.today().date())
                hashtag_counts = perform_hashtag_analysis([file_path_selected], start_date, end_date)
                if len(hashtag_counts) > 0:
                    fig = px.bar(hashtag_counts, x='Hashtag', y='Count', title=f'Hashtag Analysis - {file_name}')
                    st.plotly_chart(fig)
                else:
                    st.info(f"No tweets found within the specified date range for {file_name}.")

    


##########################################################################



with tab2:
    import matplotlib.dates as mdates
    st.header('ISSUES ANALYSIS')

    folder_path = "twitkeys"

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]
    df = None

    if files:
        data = []
        for file in files:
            keyword = os.path.splitext(os.path.basename(file))[0]
            with open(file, 'r') as f:
                file_data = json.load(f)
                for tweet_data in file_data['data']:
                    tweet = tweet_data['Text']
                    created_at = pd.to_datetime(tweet_data['Created At'])
                    data.append({
                        'keyword': keyword,
                        'text': tweet,
                        'date': created_at
                    })
        df = pd.DataFrame(data)
        

    # Create a list of available keywords
    if df is not None:
        keywords = list(df['keyword'].unique())
    else:
        keywords = []

    # Set the default selected keywords to the first 4 keywords
    default_keywords = keywords[:4]

    # Set the default time range to one month from the current date
    end_date = pd.to_datetime(datetime.today().date())
    start_date = end_date - timedelta(days=30)

    # Create widgets for selecting the keywords and time range
    selected_keywords = st.multiselect('Select keywords to compare', keywords, default=default_keywords, key='selissue')
    cols_ta, cols_tb = st.columns([1, 1])
    start_date = pd.to_datetime(cols_ta.date_input('Start date', value=start_date, key='start_date(XT')).date()
    end_date = pd.to_datetime(cols_tb.date_input('End date', value=end_date, key='end_date^H')).date()

    # Filter the data based on the selected keywords and time range
    if df is not None:
        mask = (df['keyword'].isin(selected_keywords)) & (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
        df_filtered = df.loc[mask]
    else:
        df_filtered = pd.DataFrame()

    if len(df_filtered) > 0:
        df_grouped = df_filtered.groupby(['date', 'keyword']).size().reset_index(name='count')
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.lineplot(data=df_grouped, x='date', y='count', hue='keyword', ax=ax)

        # Set the date format for the x-axis labels to 'YYYY-MM-DD'
        date_format = '%Y-%m-%d'
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))

        # Rotate the x-axis labels for better readability
        plt.xticks(rotation=0)

        ax.set_title(f"Tweets per Day for {', '.join(selected_keywords)}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Tweets")
        st.pyplot(fig)
    else:
        st.write("No data available for the selected time range and keywords.")

    ################ ISSUE SNA ####################

    st.header('TWITTER NETWORK ANALYSIS OF ISSUES')


    import glob
    from datetime import datetime, date, timezone, timedelta

    def process_json_files(files, start_date=None, end_date=None):
        G = nx.DiGraph()

        if start_date:
            start_date = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)

        if end_date:
            end_date = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)

        for file in files:
            with open(file, 'r') as f:
                file_data = json.load(f)
                for tweet_data in file_data['data']:
                    user_screen_name = tweet_data['User Screen Name']
                    mentioned_users = [user['User Screen Name'] for user in tweet_data['Mentioned Users']]
                    retweeted_user = tweet_data['Retweeted Tweet']['User Screen Name'] if 'Retweeted Tweet' in tweet_data and tweet_data['Retweeted Tweet'] else None
                    hashtags = tweet_data['Hashtags']
                    
                    tweet_date = datetime.strptime(tweet_data['Created At'], '%Y-%m-%dT%H:%M:%S%z')

                    if start_date and tweet_date < start_date:
                        continue
                    if end_date and tweet_date > end_date:
                        continue

                    # Add nodes to the graph
                    if not G.has_node(user_screen_name):
                        G.add_node(user_screen_name)
                    if retweeted_user and not G.has_node(retweeted_user):
                        G.add_node(retweeted_user)

                    # Group node for mention, reply, and retweeted users
                    group_node = 'Group_' + user_screen_name
                    if not G.has_node(group_node):
                        G.add_node(group_node, relationship='group')  # Add the group node to the graph

                    # Connect individual users to the group node
                    if mentioned_users:
                        G.add_edge(group_node, user_screen_name, relationship='mentioned_group')
                    if 'Replies' in tweet_data and tweet_data['Replies']:
                        for reply in tweet_data['Replies']:
                            replies_user = reply.get('User Screen Name')
                            if replies_user and not G.has_node(replies_user):
                                G.add_node(replies_user)
                            if replies_user:
                                G.add_edge(group_node, replies_user, relationship='replies_group')
                    if retweeted_user and not G.has_node(retweeted_user):
                        G.add_node(retweeted_user)
                        G.add_edge(group_node, retweeted_user, relationship='retweeted_group')

                    for hashtag in hashtags:
                        # Connect users who have mentioned or used the same hashtag
                        users_with_same_hashtag = [node for node in G.nodes if G.nodes[node].get('relationship') == 'mentioned' and hashtag in G.edges[user_screen_name, node]['relationship']]
                        for user in users_with_same_hashtag:
                            G.add_edge(user_screen_name, user, relationship=hashtag)

        return G

        
    
    # Folder path containing the JSON files
    folder_path = "twitkeys"

    # Get the list of JSON files in the folder
    file_list = glob.glob(os.path.join(folder_path, '*.json'))

    # Create the social network graph
    G = process_json_files(file_list)


    def visualize_social_network(G):
        nt = net(height='750px', width='100%', bgcolor='#ffffff', font_color='#333333', directed=True)

        # Define a dictionary to map relationships to colors
        relationship_colors = {
            'group': '#FF6666',  # Red color for group nodes and edges
            'mentioned_group': '#00FF00',  # Green color for mentioned users
            'replies_group': '#45CFDD',  # Blue color for replied users
            'retweeted_group': '#FF00FF',  # Purple color for retweeted users
            'hashtag': '#FFFF00'  # Yellow color for hashtag-related nodes and edges
        }

        # Helper function to check if a node exists in the graph
        def node_exists(node):
            return node in nt.get_nodes()

        # Calculate the frequency of each relationship in the graph
        relationship_frequency = {}
        for edge in G.edges:
            relationship = G.edges[edge]['relationship']
            relationship_frequency[relationship] = relationship_frequency.get(relationship, 0) + 1

        # Find the maximum frequency among all relationships
        max_frequency = max(relationship_frequency.values())

        for node in G.nodes:
            relationship = G.nodes[node].get('relationship')
            node_color = relationship_colors.get(relationship, '#9681EB')  # Default to black color for nodes without a defined relationship
            if G.nodes[node].get('relationship') == 'group' and not node_exists(node):
                nt.add_node(node, label=node, color=node_color)  # Set color for group nodes
            else:
                nt.add_node(node, label=node, color=node_color)

        for edge in G.edges:
            source = edge[0]
            target = edge[1]
            relationship = G.edges[edge]['relationship']

            if not node_exists(source):
                nt.add_node(source, label=source)  # Add the source node if it doesn't exist

            if not node_exists(target):
                nt.add_node(target, label=target)  # Add the target node if it doesn't exist

            edge_color = relationship_colors.get(relationship, '#000000')  # Default to black color for edges without a defined relationship
            width = 1 + 4 * (relationship_frequency[relationship] / max_frequency)  # Adjust the scaling factor as needed

            nt.add_edge(source, target, label=relationship, color=edge_color, width=width)

        nt.save_graph('html_files/issue_social_network.html')


    # Display the graph in Streamlit
        with open('html_files/issue_social_network.html', 'r') as f:
            html_string = f.read()
            st.components.v1.html(html_string, height=960, scrolling=True)

    seldttwitissuecol0,seldttwitissuecol1,seldttwitissuecol2=st.columns([2,1,1])

    with seldttwitissuecol0:
        default_files = [os.path.splitext(os.path.basename(file))[0] for file in files[:4]] if len(files) >= 4 else [os.path.splitext(os.path.basename(file))[0] for file in files]
        selected_files = st.multiselect('Select Issue/Topic', [os.path.splitext(os.path.basename(file))[0] for file in files], default=default_files, format_func=lambda x: f"{x}.json")

    # Calculate default start date and end date
    default_end_date = datetime.now()
    default_start_date = default_end_date - timedelta(days=30)

    # Set default time to midnight (00:00:00) for both start and end dates
    default_start_date = default_start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    default_end_date = default_end_date.replace(hour=0, minute=0, second=0, microsecond=0)

    # Date input widgets with default values
    with seldttwitissuecol1:
        start_date = st.date_input("Start Date", value=default_start_date, key='sd5nissu3twit')
    with seldttwitissuecol2:
        end_date = st.date_input("End Date", value=default_end_date, key='endtwi55uesn')


    # Process the selected JSON files and build the social network graph
    selected_files_paths = [os.path.join(folder_path, f"{file}.json") for file in selected_files]
    # Create the social network graph based on all JSON files in the folder
    G = process_json_files(file_list)

    # Process the selected JSON files and build the social network graph for selected files
    selected_G = process_json_files(selected_files_paths, start_date, end_date)

    # Visualize the social network
    # visualize_social_network(selected_G)

    ####################DEGREE CENTRALITY###########################
    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(selected_G)

    # Create a subgraph with nodes having non-zero degree centrality
    degree_subgraph = selected_G.subgraph([node for node, centrality in degree_centrality.items() if centrality > 0])

    # Function to visualize the degree centrality network and top actors using Pyvis and matplotlib
    def visualize_degree_centrality_network(subgraph, centrality_values):
        nt = net(height='750px', width='100%', bgcolor='#ffffff', font_color='#333333', directed=True)

        # Add nodes to the network with size based on degree centrality
        for node in subgraph.nodes:
            centrality = centrality_values[node]
            node_size = centrality * 20  # Adjust the scaling factor as needed
            nt.add_node(node, label=node, size=node_size)

        # Add edges to the network
        for edge in subgraph.edges:
            source = edge[0]
            target = edge[1]
            relationship = subgraph.edges[edge]['relationship']
            nt.add_edge(source, target, label=relationship)

        nt.save_graph('html_files/degree_centrality_network.html')

        

        with open('html_files/degree_centrality_network.html', 'r') as f:
            html_string = f.read()
            st.components.v1.html(html_string, height=960, scrolling=True)

        # Calculate and plot the top actors based on degree centrality
        top_actors = sorted(centrality_values, key=centrality_values.get, reverse=True)[:5]
        centrality_scores = [centrality_values[actor] for actor in top_actors]

        y_pos = np.arange(len(top_actors))

        fig, ax = plt.subplots()
        ax.barh(y_pos, centrality_scores)
        ax.set_xlabel('Degree Centrality')
        ax.set_ylabel('Top Actors')
        ax.set_title('Top Main Actors')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_actors)
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)
        plt.close()


################## BETWEENESS CENTRALITY ##########################
    # Calculate betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(selected_G)

    # Create a subgraph with nodes having non-zero betweenness centrality
    betweenness_subgraph = selected_G.subgraph([node for node, centrality in betweenness_centrality.items() if centrality > 0])



    # Function to visualize the betweenness centrality network and top actors using Pyvis and matplotlib
    def visualize_betweenness_centrality_network(subgraph, centrality_values):
        nt = net(height='750px', width='100%', bgcolor='#ffffff', font_color='#333333', directed=True)

        # Add nodes to the network with size based on betweenness centrality
        for node in subgraph.nodes:
            centrality = centrality_values[node]
            node_size = centrality * 20  # Adjust the scaling factor as needed
            nt.add_node(node, label=node, size=node_size)

        # Add edges to the network
        for edge in subgraph.edges:
            source = edge[0]
            target = edge[1]
            relationship = subgraph.edges[edge]['relationship']
            nt.add_edge(source, target, label=relationship)

        nt.save_graph('html_files/betweenness_centrality_network.html')

        
        with open('html_files/betweenness_centrality_network.html', 'r') as f:
            html_string = f.read()
            st.components.v1.html(html_string, height=960, scrolling=True)

        # Calculate and plot the top actors based on betweenness centrality
        top_actors = sorted(centrality_values, key=centrality_values.get, reverse=True)[:5]
        centrality_scores = [centrality_values[actor] for actor in top_actors]

        y_pos = np.arange(len(top_actors))

        fig, ax = plt.subplots()
        ax.barh(y_pos, centrality_scores)
        ax.set_xlabel('Betweenness Centrality')
        ax.set_ylabel('Actors')
        ax.set_title('Top Actors based on Betweenness Centrality')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_actors)
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(fig)
        plt.close()

##################################################################
    def calculate_closeness_centrality(G):
        closeness_centrality = nx.closeness_centrality(G)

        return closeness_centrality
    
      # Calculate closeness centrality
    closeness_centrality = calculate_closeness_centrality(selected_G)

        # Sort the centrality values in descending order
    sorted_centrality = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)

        # Extract the top nodes and their centrality values
    top_nodes = [node for node, centrality in sorted_centrality[:10]]
    top_centrality = [centrality for node, centrality in sorted_centrality[:10]]
        # Function to visualize the closeness centrality network using Pyvis
    def visualize_closeness_centrality_network(G, centrality_values):
        nt = net(height='750px', width='100%', bgcolor='#ffffff', font_color='#333333', directed=True)

        # Add nodes to the network with size based on closeness centrality
        for node in G.nodes:
            centrality = centrality_values[node]
            node_size = centrality * 20  # Adjust the scaling factor as needed
            nt.add_node(node, label=node, size=node_size)

        # Add edges to the network
        for edge in G.edges:
            source = edge[0]
            target = edge[1]
            relationship = G.edges[edge]['relationship']
            nt.add_edge(source, target, label=relationship)

        nt.save_graph('html_files/closeness_centrality_network.html')
        with open('html_files/closeness_centrality_network.html', 'r') as f:
            html_string = f.read()
            st.components.v1.html(html_string, height=960, scrolling=True)

        plt.figure(figsize=(10, 6))
        plt.bar(top_nodes, top_centrality)
        plt.xlabel('Nodes')
        plt.ylabel('Closeness Centrality')
        plt.title('Top Actors Based on Closeness Centrality')
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Display the chart in Streamlit
        st.pyplot(plt)
        plt.close()

    print("selected G nodes", selected_G.nodes())
    for node in selected_G.nodes():
        print(node, selected_G.nodes[node])


############################VISUGRAPJH###########################
    colviz1, colviz2, colviz3, colviz4=st.tabs(['Social Network','Main Actors', 'Bridging Actors', 'Supporting Actors'])
    with colviz1:
         visualize_social_network(selected_G)
        
    with colviz2:
        visualize_degree_centrality_network(degree_subgraph, degree_centrality)

    with colviz3:
        visualize_betweenness_centrality_network(betweenness_subgraph, betweenness_centrality)
    
    with colviz4:
        visualize_closeness_centrality_network(selected_G, closeness_centrality)

    

######################TOPIC MODELING#############################################

    import json
    import os
    import gensim
    import nltk
    import pyLDAvis
    import streamlit as st
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from datetime import datetime
    import string
    import re
   

    # Function to preprocess the text data
    def preprocess_text_data(data):
        preprocessed_text_data = []
        stop_words = set(stopwords.words('indonesian'))
        stop_words.update(['rt', 'yg', 'sih', 'dan'])  # Add "rt" and "yg" to the stop words
        lemmatizer = WordNetLemmatizer()

        for tweet_data in data['data']:
            text = tweet_data['Text']

            # Convert text to lowercase
            text = text.lower()

            # Remove URLs
            text = re.sub(r'http\S+', '', text)

            # Use regular expression to remove unwanted characters
            text = re.sub(r'[\"@():,.#?!_*]', '', text)

            # Tokenize the text
            tokens = word_tokenize(text)

            # Remove stopwords and perform lemmatization, excluding "rt" and "yg"
            processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

            # Append the processed tokens to the preprocessed text data
            preprocessed_text_data.append(processed_tokens)

        return preprocessed_text_data
    
   

    class ComplexEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, complex):
                return str(obj)  # Convert complex numbers to strings
            return super().default(obj)


    # Function to perform topic modeling
    def perform_topic_modeling(text_data):
    # Create a dictionary from the text data
        dictionary = gensim.corpora.Dictionary(text_data)

        # Create a corpus (Bag of Words representation)
        corpus = [dictionary.doc2bow(text) for text in text_data]

        # Perform topic modeling using LDA
        lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)

        # Return the LDA model, corpus, and dictionary
        return lda_model, corpus, dictionary


    # Function to create a word cloud from keywords
    def create_word_cloud(lda_model, dictionary):
        topics = lda_model.show_topics(num_topics=10, num_words=10, formatted=False)

        # Extract keywords from topics
        keywords = [word for topic in topics for word, _ in topic[1]]

        # Create a word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(keywords))

        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Keywords Word Cloud')
        st.pyplot(plt)
        plt.close()


    # Custom encoder to handle complex numbers
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, complex):
                return str(obj)  # Convert complex numbers to strings
            return super().default(obj)

    # Streamlit app
    st.title("Topic Modeling of Issue")

    # Folder path containing the JSON files
    folder_path = "twitkeys"

    # Check if the folder exists
    if not os.path.exists(folder_path):
        st.error("Folder 'twitkeys' does not exist.")
    else:
        # Get the list of JSON files in the folder
        file_list = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.json')]

        # Select a file
        selected_file = st.selectbox("Select a File", file_list)

        # Load and preprocess the text data from the selected file
        file_path = os.path.join(folder_path, selected_file)
        with open(file_path, 'r') as f:
            data = json.load(f)

        text_data = preprocess_text_data(data)

        # Perform topic modeling on the preprocessed text data
        lda_model, corpus, dictionary = perform_topic_modeling(text_data)

        # Generate the pyLDAvis visualization
        # lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary=lda_model.id2word, sort_topics=False)

        lda_display = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word, sort_topics=False)

        # Save the pyLDAvis visualization as an HTML file
        html_string = pyLDAvis.prepared_data_to_html(lda_display)

        # Display the HTML in Streamlit
        st.components.v1.html(html_string, height=800, scrolling=False)

        # # Save the pyLDAvis visualization as an HTML file
        # html_file = "lda_visualization.html"
        # with open(html_file, 'w') as f:
        #     f.write(pyLDAvis.prepared_data_to_html(lda_display))

        # # Display the saved HTML file using Streamlit's components API
        # st.components.v1.html(open(html_file, 'r').read())

        # # Display the HTML in Streamlit
        # st.components.v1.html(open(file_name, 'r').read(), height=800, scrolling=False)

        # Create and display the word cloud
        create_word_cloud(lda_model, dictionary=dictionary)





#######################HASHTAG ACCOUNT################################
    st.header('HASHTAGS ANALYSIS OF ISSUES')
  

    import streamlit as st
    import pandas as pd
    import os
    import json
    import plotly.express as px
    from datetime import datetime, timedelta

    # Function to load data from a JSON file and filter by date
    def load_data(file_path, start_date, end_date):
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        df = pd.DataFrame(data['data'])
        df['Created At'] = pd.to_datetime(df['Created At'])
        mask = (df['Created At'] >= start_date) & (df['Created At'] <= end_date)
        return df[mask]

    # Function to generate the hashtag analysis chart
    def generate_hashtag_chart(df):
        hashtags = df['Hashtags'].apply(', '.join).str.split(', ', expand=True).stack()
        hashtag_counts = hashtags.value_counts().reset_index()
        hashtag_counts.columns = ['hashtag', 'count']

        fig = px.bar(hashtag_counts, x='count', y='hashtag', orientation='h', title='Hashtag Analysis')
        return fig

    # Get the list of files in the "twikeys" folder
    file_list = os.listdir('twitkeys')
    selected_files = st.multiselect('Select files', [os.path.splitext(file)[0] for file in file_list])

    # Allow the user to set start and end dates
    start_date = st.date_input('Start Date', key='hst1ssust')
    end_date = st.date_input('End Date', key='edhast7iss')

    if selected_files and start_date and end_date:
        # Convert the start_date and end_date to datetime with UTC timezone
        start_date = pd.Timestamp(start_date).tz_localize('UTC')
        end_date = pd.Timestamp(end_date).tz_localize('UTC') + timedelta(days=1) - timedelta(seconds=1)

        # Create dynamic columns for each selected file
        num_cols = len(selected_files)
        columns = st.columns(num_cols)

        for idx, file in enumerate(selected_files):
            columns[idx].write(f"## Analysis for {file}")
            file_path = os.path.join('twitkeys', f"{file}.json")
            df = load_data(file_path, start_date, end_date)
            columns[idx].plotly_chart(generate_hashtag_chart(df))
    else:
        st.warning("Please select at least one file and set start and end dates.")



################################ SENTIMENT ANALYSIS#################################

    from nltk.sentiment import SentimentIntensityAnalyzer
    import numpy as np

    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


    # Function to load and preprocess the text data
    @st.cache_data
    def load_and_preprocess_data(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Preprocess the text (e.g., lowercase, tokenization, stopwords removal, etc.)
        preprocessed_text_data = []
        stop_words = set(stopwords.words('indonesian'))
        stemmer = StemmerFactory().create_stemmer()

        for tweet_data in data['data']:
            text = tweet_data['Text']

            # Convert text to lowercase
            text = text.lower()

            # Tokenize the text
            tokens = word_tokenize(text)

            # Remove stopwords and perform stemming
            processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

            # Append the processed tokens to the preprocessed text data
            preprocessed_text_data.append(processed_tokens)

        # Return the preprocessed text data
        return preprocessed_text_data


    # Function to perform sentiment analysis using VADER
    @st.cache_data
    def perform_sentiment_analysis(text_data_list):
        # Initialize the VADER sentiment intensity analyzer
        sia = SentimentIntensityAnalyzer()

        sentiment_scores_list = []
        for text_data in text_data_list:
            # Perform sentiment analysis on each text data
            sentiment_scores = []
            for text in text_data:
                sentiment_score = sia.polarity_scores(' '.join(text))
                sentiment_scores.append(sentiment_score)

            # Append the sentiment scores of the current file to the list
            sentiment_scores_list.append(sentiment_scores)

        # Convert the sentiment scores to a list of DataFrames
        df_sentiment_list = [pd.DataFrame(scores) for scores in sentiment_scores_list]

        # Return the list of DataFrames with sentiment scores
        return df_sentiment_list


    # Main Streamlit app
    st.title("Sentiment Analysis")

    # Folder path containing the JSON files
    folder_path = "twitkeys"

    # Get the list of files in the "twikeys" folder
    file_list = os.listdir('twitkeys')
    selected_files = st.multiselect('Select files', [os.path.splitext(file)[0] for file in file_list], key='sel5entissuetwit')

    # Date selection
    start_date = st.date_input('Start Date', key='j4ncuk')
    end_date = st.date_input('End Date', key='ban5at')

    # Check if any files and dates are selected
    if len(selected_files) == 0 or start_date is None or end_date is None:
        st.warning("Please choose at least one file and select a date range.")
    else:
        # Define the number of columns based on the number of selected files
        num_columns = len(selected_files)

        # Create a grid layout with the specified number of columns
        columns = st.columns(num_columns)

        # Load and preprocess the text data for each selected file
        text_data_list = []
        for file_name in selected_files:
            # File path of the current file
            file_path = os.path.join('twitkeys', f"{file_name}.json")

            if not os.path.isfile(file_path):
                st.warning(f"File '{file_name}.json' not found.")
                continue

            # Load and preprocess the text data from the file
            text_data = load_and_preprocess_data(file_path)

            # Append the preprocessed text data to the list
            text_data_list.append(text_data)

        # Perform sentiment analysis on the preprocessed text data for each file
        df_sentiment_list = perform_sentiment_analysis(text_data_list)

        # Calculate the sentiment distribution for each file
        sentiment_distributions = []
        for df_sentiment in df_sentiment_list:
            sentiment_distribution = df_sentiment.mean().drop("compound")
            sentiment_distributions.append(sentiment_distribution)

        # Iterate over the selected files and display sentiment analysis results and charts
        for i, file_name in enumerate(selected_files):
            df_sentiment = df_sentiment_list[i]
            sentiment_distribution = sentiment_distributions[i]

            # Display the sentiment analysis results in the current column
            with columns[i]:
                # Plot the sentiment distribution as a pie chart
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.pie(sentiment_distribution.values, labels=sentiment_distribution.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                ax.set_title(f"Sentiment Distribution: {file_name}")

                # Display the chart
                st.pyplot(fig)
                with st.expander(""):
                    st.dataframe(df_sentiment)





################################# SENTIMENT ANALYSIS PER USER PER FILES  ##############################
    @st.cache_data
    def load_and_preprocess_data(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Preprocess the text (e.g., lowercase, tokenization, stopwords removal, etc.)
        preprocessed_text_data = []
        stop_words = set(stopwords.words('indonesian'))
        stemmer = StemmerFactory().create_stemmer()

        for tweet_data in data['data']:
            text = tweet_data['Text']
            user = tweet_data['User Name']  # Get the user name

            # Convert text to lowercase
            text = text.lower()

            # Tokenize the text
            tokens = word_tokenize(text)

            # Remove stopwords and perform stemming
            processed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

            # Append the processed tokens and the user to the preprocessed text data
            preprocessed_text_data.append((processed_tokens, user))

        # Return the preprocessed text data
        return preprocessed_text_data

       # Perform sentiment analysis per user
    @st.cache_data
    def perform_sentiment_analysis_per_user(text_data):
        # Initialize the VADER sentiment intensity analyzer
        sia = SentimentIntensityAnalyzer()

        # Create a dictionary to store sentiment scores per user
        user_sentiment_scores = {}

        # Perform sentiment analysis on each text per user
        for text, user in text_data:
            sentiment_score = sia.polarity_scores(' '.join(text))

            # Add the sentiment score to the user's scores
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

        # Calculate the average sentiment scores per user
        user_sentiment_scores_avg = {}
        for user, scores in user_sentiment_scores.items():
            user_sentiment_scores_avg[user] = {
                'positive': np.mean(scores['positive']),
                'negative': np.mean(scores['negative']),
                'neutral': np.mean(scores['neutral']),
                'compound': np.mean(scores['compound'])
            }

        # Convert the sentiment scores to a DataFrame
        df_sentiment_per_user = pd.DataFrame.from_dict(user_sentiment_scores_avg, orient='index')

        # Return the DataFrame with sentiment scores per user
        return df_sentiment_per_user

   # Main Streamlit app
    st.title("Sentiment Analysis per user")

    # Folder path containing the JSON files
    folder_path = "twitkeys"

    # Get the list of JSON files in the folder
    file_list = [os.path.splitext(file_name)[0] for file_name in os.listdir(folder_path) if file_name.endswith('.json')]

    # Select files
    selected_files = st.multiselect("Select Files", file_list, default=file_list[:1], key="file_selector_sent")

    # Iterate over the selected files
    for file_name in selected_files:
        # List to store preprocessed text data from the current file
        preprocessed_text_data = []

        # File path of the current file
        file_path = os.path.join(folder_path, file_name + ".json")  # Add the '.json' extension back

        # Load and preprocess the text data from the file
        text_data = load_and_preprocess_data(file_path)

        # Append the preprocessed text data to the list
        preprocessed_text_data.extend(text_data)

        # Perform sentiment analysis per user on the preprocessed text data
        df_sentiment_per_user = perform_sentiment_analysis_per_user(preprocessed_text_data)

        # Display the sentiment analysis results per user
        st.subheader(f"Sentiment Analysis per User: {file_name}")

        with st.expander (""):
            st.dataframe(df_sentiment_per_user)

        # Plot the sentiment scores per user as a bar chart
        ax = df_sentiment_per_user.plot(kind='bar', rot=0, fontsize=5)
        plt.xlabel('User', fontsize=7)
        plt.ylabel('Sentiment Score', fontsize=7)
        plt.title(f"Sentiment Analysis per User: {file_name}")
        plt.xticks(rotation='vertical', fontsize=5)
        plt.legend(fontsize=5)
        plt.tight_layout()

        # Modify the text of user in the bar chart
        for p in ax.patches:
            ax.annotate(str(round(p.get_height(), 2)), (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=5)

        st.pyplot(plt)
    

######################### LOCATION ##############################

   
    import folium
    from streamlit_folium import folium_static, st_folium
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderUnavailable
    import os
    import time
    
    st.header('Location')
    # Create a geolocator object
    geolocator = Nominatim(user_agent='twitter_map_app')

    # Define a function to perform geocoding with caching
    @st.cache_data
    def geocode_location(location):
        try:
            location_data = geolocator.geocode(location, timeout=5)  # Increase the timeout value as needed
            if location_data:
                return location_data.latitude, location_data.longitude
        except GeocoderUnavailable:
            st.warning(f"Geocoding service is unavailable for location: {location}")
        return None, None
    
    

    # Get the file paths of all JSON files in the "twitkeys" folder
    file_paths = glob.glob('twitkeys/*.json')

    # Sort the file paths by modification time (newest to oldest)
    file_paths.sort(key=os.path.getmtime, reverse=True)

    # Select the four newest files
    default_files = file_paths[:1]

    # Allow users to select multiple files using a multiselect widget
    selected_files = st.multiselect("Select keyword", file_paths, default=default_files)

    for file_path in selected_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract the filename without extension

    # Define variables to store the min/max latitude and longitude
    min_latitude = float('inf')
    max_latitude = float('-inf')
    min_longitude = float('inf')
    max_longitude = float('-inf')

    # Iterate over the selected files
    for file_path in selected_files:
        with open(file_path, 'r') as f:
            data = json.load(f)

        user_data = data['data']

        # Perform geocoding for each user location
        for user in user_data:
            location = user.get('User Location')
            if location:
                latitude, longitude = geocode_location(location)
                user['Latitude'] = latitude
                user['Longitude'] = longitude
                time.sleep(1)  # Add a 1-second delay between requests

                # Update the min/max latitude and longitude
                if latitude is not None:
                    min_latitude = min(min_latitude, latitude)
                    max_latitude = max(max_latitude, latitude)
                if longitude is not None:
                    min_longitude = min(min_longitude, longitude)
                    max_longitude = max(max_longitude, longitude)

    # Calculate the center latitude and longitude
    center_latitude = (min_latitude + max_latitude) / 2
    center_longitude = (min_longitude + max_longitude) / 2

    # Create a Folium map object
    m = folium.Map(location=[center_latitude, center_longitude], zoom_start=2)

    # Add markers to the map
    for user in user_data:
        latitude = user.get('Latitude')
        longitude = user.get('Longitude')
        user_name = user.get('User Name')

        if latitude is not None and longitude is not None:
            popup = f"User: {user_name}\nLocation: {user['User Location']}"
            folium.Marker([latitude, longitude], popup=popup, tooltip=user_name).add_to(m)

    # Display the map for the current file
    st.subheader(f"User Map in The Conversation on {file_name}")
    st_folium(m, width=1500, height=600)

#################################################################
    

    st.title("Gender Prediction from Twitter Data")
    st.header("Predicted Gender")

    
    def preprocess_tweet(tweet, training_columns):
        processed_tweet = {
            'User Name': str(tweet[0]),
            'User Description': str(tweet[1]).lower() if tweet[1] else '',
            'Text': str(tweet[2]).lower(),
        }

        processed_tweet = {k: float(v) if isinstance(v, str) and v.isnumeric() else v for k, v in processed_tweet.items()}
        processed_tweet = pd.DataFrame(processed_tweet, index=[0])

        # Perform one-hot encoding on the categorical variables
        processed_tweet_encoded = pd.get_dummies(processed_tweet)
        processed_tweet_encoded = processed_tweet_encoded.reindex(columns=training_columns, fill_value=0)

        return processed_tweet_encoded.values.flatten()

    
    def predict_gender(model, features, training_columns):
        processed_tweet = preprocess_tweet(features, training_columns)
        prediction = model.predict([processed_tweet])
        return prediction[0]



    dfout = pd.read_json('output1.json')

    # st.dataframe(dfout)
    # print ("DFOUT:",dfout)

    # Prepare the data
    if 'Gender' in dfout.columns:
        X = dfout.drop('Gender', axis=1)
    else:
        X = dfout.copy()

    if 'Gender' in dfout.columns:
        y = dfout['Gender']
    else:
        # Handle the case when 'Gender' column is missing
        # For example, you can print an error message or take appropriate action
        st.write('Gender not in df.columns')

    # Perform one-hot encoding on the categorical variables in X
    X_encoded = pd.get_dummies(X)

    # Get the training columns from the X_encoded DataFrame
    training_columns = X_encoded.columns

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Create a Gradient Boosting Classifier model
    model = GradientBoostingClassifier()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Predict the gender for the test data
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    st.write('Accuracy:', accuracy)

    # Save the trained model to a file
    joblib.dump(model, 'modelgend.pkl')

    # Get the file paths of all JSON files in the "twitkeys" folder
    file_paths = glob.glob('twitkeys/*.json')

    # Sort the file paths by modification time (newest to oldest)
    file_paths.sort(key=os.path.getmtime, reverse=True)

    file_list = [os.path.splitext(os.path.basename(file_path))[0] for file_path in file_paths]

    # Select files
    selected_files = st.multiselect("Select Files", file_list, default=file_list[:1], key="gensel")

    data_list = []

    # Define the number of columns based on the number of selected files
    num_columns = len(selected_files)

    # Create a grid layout with the specified number of columns
    columns = st.columns(num_columns)

    for i, file_name in enumerate(selected_files):
        file_path = os.path.join('twitkeys', file_name + '.json')

        with open(file_path, 'r') as file:
            json_data = json.load(file)
            data_list.extend(json_data["data"])

        dfgend = pd.DataFrame(data_list)
        # Drop irrelevant columns
        columns_to_drop = ["User Screen Name", "User Location", "Hashtags", "Source", "In Reply To", "Mentioned Users",
                        "Tweet URL", "Created At", "User Location", "Retweet Count", "Reply Count", "Mention Count",
                        "Longitude", "Latitude", "Replies", "Retweeted Tweet", "Tweet ID", "Profile Image URL"]
        dfgend = dfgend.drop(columns_to_drop, axis=1)
        dfgend['Gender'] = ''

        dfgend = dfgend.drop_duplicates(subset='User Name')

        # Load the model from the output.json file
        model = joblib.load('modelgend.pkl')

        # Predict gender for each tweet
        for index, tweet in dfgend.iterrows():
            features = [tweet['User Name'], tweet['User Description'], tweet['Text']]
            processed_tweet = preprocess_tweet(features, training_columns)
            prediction = predict_gender(model, processed_tweet, training_columns)
            dfgend.at[index, 'Gender'] = prediction

        # Group by gender to get gender distribution
        gender_counts = dfgend.groupby('Gender').size().reset_index(name='Count')

        # Create a pie chart for gender distribution in the corresponding column
        with columns[i]:
            fig = px.pie(gender_counts, values='Count', names='Gender', title='Gender Distribution - ' + file_name)
            st.plotly_chart(fig)


    

##################################################################
with tab3:
    st.header("Data Mining")
    container3=st.container()
    with container3:
       
          
        colta, coltb = st.columns([2, 2])
        with colta:
            import pytz
            timezone = pytz.timezone('Asia/Jakarta')  # Replace 'YOUR_TIMEZONE' with your desired timezone
            
            with st.form(key="taccountform"):
                accounts = st.text_input(
                    label='# Enter Account:',
                    value='',
                    key='1'
                )
                account_list = accounts.split('\n')  # Split input into a list of accounts

                start_date = st.date_input("Start Date")
                end_date = st.date_input("End Date")

                submit = st.form_submit_button(label="Submit")
                if submit:
                    for account in account_list:
                        try:
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
                            # Calculate the start and end dates
                            days_ago = 10  # Number of days ago from today
                            start_date = datetime.now() - timedelta(days=days_ago)
                            end_date = datetime.now()

                            # Convert start_date and end_date to offset-aware datetimes
                            start_date = timezone.localize(start_date)
                            end_date = timezone.localize(end_date)

                            # get the user's tweets within the specified date range
                            tweets = api.user_timeline(screen_name=account, count=50, tweet_mode='extended')
                            tweets_list = [tweet._json for tweet in tweets if tweet.created_at >= start_date and tweet.created_at <= end_date]

                            # search for tweets mentioning the user within the specified date range
                            mention_tweets = api.search_tweets(q=f"@{account}", count=10, tweet_mode='extended')
                            mention_tweets_list = [tweet._json for tweet in mention_tweets if tweet.created_at >= start_date and tweet.created_at <= end_date]

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
                                with open(file_path, 'w') as json_file:
                                    json.dump(user_data, json_file)
                        except tweepy.TweepyException as e:
                            st.error(f"Error occurred for account: {account}. Error message: {str(e)}")

         
        with coltb:

            
                
            
            with st.form(key="tkeysform"):
                # Add tag input for keywords
                keywords = st.text_input(label="Enter Keyword(s)", help="Enter one or more keywords separated by commas")

                # Add tag input for start date and end date
                start_date = st.date_input("Start Date")
                end_date = st.date_input("End Date")

                # Add search button within the form
                search_button = st.form_submit_button(label="Search")

                if search_button and keywords:
                    keyword_list = [keyword.strip() for keyword in keywords.split(",")]
                    for keyword in keyword_list:
                        results = []

                        # Retrieve recent tweets within the specified date range
                        max_results = 50
                        query = f"{keyword} since:{start_date} until:{end_date}"
                        tweets = api.search_tweets(q=query, count=max_results, tweet_mode="extended")

                        # Process each tweet
                        for tweet in tweets:
                            # Extract tweet information
                            user = tweet.user
                            user_name = user.name
                            user_screen_name = user.screen_name
                            profile_image_url = user.profile_image_url_https
                            user_description = user.description
                            user_location = user.location
                            tweet_id = tweet.id_str
                            retweet_count = tweet.retweet_count
                            mention_count = len(tweet.entities['user_mentions'])
                            quote_count = 0  # Quote count is not available in v1.1 API
                            hashtags = [tag['text'] for tag in tweet.entities['hashtags']]
                            like_count = tweet.favorite_count
                            # reply_count = tweet.reply_count
                            reply_count = len(api.search_tweets(f"to:{user_screen_name} since_id:{tweet_id}", count=0))

                            mentioned_users = []
                            for mention in tweet.entities['user_mentions']:
                                mentioned_user_screen_name = mention['screen_name']
                                mentioned_users.append({
                                    'User Screen Name': mentioned_user_screen_name
                                })

                            retweeted_tweet = None
                            if 'retweeted_status' in tweet._json:
                                retweeted_tweet = tweet.retweeted_status
                                retweeted_user = retweeted_tweet.user
                                retweeted_user_name = retweeted_user.name
                                retweeted_user_screen_name = retweeted_user.screen_name
                                retweeted_profile_image_url = retweeted_user.profile_image_url_https
                                retweeted_tweet_id = retweeted_tweet.id_str
                                retweeted_text = retweeted_tweet.full_text

                                retweeted_tweet_data = {
                                    'User Name': retweeted_user_name,
                                    'User Screen Name': retweeted_user_screen_name,
                                    'Profile Image URL': retweeted_profile_image_url,
                                    'Tweet ID': retweeted_tweet_id,
                                    'Text': retweeted_text,
                                    'Tweet URL': f"https://twitter.com/{retweeted_user_screen_name}/status/{retweeted_tweet_id}"
                                }

                            if tweet.in_reply_to_status_id:
                                in_reply_to_status = api.get_status(tweet.in_reply_to_status_id, tweet_mode="extended")
                                in_reply_to_name = in_reply_to_status.user.name
                            else:
                                in_reply_to_name = None

                            if tweet.coordinates is not None:
                                longitude = tweet.coordinates['coordinates'][0]
                                latitude = tweet.coordinates['coordinates'][1]
                            else:
                                longitude = None
                                latitude = None

                            if tweet.place is not None:
                                location_name = tweet.place.name
                                location_type = tweet.place.place_type
                                location_full_address = tweet.place.full_name
                            else:
                                location_name = None
                                location_type = None
                                location_full_address = None

                            # Extract media entities
                            media_entities = tweet.entities.get('media', [])
                            media_urls = [media['media_url_https'] for media in media_entities]

                            # Extract relevant fields from tweet and user
                            tweet_data = {
                                'User Name': user_name,
                                'User Screen Name': user_screen_name,
                                'Profile Image URL': profile_image_url,
                                'User Description': user_description,
                                'User Location': user_location,
                                'Created At': tweet.created_at,
                                'Tweet ID': tweet_id,
                                'Text': tweet.full_text,
                                'Hashtags': hashtags,
                                'Tweet URL': f"https://twitter.com/{user_screen_name}/status/{tweet_id}",
                                'Source': tweet.source or '',
                                'Retweet Count': retweet_count,
                                'Mention Count': mention_count,
                                'Quote Count': quote_count,
                                'Reply Count': reply_count,
                                'Like Count': like_count,
                                'Mentioned Users': mentioned_users,
                                'Retweeted Tweet': retweeted_tweet_data if retweeted_tweet else None,
                                'In Reply To': in_reply_to_name,
                                'Longitude': longitude,
                                'Latitude': latitude,
                                'Location Name': location_name,
                                'Location Type': location_type,
                                'Location Full Address': location_full_address,
                                'Media URLs': media_urls,
                                'Replies': []
                            }

                            # Search for replies to the tweet
                            replies_query = f"to:{user_screen_name} since_id:{tweet_id}"
                            replies = api.search_tweets(q=replies_query, tweet_mode="extended", count=max_results)

                            for reply in replies:
                                # Extract reply information
                                reply_user = reply.user
                                reply_user_name = reply_user.name
                                reply_user_screen_name = reply_user.screen_name
                                reply_id = reply.id_str
                                reply_text = reply.full_text
                                reply_created_at = reply.created_at
                                reply_retweet_count = reply.retweet_count
                                reply_like_count = reply.favorite_count

                                # Extract relevant fields from reply
                                reply_data = {
                                    'User Name': reply_user_name,
                                    'User Screen Name': reply_user_screen_name,
                                    'Tweet ID': reply_id,
                                    'Text': reply_text,
                                    'Created At': reply_created_at,
                                    'Retweet Count': reply_retweet_count,
                                    'Like Count': reply_like_count,
                                    'Tweet URL': f"https://twitter.com/{reply_user_screen_name}/status/{reply_id}"
                                }

                                tweet_data['Replies'].append(reply_data)
                        

                            results.append(tweet_data)

                        # Create a directory if it doesn't exist
                        os.makedirs("twitkeys", exist_ok=True)

                        # Save the results in a JSON file named with the keyword
                        file_path = os.path.join("twitkeys", f"{keyword}.json")
                        try:
                            if os.path.exists(file_path):
                                # Load existing data from the file
                                with open(file_path, 'r') as json_file:
                                    existing_data = json.load(json_file)

                                # Append new data to the existing data
                                existing_data['data'].extend(results)

                                # Write the combined data back to the file
                                with open(file_path, 'w') as json_file:
                                    json.dump(existing_data, json_file, cls=DateTimeEncoder)
                            else:
                                # Create a new file and write the data to it
                                output = {"data": results}
                                with open(file_path, 'w') as json_file:
                                    json.dump(output, json_file, cls=DateTimeEncoder)

                        except Exception as e:
                            st.error(f"Error saving results to JSON file: {e}")


        colc, cold=st.columns([2,2])
        with colc:
            container3a=st.container()
            with container3a:

                json_directory = "twittl"

                    # Get the list of JSON files in the directory
                json_files = [file for file in os.listdir(json_directory) if file.endswith(".json")]

                # Loop through each JSON file
                for json_file in json_files:
                    # Open the JSON file and load the data
                    with open(os.path.join(json_directory, json_file), 'r') as f:
                        user_data = json.load(f)

                    # Extract the tweets from the user data
                    tweets_list = user_data['tweets']
                    # user_name = tweet['user']['name']

                    # Create a placeholder for the tweets
                    tweet_placeholder = st.empty()
                    with tweet_placeholder:
                        # Display the tweets
                        # st.subheader(f"Tweets from {json_file[:-10]}")  # Remove the '_data.json' part from the filename
                        for tweet in tweets_list:
                            user_name = tweet['user']['name']
                            full_text = tweet['full_text']
                            
                            # print (user_name)
                            screen_name=tweet['user']['screen_name']
                            # st.write(f"{screen_name}")
                            tweet_content = f"{user_name}: {full_text}"
                            st.write(tweet_content)
                                # st.write("---")
                            time.sleep(0.1)  # Add a delay between each tweet (adjust as needed)
                

        with cold:
            container3b=st.container()
            with container3b:

                def display_tweets(file_path):
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        tweets = data['data']

                    for tweet in tweets:
                        user_name=tweet['User Name']
                        created_at=tweet['Created At']
                        text=tweet['Text']
                        
                        tweet_content = f"{user_name}:{created_at}{text}"
                        st.write(tweet_content)
                        # st.write("---")
                        time.sleep(0.2)  # Add a delay between each tweet (adjust as needed)

                directory = "twitkeys"
                json_files = [file for file in os.listdir(directory) if file.endswith('.json')]

                for json_file in json_files:
                    file_path = os.path.join(directory, json_file)
                    # st.header(json_file[:-5])
                    tweet_placeholder = st.empty()
                    with tweet_placeholder:
                        display_tweets(file_path)


            