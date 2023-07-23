import streamlit as st
import json
import httpx
import os
import pandas as pd
import jmespath
import requests
from PIL import Image
import base64
from io import BytesIO

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
    st.image("img/instagramicon.png", width=100)
with b:
    st.title("Instagram Analysis")



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

############################

    import os
    import json
    import requests
    from PIL import Image
    from io import BytesIO
    import streamlit as st

    folder_path = "insperson"
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
                    account_data = json.load(f)

                # Extract relevant information from the channel details
                profile_image_url = account_data['profile_image']
                name = account_data['name']
                category = account_data['category']
                bio = account_data['bio']
                bio_links = account_data['bio_links']
                homepage = account_data['homepage']
                followers = int(account_data['followers'])
                follows = int(account_data['follows'])
                video_count = int(account_data['video_count'])
                image_count = int(account_data['image_count'])

                try:
                    response = requests.get(profile_image_url)
                    profile_image = Image.open(BytesIO(response.content))

                    col.image(profile_image, width=200)
                except:
                    col.write("Error loading profile image")

                col.subheader(name)
                col.write(category)
                col.write(bio)
                # col.write(str(homepage))
                col.write(f"{followers} followers")
                col.write(f"{follows} follows")
                col.write(f"{image_count} images")
                col.write(f"{video_count} videos")
        
########################TIME SERIES ANALYSIS OF KEY PERSOM ##################

        import matplotlib.pyplot as plt
        import seaborn as sns
        import streamlit as st
        import pandas as pd
        import os
        import json
        from datetime import datetime
     
        import plotly.graph_objects as go
     
       

        st.header('Time Series Analysis of The Key Persons')

       

        # Function to load data from JSON files
        def load_data(folder_path, selected_files):
            data = {}
            for filename in os.listdir(folder_path):
                if filename.endswith(".json") and filename.split(".json")[0] in selected_files:
                    with open(os.path.join(folder_path, filename)) as file:
                        profile_name = filename.split(".json")[0]
                        data[profile_name] = json.load(file)
            return data

        def filter_data_by_date(data, start_date, end_date):
            filtered_data = {}
            for profile, posts in data.items():
                filtered_posts = [
                    post for post in posts if start_date <= datetime.fromisoformat(post['Posted on']).date() <= end_date
                ]
                if filtered_posts:
                    filtered_data[profile] = filtered_posts
            return filtered_data

        # Streamlit app
        st.title("Instagram Posts Time Series")

        # Input folder containing JSON files
        folder_path = "inspersonpost"
        # files_with_extension = [filename.split(".json")[0] for filename in os.listdir(folder_path) if filename.endswith(".json")]
        files = [file.replace(".json", "") for file in os.listdir(folder_path) if file.endswith(".json")]
        selected_files = st.multiselect("Select account:", files)



        # Load data from selected JSON files
        data = load_data(folder_path, selected_files)

        # Input date range
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")

        # Filter data by date range
        filtered_data = filter_data_by_date(data, start_date, end_date)

        # Convert data to time series format
        time_series_data = {}
        for profile, posts in filtered_data.items():
            time_series_data[profile] = [post['Posted on'] for post in posts]

        # Create the time series chart
        fig = go.Figure()
        for profile, dates in time_series_data.items():
            fig.add_trace(go.Scatter(x=dates, y=[i for i in range(1, len(dates)+1)], mode='lines+markers', name=profile))

        fig.update_layout(title=f"Instagram Posts Time Series",
                        xaxis_title="Date",
                        yaxis_title="Number of Posts",
                        xaxis_tickformat="%Y-%m-%d")

        # Display the time series chart
        st.plotly_chart(fig)

#################################SNA  ###################################################

        
        import networkx as nx
        from pyvis.network import Network
        import datetime
        from pandas.api.types import is_datetime64_any_dtype as is_datetime


        def process_json_file(file_path, file_name):
            with open(file_path, "r") as file:
                data = json.load(file)

            # Process the JSON data into a DataFrame (extract user, comments, answers, and Posted on data)
            user_data = []
            comments_data = []
            answers_data = []
            posted_on_data = []
            for post in data:
                user_data.append(post["user"])

                # Extract the "Posted on" information or use "N/A" if not present
                posted_on = post.get("Posted on", "N/A")
                # If "Posted on" is not in the expected format, use "N/A"
                try:
                    posted_on = pd.to_datetime(posted_on)
                except:
                    posted_on = "N/A"
                posted_on_data.append(posted_on)

                for comment in post.get("Comments", []):
                    # Extract the "Owner" information or use "N/A" if not present
                    owner = comment.get("Owner", "N/A")
                    comments_data.append((post["user"], owner))

                    # Extract answers and answer owner information (if present)
                    for answer in comment.get("Answers", []):
                        answer_owner = answer.get("Owner", "N/A")
                        answers_data.append((comment["Owner"], answer_owner))

            user_df = pd.DataFrame({"User": user_data, "Posted on": posted_on_data})
            comments_df = pd.DataFrame(comments_data, columns=["User", "CommentOwner"])
            answers_df = pd.DataFrame(answers_data, columns=["CommentOwner", "AnswerOwner"])

            # Add a new column with the file name to user_df
            user_df["File"] = file_name

            return user_df, comments_df, answers_df

        def create_social_network_graph(user_df, comments_df, answers_df):
            G = nx.DiGraph()  # Use DiGraph to create directed edges for comments and answers

            # Add nodes for users, comment owners, and answer owners
            for user in user_df["User"]:
                G.add_node(user, color="orange")  # Set the color of user nodes to blue

            for comment_owner in comments_df["CommentOwner"]:
                G.add_node(comment_owner, color="#1E90FF")  # Set the color of comment owner nodes to green

            for answer_owner in answers_df["AnswerOwner"]:
                G.add_node(answer_owner, color="#BAEAFE")  # Set the color of answer owner nodes to red

            # Calculate the count of comments for each post and store it in a dictionary
            comment_counts = comments_df["User"].value_counts().to_dict()

            # Create edges for comments
            for _, row in comments_df.iterrows():
                comment_owner = row["CommentOwner"]
                post_user = row["User"]
                count = comment_counts.get(post_user, 0)  # Get the count of comments for this post

                # Set the edge width based on the count of comments (make it bigger for more comments)
                edge_width = count + 1  # Add 1 to avoid zero-width edges
                G.add_edge(comment_owner, post_user, label="comment", value=edge_width, color="#1E90FF", arrows="to")

            # Create edges for answers
            for _, row in answers_df.iterrows():
                comment_owner = row["CommentOwner"]
                answer_owner = row["AnswerOwner"]

                G.add_edge(comment_owner, answer_owner, label="answer", value=1, color="#BAEAFE", arrows="to")

            # Calculate centrality degree for each node
            centrality_degree = nx.degree_centrality(G)

            # Set node size based on centrality degree (make it bigger for higher centrality)
            min_node_size = 10  # Minimum node size
            max_node_size = 50  # Maximum node size
            for node, centrality in centrality_degree.items():
                node_size = min_node_size + (max_node_size - min_node_size) * centrality
                G.nodes[node]["size"] = node_size

            return G
        
        def filter_data_by_date(data_df, start_date, end_date):
            # Check if "Posted on" column exists in the DataFrame
            if "Posted on" not in data_df:
                return data_df

            # Convert the "Posted on" column to datetime objects (if not already)
            if not is_datetime(data_df["Posted on"]):
                data_df["Posted on"] = pd.to_datetime(data_df["Posted on"])

            # Convert start_date and end_date to pandas datetime objects with UTC timezone
            start_date = pd.to_datetime(start_date).tz_localize('UTC')
            end_date = pd.to_datetime(end_date).tz_localize('UTC')

            # Filter the DataFrame based on the date range
            filtered_df = data_df[(data_df["Posted on"] >= start_date) & (data_df["Posted on"] <= end_date)]
            return filtered_df


        st.title("Social Network Analytics of Instagram Account")

        # Get a list of files from the "inspersonpost" folder
        folder_path = "inspersonpost"
        files = [file.replace(".json", "") for file in os.listdir(folder_path) if file.endswith(".json")]

        # Find the newest file based on modification time
        newest_file = max(files, key=lambda file: os.path.getmtime(os.path.join(folder_path, f"{file}.json")))

        col1, col2, col3 = st.columns([2, 1, 1])

        # Multi-select files
        with col1:
            selected_files = st.multiselect("Select files", files, default=[newest_file])  # Set default selection to the newest file

        # Filter by start and end date
        with col2:
            start_date = st.date_input("Start Date", key='stdinstkp')

        with col3: 
            end_date = st.date_input("End Date", key='edsnakpinst')

        # Process selected files and filter data by date
        user_data_list = []
        comments_data_list = []

        # Initialize answers_data, all_user_data, and all_comments_data as None
        answers_data = None
        all_user_data = None
        all_comments_data = None

        # Check if any files are selected
        if len(selected_files) > 0:
            G = nx.DiGraph()  # Initialize the graph outside the loop

            for file in selected_files:
                file_path = os.path.join(folder_path, f"{file}.json")  # Add .json extension to the selected file
                print("Processing file:", file_path)  # Print the file path for debugging

                # Pass both file_path and file name to process_json_file
                user_data, comments_data, new_answers_data = process_json_file(file_path, file)

                # Add nodes for users, comment owners, and answer owners with respective colors for each file
                for user in user_data["User"]:
                    G.add_node(user, color="orange")  # Set the color of user nodes to blue

                for comment_owner in comments_data["CommentOwner"]:
                    G.add_node(comment_owner, color="#1E90FF")  # Set the color of comment owner nodes to green

                for answer_owner in new_answers_data["AnswerOwner"]:
                    G.add_node(answer_owner, color="#BAEAFE")  # Set the color of answer owner nodes to red

                user_data_list.append(user_data)
                comments_data_list.append(comments_data)

                # If answers_data is None (first iteration), set it to the new_answers_data
                if answers_data is None:
                    answers_data = new_answers_data
                else:
                    # Otherwise, concatenate the new_answers_data to the existing answers_data
                    answers_data = pd.concat([answers_data, new_answers_data], ignore_index=True)

            # Concatenate data from all files
            all_user_data = pd.concat(user_data_list, ignore_index=True)
            all_comments_data = pd.concat(comments_data_list, ignore_index=True)

            # Filter data by date
            if start_date and end_date:
                all_user_data = filter_data_by_date(all_user_data, start_date, end_date)
                all_comments_data = filter_data_by_date(all_comments_data, start_date, end_date)



     # Check if any files are selected and data is available
            if len(selected_files) > 0 and all_user_data is not None and all_comments_data is not None:
                # Create social network graph
                # Create social network graph
                graph = create_social_network_graph(all_user_data, all_comments_data, answers_data)

                # Visualize the graph using Pyvis and Streamlit
                nt = Network(notebook=False)

                # Add nodes and edges from the NetworkX graph to the Pyvis Network object
                for node, data in graph.nodes(data=True):
                    # Get the color attribute if available, or use a default color if it's missing
                    node_color = data.get("color", "orange")
                    nt.add_node(node, color=node_color, size=data.get("size", 10))  # Use "size" attribute if available

                for edge in graph.edges():
                    nt.add_edge(edge[0], edge[1], value=graph.edges[edge].get("value", 1), color=graph.edges[edge].get("color", "#1E90FF"),
                                title=graph.edges[edge].get("label", "comment"), arrows=graph.edges[edge].get("arrows", "to"))

                # Set the smoothness of the edges
                nt.set_edge_smooth("dynamic")

                # Save the graph to an HTML file
                nt.save_graph('html_files/instaaccount_social_network.html')

            else:
                if len(selected_files) == 0:
                    st.warning("Please select at least one file.")
                else:
                    st.warning("No data available for the selected date range.")

            
        

#####################################################################################

            import matplotlib.pyplot as plt
            import pandas as pd
            # Create social network graph
            graph = create_social_network_graph(all_user_data, all_comments_data, answers_data)

            # Calculate the degree centrality for each node
            degree_centrality = nx.degree_centrality(graph)

            # Sort the nodes based on degree centrality in descending order
            sorted_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)

            # Select the top ten nodes with highest degree centrality
            top_ten_nodes = sorted_nodes[:10]

            # Create a subgraph with the top ten nodes and their neighboring edges
            subgraph = graph.subgraph(top_ten_nodes)

            # Visualize the subgraph using Pyvis and Streamlit
            nt_subgraph = Network(notebook=False)
            nt_subgraph.from_nx(subgraph)
            nt_subgraph.save_graph("html_files/instaaccount_degree_centrality_graph.html")

            # Calculate the degree centrality scores and ranks for the top ten actors
            top_ten_degree_scores = [degree_centrality[node] for node in top_ten_nodes]
            top_ten_degree_ranks = list(range(1, len(top_ten_nodes) + 1))

            # Create a DataFrame to hold the degree centrality scores and ranks
            top_ten_degree_df = pd.DataFrame({
                "Actor": top_ten_nodes,
                "Degree Centrality Score": top_ten_degree_scores,
                "Rank": top_ten_degree_ranks
            })

            # Generate a dynamic message based on the network data
            message = f"Below is a visual representation of the social network graph with the top ten actors based on degree centrality.\n"
            message += f"The top ten actors are: {', '.join(top_ten_nodes)}.\n"
            message += f"They have the following degree centrality scores: {', '.join(map(str, top_ten_degree_scores))}.\n"
            message += f"Degree centrality is a measure of the number of connections a node has in the network.\n"
            message += f"Higher degree centrality indicates that the actor is more central to the network."

    
    ######################################BETWEENESS CENTRALITY #############################################
            graph = create_social_network_graph(all_user_data, all_comments_data, answers_data)

            # Calculate the betweenness centrality for each node
            betweenness_centrality = nx.betweenness_centrality(graph)

            # Sort the nodes based on betweenness centrality in descending order
            sorted_nodes_by_betweenness = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)

            # Select the top ten nodes with highest betweenness centrality
            top_ten_nodes_by_betweenness = sorted_nodes_by_betweenness[:10]

            # Create a subgraph with the top ten nodes and their neighboring edges
            subgraph_by_betweenness = graph.subgraph(top_ten_nodes_by_betweenness)

            # Visualize the subgraph with betweenness centrality using Pyvis and Streamlit
            nt_subgraph_by_betweenness = Network(notebook=False)
            nt_subgraph_by_betweenness.from_nx(subgraph_by_betweenness)
            nt_subgraph_by_betweenness.save_graph("html_files/instaaccount_betweenness_centrality_graph.html")

            # Calculate the betweenness centrality scores and ranks for the top ten nodes
            top_ten_betweenness_scores = [betweenness_centrality[node] for node in top_ten_nodes_by_betweenness]
            top_ten_betweenness_ranks = list(range(1, len(top_ten_nodes_by_betweenness) + 1))

            # Create a DataFrame to hold the betweenness centrality scores and ranks
            top_ten_betweenness_df = pd.DataFrame({
                "Node": top_ten_nodes_by_betweenness,
                "Betweenness Centrality Score": top_ten_betweenness_scores,
                "Rank": top_ten_betweenness_ranks
            })

            # Generate a dynamic message based on the network data for betweenness centrality
            message_betweenness = f"Below is a visual representation of the subgraph with the top ten nodes based on betweenness centrality.\n"
            message_betweenness += f"The top ten nodes are: {', '.join(top_ten_nodes_by_betweenness)}.\n"
            message_betweenness += f"They have the following betweenness centrality scores: {', '.join(map(str, top_ten_betweenness_scores))}.\n"
            message_betweenness += f"Betweenness centrality is a measure of how often a node acts as a bridge along the shortest path between two other nodes.\n"
            message_betweenness += f"Higher betweenness centrality indicates that the node plays a significant role in connecting other nodes in the network."

        
            
    #################################################################################################
            
            import networkx as nx
            from pyvis.network import Network
            import datetime
            import pandas as pd

            graph = create_social_network_graph(all_user_data, all_comments_data, answers_data)
            # Calculate the closeness centrality for each node
            closeness_centrality = nx.closeness_centrality(graph)

            # Sort the nodes based on closeness centrality in descending order
            sorted_nodes_by_closeness = sorted(closeness_centrality, key=closeness_centrality.get, reverse=True)

            # Select the top ten nodes with highest closeness centrality
            top_ten_nodes_by_closeness = sorted_nodes_by_closeness[:10]

            # Create a subgraph with the top ten nodes and their neighboring edges
            subgraph_by_closeness = graph.subgraph(top_ten_nodes_by_closeness)

            # Visualize the subgraph with closeness centrality using Pyvis and Streamlit
            nt_subgraph_by_closeness = Network(notebook=False)
            nt_subgraph_by_closeness.from_nx(subgraph_by_closeness)
            nt_subgraph_by_closeness.save_graph("html_files/instaaccount_closeness_centrality_graph.html")

            # Calculate the closeness centrality scores and ranks for the top ten nodes
            top_ten_closeness_scores = [closeness_centrality[node] for node in top_ten_nodes_by_closeness]
            top_ten_closeness_ranks = list(range(1, len(top_ten_nodes_by_closeness) + 1))

            # Create a DataFrame to hold the closeness centrality scores and ranks
            top_ten_closeness_df = pd.DataFrame({
                "Node": top_ten_nodes_by_closeness,
                "Closeness Centrality Score": top_ten_closeness_scores,
                "Rank": top_ten_closeness_ranks
            })

            # Generate a dynamic message based on the network data for closeness centrality
            message_closeness = f"Below is a visual representation of the subgraph with the top ten nodes based on closeness centrality.\n"
            message_closeness += f"The top ten nodes are: {', '.join(top_ten_nodes_by_closeness)}.\n"
            message_closeness += f"They have the following closeness centrality scores: {', '.join(map(str, top_ten_closeness_scores))}.\n"
            message_closeness += f"Closeness centrality is a measure of how quickly a node can reach other nodes in the network.\n"
            message_closeness += f"Higher closeness centrality indicates that the node is more central and well-connected in the network."


    #########################################################################################
            colinstkpviz1, colinstkpviz2, colinstkpviz3, colinstkpviz4=st.tabs(['Social Network','Main Actors', 'Bridging Actors', 'Supporting Actors'])

            with colinstkpviz1:
            # Display the graph in Streamlit
                with open('html_files/instaaccount_social_network.html', 'r') as f:
                    html_string = f.read()
                st.components.v1.html(html_string, height=700, scrolling=True)
                
            with colinstkpviz2:
                # Display the graph in Streamlit
                with open("html_files/instaaccount_degree_centrality_graph.html", "r") as f:
                    html_string = f.read()
                st.components.v1.html(html_string, height=610, scrolling=True)

                # Display the dynamic message using st.markdown to render formatted markdown
                st.subheader("Social Network Analysis Details")
                st.markdown(message)

                # Display the DataFrame and chart side by side using st.columns
                col1, col2 = st.columns([1, 1])  # Adjust the column widths as needed

                # Display the DataFrame in the first column
                col1.subheader("Top Ten Actors by Degree Centrality Score")
                col1.dataframe(top_ten_degree_df.set_index("Actor"))

                # Display the chart in the second column
                col2.subheader("Top Ten Actors Chart")
                col2.bar_chart(top_ten_degree_df.set_index("Actor")["Degree Centrality Score"])

            with colinstkpviz3:
                # Display the graph with betweenness centrality in Streamlit
                with open("html_files/instaaccount_betweenness_centrality_graph.html", "r") as f:
                    html_string_betweenness = f.read()
                st.components.v1.html(html_string_betweenness, height=610, scrolling=True)

                # Display the dynamic message for betweenness centrality
                st.caption(message_betweenness)

                # Display the DataFrame and chart side by side using st.columns
                col1, col2 = st.columns([1, 1])  # Adjust the column widths as needed

                # Display the DataFrame in the first column for betweenness centrality
                col1.subheader("Top Ten Nodes by Betweenness Centrality Score")
                col1.dataframe(top_ten_betweenness_df.set_index("Node"))

                # Display the chart in the second column for betweenness centrality
                col2.subheader("Top Ten Nodes Chart for Betweenness Centrality")
                col2.bar_chart(top_ten_betweenness_df.set_index("Node")["Betweenness Centrality Score"])

            with colinstkpviz4:

                    # Display the graph with closeness centrality in Streamlit
                with open("html_files/instaaccount_closeness_centrality_graph.html", "r") as f:
                    html_string_closeness = f.read()
                st.components.v1.html(html_string_closeness, height=610, scrolling=True)

                # Display the dynamic message for closeness centrality
                st.caption(message_closeness)

                # Display the DataFrame and chart side by side using st.columns
                col1, col2 = st.columns(2)  # Adjust the column widths as needed

                # Display the DataFrame in the first column for closeness centrality
                col1.subheader("Top Ten Nodes by Closeness Centrality Score")
                col1.dataframe(top_ten_closeness_df.set_index("Node"))

                # Display the chart in the second column for closeness centrality
                col2.subheader("Top Ten Nodes Chart for Closeness Centrality")
                col2.bar_chart(top_ten_closeness_df.set_index("Node")["Closeness Centrality Score"])




######################################TOPIC MODELING #####################################

        import streamlit as st
        import os
        import json
        import pandas as pd
        import gensim
        from gensim import corpora
        import pyLDAvis.gensim_models as gensimvis
        import pyLDAvis
        from wordcloud import WordCloud

        # Function to load data from JSON files in the "inspersonpost" folder
        def load_data(folder_path):
            data = []
            for filename in os.listdir(folder_path):
                if filename.endswith(".json"):
                    with open(os.path.join(folder_path, filename), "r") as file:
                        posts = json.load(file)
                        data.extend(posts)
            return data

        # Function to perform topic modeling
        def perform_topic_modeling(data):
            # Extract caption text from the data
            captions = [post["Caption"] for post in data]

            # Tokenize the captions and remove stopwords
            stop_words = set()
            # Add your custom stopwords here if needed
            stop_words.update(["dan", "di", "ke", "dari", "untuk", "yang", "juga", "tetapi", "akan", "walau",
              "walaupun", "meski", "meskipun", "adapun", "jika", "maka", "karena", "sebab",
                 "oleh", "dengan", "dalam", "ini", "itu", "pada", "atau", "ada", "hal", "baca", 
                 "saat", "ketika", "tersebut", "apabila", "kita", "saya", "kami", "kamu", "aku",
                    "bisa", "dapat", "bagaimana", "menjadi", "sebagai", "tidak", "iya", "antara",
                       "atas", "bawah", "ini"])

            tokenized_captions = [
                [word for word in caption.lower().split() if word not in stop_words]
                for caption in captions
            ]

            # Create a Gensim dictionary and corpus
            dictionary = corpora.Dictionary(tokenized_captions)
            corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_captions]

            # Perform LDA topic modeling
            num_topics = 5  # You can change the number of topics as needed
            lda_model = gensim.models.LdaModel(
                corpus, num_topics=num_topics, id2word=dictionary, passes=20
            )

            # Visualize the topics using PyLDAvis
            vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
            pyLDAvis.save_html(vis_data, "topic_modeling_result.html")

            return lda_model, vis_data
        
        def display_wordcloud(lda_model, topic_num):
            words = dict(lda_model.show_topic(topic_num, topn=30))
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(words)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        # Streamlit app
        st.title("Instagram Caption Topic Modeling")

        # Load data from the "inspersonpost" folder
        folder_path = "inspersonpost"
        data = load_data(folder_path)

        # Create a list of filenames for the select box
        file_names = [os.path.splitext(filename)[0] for filename in os.listdir(folder_path) if filename.endswith(".json")]

        # User input: Select file and date range
        inskptmcol1, inskptmcol2, inskptmcol3=st.columns([2,1,1])
        with inskptmcol1:
            selected_file = st.selectbox("Select a user", file_names, index=0, key='selp3rin4t')
        with inskptmcol2:
            start_date = st.date_input("Start Date", key='st4perins5kp')
        with inskptmcol3:
            end_date = st.date_input("End Date", key='3dtpe(kpin5t)')

        # Filter data based on selected date range
        filtered_data = [
            post for post in data
            if start_date <= pd.to_datetime(post["Posted on"]).date() <= end_date
        ]

        st.set_option('deprecation.showPyplotGlobalUse', False)

        if len(filtered_data) == 0:
            st.warning("No posts found within the specified date range.")
        else:
            # Perform topic modeling
            lda_model, vis_data = perform_topic_modeling(filtered_data)

            # Display the PyLDAvis visualization
            with open("topic_modeling_result.html", "r") as file:
                html = file.read()
            st.components.v1.html(html, width=1500, height=800)

            # Display selectable word cloud for each topic
            st.subheader("Word Cloud for Each Topic")
            topic_list = [f"Topic {topic_num + 1}" for topic_num in range(lda_model.num_topics)]
            selected_topic = st.selectbox("Select a topic", topic_list, index=0)

            if selected_topic:
                topic_num = int(selected_topic.split()[-1]) - 1
                display_wordcloud(lda_model, topic_num)


##################################SENTIMENT ANALYSIS ########################################
   
        from pathlib import Path
        from textblob import TextBlob
        import streamlit as st
        import json
        import os
        import pandas as pd
        import plotly.graph_objects as go

        # Function to read JSON files and extract data
        def read_json_file(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
            return data

        # Function to perform sentiment analysis on Bahasa Indonesia text
        def analyze_sentiment(text):
            blob = TextBlob(text)
            return blob.sentiment.polarity

        # Function to process the data and create pie chart
        def process_data(data):
            df = pd.DataFrame(data)
            df["Caption Sentiment"] = df["Caption"].apply(analyze_sentiment)
            df["Comments Sentiment"] = df["Comments"].apply(lambda comments: sum(analyze_sentiment(comment["Text"]) for comment in comments) / len(comments) if comments else 0)
            return df

        # Function to plot the pie chart
        def plot_pie_chart(values, labels, colors, title):
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))])
            fig.update_layout(title_text=title)
            return fig

        def display_pie_charts(files_list, start_date, end_date):
            num_files = len(files_list)
            num_columns = num_files * 2  # Set the number of columns based on the number of selected JSON files (2 columns for each file)
            columns = st.columns(num_columns)

            # Convert start_date and end_date to pandas Timestamp objects with UTC timezone
            start_date = pd.to_datetime(start_date).tz_localize('UTC')
            end_date = pd.to_datetime(end_date).tz_localize('UTC')

            for i, file_name in enumerate(files_list):
                file_path = f"inspersonpost/{file_name}.json"
                data = read_json_file(file_path)
                df = process_data(data)

                # Convert the "Posted on" column to pandas Timestamp objects with UTC timezone
                df["Posted on"] = pd.to_datetime(df["Posted on"])

                # Filter data based on date range
                mask = (df["Posted on"] >= start_date) & (df["Posted on"] <= end_date)
                df = df[mask]


                # Caption Sentiment
                positive_caption = df[df["Caption Sentiment"] > 0].shape[0]
                neutral_caption = df[df["Caption Sentiment"] == 0].shape[0]
                negative_caption = df[df["Caption Sentiment"] < 0].shape[0]
                total_caption = df.shape[0]
                caption_trace = plot_pie_chart([positive_caption, neutral_caption, negative_caption], ['Positive', 'Neutral', 'Negative'],
                                            ['#66CC99', '#FFCC99', '#FF9999'], f"{file_name} - Caption Sentiment")

                # Comments Sentiment
                positive_comments = df[df["Comments Sentiment"] > 0].shape[0]
                neutral_comments = df[df["Comments Sentiment"] == 0].shape[0]
                negative_comments = df[df["Comments Sentiment"] < 0].shape[0]
                total_comments = df.shape[0]
                comments_trace = plot_pie_chart([positive_comments, neutral_comments, negative_comments], ['Positive', 'Neutral', 'Negative'],
                                                ['#66CC99', '#FFCC99', '#FF9999'], f"{file_name} - Comments Sentiment")

                # Display the pie charts in the corresponding columns
                with columns[i * 2]:
                    st.plotly_chart(caption_trace)
                with columns[i * 2 + 1]:
                    st.plotly_chart(comments_trace)

        # Main Streamlit app
        st.title("Sentiment Analysis of Posts and Comments")
        files_list = [Path(file_name).stem for file_name in os.listdir("inspersonpost") if file_name.endswith(".json")]
        selected_files = st.multiselect("Select account", files_list)

        if selected_files:
            start_date = st.date_input("Select Start Date", value=pd.Timestamp('2023-06-01'))
            end_date = st.date_input("Select End Date", value=pd.Timestamp('2023-06-30'))

            filtered_files = [file_name for file_name in selected_files]
            display_pie_charts(filtered_files, start_date, end_date)


#######################SENTIMENT ANALYSIS PER USER ###############
        import json
        import pandas as pd
        import streamlit as st
        from pathlib import Path
        from nltk.sentiment import SentimentIntensityAnalyzer

        # Download the Bahasa Indonesian lexicon (only needed once)
        import nltk
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('vader_lexicon')

        # Function to read JSON files and extract data
        def read_json_file(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
            return data

        # Function to perform sentiment analysis using Bahasa Indonesian lexicon
        def analyze_sentiment(text):
            text = str(text)  # Convert to a string to handle possible encoding issues
            sid = SentimentIntensityAnalyzer()
            sentiment_scores = sid.polarity_scores(text)
            return sentiment_scores

        # Streamlit app
        st.title("Sentiment Analysis per Comment Owner")
        # st.write("Select JSON files from 'inspersonpost' folder and choose the date range.")

        files_list = [Path(file_name).stem for file_name in os.listdir("inspersonpost") if file_name.endswith(".json")]

        # Find the newest file based on modification time
        newest_file = max(files_list, key=lambda file: os.path.getmtime(f"inspersonpost/{file}.json"))

        # Default date range (e.g., last 7 days)
        default_end_date = pd.to_datetime("today").date()
        default_start_date = default_end_date - pd.Timedelta(days=7)

        # Multi-select files
        selected_files = st.multiselect("Select account", files_list, default=[newest_file], key='selfil5entu5erinst4kp')

        data = []
        for file_name in selected_files:
            file_path = f"inspersonpost/{file_name}.json"
            data.extend(read_json_file(file_path))

        df = pd.DataFrame(data)

        # Step 2: Filter data based on start and end dates
        start_date = st.date_input("Select start date", value=default_start_date)
        end_date = st.date_input("Select end date", value=default_end_date)

        filtered_data = df[(df["Posted on"] >= str(start_date)) & (df["Posted on"] <= str(end_date))]

        # Step 3: Group comments by comment owner and calculate sentiment percentages
        if not filtered_data.empty:
            comments_by_owner = filtered_data.explode("Comments").reset_index(drop=True)
            comments_by_owner = pd.concat([comments_by_owner.drop(['Comments'], axis=1), comments_by_owner['Comments'].apply(pd.Series)], axis=1)

            # Perform sentiment analysis on comments
            comments_by_owner["Sentiment Scores"] = comments_by_owner["Text"].apply(analyze_sentiment)
            comments_by_owner["Compound Score"] = comments_by_owner["Sentiment Scores"].apply(lambda score: score["compound"])
            comments_by_owner["Sentiment"] = comments_by_owner["Compound Score"].apply(lambda score: "positive" if score >= 0 else "negative")

            # Calculate sentiment percentages per comment owner
            sentiment_counts = comments_by_owner.groupby(["Owner", "Sentiment"]).size().unstack(fill_value=0)
            sentiment_counts["Total"] = sentiment_counts.sum(axis=1)

            # Handle the case when there are no comments with negative sentiment
            if "negative" in sentiment_counts.columns:
                sentiment_counts["Negative Percentage"] = sentiment_counts["negative"] / sentiment_counts["Total"]
            else:
                sentiment_counts["Negative Percentage"] = 0

            # Handle the case when there are no comments with positive sentiment
            if "positive" in sentiment_counts.columns:
                sentiment_counts["Positive Percentage"] = sentiment_counts["positive"] / sentiment_counts["Total"]
            else:
                sentiment_counts["Positive Percentage"] = 0

            sentiment_counts = sentiment_counts[["Positive Percentage", "Negative Percentage"]]

            st.write("Sentiment Analysis per comment owner Results:")
            st.bar_chart(sentiment_counts)

            # Step 4: Display the comments table
            st.write("Filtered Comments:")
            st.dataframe(comments_by_owner[["Owner", "Text", "Sentiment"]])
        else:
            st.write("No comments found within the selected date range.")




###############TAB 2###########################

with tab2:
    st.header("ISSUE")

##########################TIME SERIES ANALYSIS ####################
    import datetime
    from dateutil.parser import parse

    def load_data(folder_path, selected_files):
        data = {}
        for filename in os.listdir(folder_path):
            if filename.endswith(".json") and filename.split(".json")[0] in selected_files:
                with open(os.path.join(folder_path, filename)) as file:
                    keyword = filename.split(".json")[0]
                    data[keyword] = json.load(file)
        return data

    def filter_data_by_date(data, start_date, end_date):
        filtered_data = {}
        for keyword, posts in data.items():
            filtered_posts = [
                post for post in posts if start_date <= parse(post['Posted on']).date() <= end_date
            ]
            if filtered_posts:
                filtered_data[keyword] = filtered_posts
        return filtered_data

    # Streamlit app
    st.title("Instagram Posts Time Series")

    # Input folder containing JSON files
    folder_path = "instaposts"
    # files_with_extension = [filename.split(".json")[0] for filename in os.listdir(folder_path) if filename.endswith(".json")]
    files = [file.replace(".json", "") for file in os.listdir(folder_path) if file.endswith(".json")]
    selected_files = st.multiselect("Select keyword:", files)

    # Load data from selected JSON files
    data = load_data(folder_path, selected_files)

    # Input date range
    start_date = st.date_input("Start Date", key='instissuestd')
    end_date = st.date_input("End Date", key='dtissueinsta')

    # Filter data by date range
    filtered_data = filter_data_by_date(data, start_date, end_date)

    # Convert data to time series format
    time_series_data = {}
    for keyword, posts in filtered_data.items():
        time_series_data[keyword] = [post['Posted on'] for post in posts]

    # Create the time series chart
    fig = go.Figure()
    for profile, dates in time_series_data.items():
        fig.add_trace(go.Scatter(x=dates, y=[i for i in range(1, len(dates)+1)], mode='lines+markers', name=profile))

    # Add title to the chart using selected keywords
    selected_keywords_str = ', '.join(selected_files)
    chart_title = f"Instagram Posts Time Series for Keywords: {selected_keywords_str}"
    fig.update_layout(title=chart_title, xaxis_title="Date", yaxis_title="Number of Posts", xaxis_tickformat="%Y-%m-%d")

    # Display the time series chart
    st.plotly_chart(fig)

##########################SNA ISSUE##########################################
    import networkx as nx
    from pyvis.network import Network
    import json
    import pandas as pd
    import os
    from pandas.api.types import is_datetime64_any_dtype as is_datetime

    def process_json_file(file_path, file_name):
        with open(file_path, "r") as file:
            data = json.load(file)

        # Process the JSON data into a DataFrame (extract user, comments, answers, Tagged Users, and Posted on data)
        user_data = []
        comments_data = []
        answers_data = []
        tagged_users_data = []
        mentions_data = []  # For mention relationships
        posted_on_data = []

        for post in data:
            user_data.append(post["user"])

            # Extract the "Posted on" information or use "N/A" if not present
            posted_on = post.get("Posted on", "N/A")
            # If "Posted on" is not in the expected format, use "N/A"
            try:
                posted_on = pd.to_datetime(posted_on)
            except:
                posted_on = "N/A"
            posted_on_data.append(posted_on)

            # Extract Tagged Users information
            tagged_users = post.get("Tagged Users", [])
            for tagged_user in tagged_users:
                tagged_users_data.append((post["user"], tagged_user))

            for comment in post.get("Comments", []):
                # Extract the "Owner" information or use "N/A" if not present
                owner = comment.get("Owner", "N/A")
                comments_data.append((post["user"], owner))

                # Extract mentions and mention owner information (if present)
                for mentioned_user in get_mentions(comment.get("Text", "")):
                    mentions_data.append((comment["Owner"], mentioned_user))

                # Extract answers and answer owner information (if present)
                for answer in comment.get("Answers", []):
                    answer_owner = answer.get("Owner", "N/A")
                    answers_data.append((comment["Owner"], answer_owner))

        user_df = pd.DataFrame({"User": user_data, "Posted on": posted_on_data})
        comments_df = pd.DataFrame(comments_data, columns=["User", "CommentOwner"])
        answers_df = pd.DataFrame(answers_data, columns=["CommentOwner", "AnswerOwner"])
        tagged_users_df = pd.DataFrame(tagged_users_data, columns=["User", "TaggedUser"])
        mentions_df = pd.DataFrame(mentions_data, columns=["CommentOwner", "MentionedUser"])

        # Add a new column with the file name to user_df
        user_df["File"] = file_name

        return user_df, comments_df, answers_df, tagged_users_df, mentions_df

    def get_mentions(text):
        # This function extracts mentioned usernames from the comment text
        # For example, if the text is "Hello @user1 and @user2, how are you?", it will return ['user1', 'user2']
        return [word[1:] for word in text.split() if word.startswith("@")]


    def create_social_network_graph(user_df, comments_df, answers_df, tagged_users_df, mentions_df):
        G = nx.DiGraph()

        # Add nodes for users, comment owners, answer owners, tagged users, and mentioned users
        for user in user_df["User"]:
            G.add_node(user, color="orange", group="user")  # Set the color of user nodes to orange

        for comment_owner in comments_df["CommentOwner"]:
            G.add_node(comment_owner, color="#1E90FF", group="comment_owner")  # Set the color of comment owner nodes to blue

        for answer_owner in answers_df["AnswerOwner"]:
            G.add_node(answer_owner, color="#BAEAFE", group="answer_owner")  # Set the color of answer owner nodes to light blue

        for tagged_user in tagged_users_df["TaggedUser"]:
            G.add_node(tagged_user, color="#BFDB38", group="tagged_user")  # Set the color of tagged users nodes to yellow

        for mentioned_user in mentions_df["MentionedUser"]:
            G.add_node(mentioned_user, color="purple", group="mentioned_user")  # Set the color of mentioned users nodes to purple

        # Calculate the count of comments for each post and store it in a dictionary
        comment_counts = comments_df["User"].value_counts().to_dict()

       # Create edges for comments
        for _, row in comments_df.iterrows():
            comment_owner = row["CommentOwner"]
            post_user = row["User"]
            count = comment_counts.get(post_user, 0)  # Get the count of comments for this post

            # Set the edge width based on the count of comments (make it bigger for more comments)
            edge_width = count + 1  # Add 1 to avoid zero-width edges
            edge_label = f"comment (count: {count})"
            G.add_edge(comment_owner, post_user, value=edge_width, color="#1E90FF", arrows="to", label=edge_label)

        # Create edges for answers
        for _, row in answers_df.iterrows():
            comment_owner = row["CommentOwner"]
            answer_owner = row["AnswerOwner"]
            count = answers_df["CommentOwner"].value_counts().get(comment_owner, 0)  # Get the count of answers for this comment owner

            # Set the edge width based on the count of answers (make it bigger for more answers)
            edge_width = count + 1  # Add 1 to avoid zero-width edges
            G.add_edge(comment_owner, answer_owner, value=int(edge_width), color="#BAEAFE", arrows="to", label="answer")

        # Create mention edges
        for _, row in mentions_df.iterrows():
            comment_owner = row["CommentOwner"]
            mentioned_user = row["MentionedUser"]
            count = mentions_df["CommentOwner"].value_counts().get(comment_owner, 0)  # Get the count of mentions for this comment owner

            # Set the edge width based on the count of mentions (make it bigger for more mentions)
            edge_width = count + 1  # Add 1 to avoid zero-width edges
            G.add_edge(comment_owner, mentioned_user, value=int(edge_width), color="#FFD700", arrows="to", label="mention")

        # Create tag edges
        for _, row in tagged_users_df.iterrows():
            user = row["User"]
            tagged_user = row["TaggedUser"]
            count = tagged_users_df["User"].value_counts().get(user, 0)  # Get the count of tags for this user

            # Set the edge width based on the count of tags (make it bigger for more tags)
            edge_width = count + 1  # Add 1 to avoid zero-width edges
            G.add_edge(user, tagged_user, value=int(edge_width), color="#BFDB38", arrows="to", label="tag")

        return G

    def filter_data_by_date(data_df, start_date, end_date):
        # Check if "Posted on" column exists in the DataFrame
        if "Posted on" not in data_df:
            return data_df

        # Convert the "Posted on" column to datetime objects (if not already)
        if not is_datetime(data_df["Posted on"]):
            data_df["Posted on"] = pd.to_datetime(data_df["Posted on"])

        # Convert start_date and end_date to pandas datetime objects with UTC timezone
        start_date = pd.to_datetime(start_date).tz_localize('UTC')
        end_date = pd.to_datetime(end_date).tz_localize('UTC')

        # Filter the DataFrame based on the date range
        filtered_df = data_df[(data_df["Posted on"] >= start_date) & (data_df["Posted on"] <= end_date)]
        return filtered_df


    st.title("Social Network Analytics of Instagram Issue")

    # Get a list of files from the "inspersonpost" folder
    folder_path = "instaposts"
    files = [file.replace(".json", "") for file in os.listdir(folder_path) if file.endswith(".json")]

    # Find the newest file based on modification time
    newest_file = max(files, key=lambda file: os.path.getmtime(os.path.join(folder_path, f"{file}.json")))

    col1, col2, col3 = st.columns([2, 1, 1])

    # Multi-select files
    with col1:
        selected_files = st.multiselect("Select files", files, default=[newest_file], key='msinsta')  # Set default selection to the newest file

    # Filter by start and end date
    with col2:
        start_date = st.date_input("Start Date", key='stdinstissue')

    with col3: 
        end_date = st.date_input("End Date", key='edsnaissueinst')

    # Process selected files and filter data by date
    user_data_list = []
    comments_data_list = []

    # Initialize answers_data, all_user_data, and all_comments_data as None
    answers_data = None
    all_user_data = None
    all_comments_data = None
    all_tagged_users_data = None

    # Check if any files are selected
    if len(selected_files) > 0:
        G = nx.DiGraph()  # Initialize the graph outside the loop
        user_data_list = []  # Define user_data_list
        comments_data_list = []  # Define comments_data_list
        answers_data_list = []  # Define answers_data_list
        tagged_users_data_list = []  # Define tagged_users_data_list
        mentions_data_list = []  # Define mentions_data_list

        for file in selected_files:
            file_path = os.path.join(folder_path, f"{file}.json")  # Add .json extension to the selected file
            print("Processing file:", file_path)  # Print the file path for debugging

            # Pass both file_path and file name to process_json_file
            user_data, comments_data, answers_data, new_tagged_users_data, new_mentions_data = process_json_file(file_path, file)  # Update here

            # Add nodes for users, comment owners, and answer owners with respective colors for each file
            for user in user_data["User"]:
                G.add_node(user, color="orange")  # Set the color of user nodes to blue

            for comment_owner in comments_data["CommentOwner"]:
                G.add_node(comment_owner, color="#1E90FF")  # Set the color of comment owner nodes to green

            for answer_owner in answers_data["AnswerOwner"]:  # Use answers_data here
                G.add_node(answer_owner, color="#BAEAFE")  # Set the color of answer owner nodes to red

            for tagged_user in new_tagged_users_data["TaggedUser"]:
                G.add_node(tagged_user, color="#BFDB38ell")  # Set the color of tagged users nodes to yellow

            for mentioned_user in new_mentions_data["MentionedUser"]:
                G.add_node(mentioned_user, color="purple")  # Set the color of mentioned users nodes to purple

            user_data_list.append(user_data)
            comments_data_list.append(comments_data)
            answers_data_list.append(answers_data)
            tagged_users_data_list.append(new_tagged_users_data)
            mentions_data_list.append(new_mentions_data)

        # Concatenate data from all files
        all_user_data = pd.concat(user_data_list, ignore_index=True)
        all_comments_data = pd.concat(comments_data_list, ignore_index=True)
        all_answers_data = pd.concat(answers_data_list, ignore_index=True)
        all_tagged_users_data = pd.concat(tagged_users_data_list, ignore_index=True)
        all_mentions_data = pd.concat(mentions_data_list, ignore_index=True)

        # Filter data by date
        if start_date and end_date:
            all_user_data = filter_data_by_date(all_user_data, start_date, end_date)
            all_comments_data = filter_data_by_date(all_comments_data, start_date, end_date)

        if len(selected_files) > 0 and all_user_data is not None and all_comments_data is not None:
    # Create social network graph
            graph = create_social_network_graph(all_user_data, all_comments_data, all_answers_data, all_tagged_users_data, all_mentions_data)

            # Visualize the graph using Pyvis and Streamlit
            nt = Network(notebook=False, height='800px', width='100%', directed=True)  # Remove show_edge_labels method


            # Define colors for each group
            color_map = {
                "user": "orange",
                "comment_owner": "#1E90FF",
                "answer_owner": "#BAEAFE",
                "tagged_user": "#BFDB38",
                "mentioned_user": "purple"
            }

            for node, data in graph.nodes(data=True):
                # Get the color attribute based on the group or use a default color if the group is missing
                node_color = color_map.get(data.get("group", "user"), "orange")
                nt.add_node(node, color=node_color, size=data.get("size", 10))  # Use "size" attribute if available

            for edge in graph.edges():
                # Set the edge width based on the edge value (count)
                edge_width = graph.edges[edge].get("value", 1) + 1  # Add 1 to avoid zero-width edges

                nt.add_edge(
                    edge[0], edge[1], value=graph.edges[edge].get("value", 1),
                    width=edge_width, color=graph.edges[edge].get("color", "#1E90FF"),
                    title=graph.edges[edge].get("label", "comment"), arrows=graph.edges[edge].get("arrows", "to")
                )

            # Set the smoothness of the edges
            nt.set_edge_smooth("dynamic")

            # Save the graph to an HTML file
            nt.save_graph('html_files/instaissue_social_network.html')

        else:
            if len(selected_files) == 0:
                st.warning("Please select at least one file.")
            else:
                st.warning("No data available for the selected date range.")

        
        
    
#################################DEGREE CENTRALITY NETWORK ####################################


        import matplotlib.pyplot as plt
        import pandas as pd
        # Create social network graph
        graph = create_social_network_graph(all_user_data, all_comments_data, answers_data, all_tagged_users_data, all_mentions_data)

        # Calculate the degree centrality for each node
        degree_centrality = nx.degree_centrality(graph)

        # Sort the nodes based on degree centrality in descending order
        sorted_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)

        # Select the top ten nodes with highest degree centrality
        top_ten_nodes = sorted_nodes[:10]

        # Create a subgraph with the top ten nodes and their neighboring edges
        subgraph = graph.subgraph(top_ten_nodes)

        # Visualize the subgraph using Pyvis and Streamlit
        nt_subgraph = Network(notebook=False)
        nt_subgraph.from_nx(subgraph)
        nt_subgraph.save_graph("html_files/instaissue_degree_centrality_graph.html")

        # Calculate the degree centrality scores and ranks for the top ten actors
        top_ten_degree_scores = [degree_centrality[node] for node in top_ten_nodes]
        top_ten_degree_ranks = list(range(1, len(top_ten_nodes) + 1))

        # Create a DataFrame to hold the degree centrality scores and ranks
        top_ten_degree_df = pd.DataFrame({
            "Actor": top_ten_nodes,
            "Degree Centrality Score": top_ten_degree_scores,
            "Rank": top_ten_degree_ranks
        })

        # Generate a dynamic message based on the network data
        message = f"Above is a visual representation of the social network graph with the top ten actors based on degree centrality.\n"
        message += f"The top ten actors are: {', '.join(top_ten_nodes)}.\n"
        message += f"They have the following degree centrality scores: {', '.join(map(str, top_ten_degree_scores))}.\n"
        message += f"Degree centrality is a measure of the number of connections a node has in the network.\n"
        message += f"Higher degree centrality indicates that the actor is more central to the network."


    ######################################BETWEENESS CENTRALITY #############################################
        graph = create_social_network_graph(all_user_data, all_comments_data, answers_data, all_tagged_users_data, all_mentions_data)

        # Calculate the betweenness centrality for each node
        betweenness_centrality = nx.betweenness_centrality(graph)

        # Sort the nodes based on betweenness centrality in descending order
        sorted_nodes_by_betweenness = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)

        # Select the top ten nodes with highest betweenness centrality
        top_ten_nodes_by_betweenness = sorted_nodes_by_betweenness[:10]

        # Create a subgraph with the top ten nodes and their neighboring edges
        subgraph_by_betweenness = graph.subgraph(top_ten_nodes_by_betweenness)

        # Visualize the subgraph with betweenness centrality using Pyvis and Streamlit
        nt_subgraph_by_betweenness = Network(notebook=False)
        nt_subgraph_by_betweenness.from_nx(subgraph_by_betweenness)
        nt_subgraph_by_betweenness.save_graph("html_files/instaissue_betweenness_centrality_graph.html")

        # Calculate the betweenness centrality scores and ranks for the top ten nodes
        top_ten_betweenness_scores = [betweenness_centrality[node] for node in top_ten_nodes_by_betweenness]
        top_ten_betweenness_ranks = list(range(1, len(top_ten_nodes_by_betweenness) + 1))

        # Create a DataFrame to hold the betweenness centrality scores and ranks
        top_ten_betweenness_df = pd.DataFrame({
            "Node": top_ten_nodes_by_betweenness,
            "Betweenness Centrality Score": top_ten_betweenness_scores,
            "Rank": top_ten_betweenness_ranks
        })

        # Generate a dynamic message based on the network data for betweenness centrality
        message_betweenness = f"Above is a visual representation of the subgraph with the top ten nodes based on betweenness centrality.\n"
        message_betweenness += f"The top ten nodes are: {', '.join(top_ten_nodes_by_betweenness)}.\n"
        message_betweenness += f"They have the following betweenness centrality scores: {', '.join(map(str, top_ten_betweenness_scores))}.\n"
        message_betweenness += f"Betweenness centrality is a measure of how often a node acts as a bridge along the shortest path between two other nodes.\n"
        message_betweenness += f"Higher betweenness centrality indicates that the node plays a significant role in connecting other nodes in the network."


        
    #################################################################################################
        
        import networkx as nx
        from pyvis.network import Network
        import datetime
        import pandas as pd

        graph = create_social_network_graph(all_user_data, all_comments_data, answers_data, all_tagged_users_data, all_mentions_data)
        # Calculate the closeness centrality for each node
        closeness_centrality = nx.closeness_centrality(graph)

        # Sort the nodes based on closeness centrality in descending order
        sorted_nodes_by_closeness = sorted(closeness_centrality, key=closeness_centrality.get, reverse=True)

        # Select the top ten nodes with highest closeness centrality
        top_ten_nodes_by_closeness = sorted_nodes_by_closeness[:10]

        # Create a subgraph with the top ten nodes and their neighboring edges
        subgraph_by_closeness = graph.subgraph(top_ten_nodes_by_closeness)

        # Visualize the subgraph with closeness centrality using Pyvis and Streamlit
        nt_subgraph_by_closeness = Network(notebook=False)
        nt_subgraph_by_closeness.from_nx(subgraph_by_closeness)
        nt_subgraph_by_closeness.save_graph("html_files/instaissue_closeness_centrality_graph.html")

        # Calculate the closeness centrality scores and ranks for the top ten nodes
        top_ten_closeness_scores = [closeness_centrality[node] for node in top_ten_nodes_by_closeness]
        top_ten_closeness_ranks = list(range(1, len(top_ten_nodes_by_closeness) + 1))

        # Create a DataFrame to hold the closeness centrality scores and ranks
        top_ten_closeness_df = pd.DataFrame({
            "Node": top_ten_nodes_by_closeness,
            "Closeness Centrality Score": top_ten_closeness_scores,
            "Rank": top_ten_closeness_ranks
        })

        # Generate a dynamic message based on the network data for closeness centrality
        message_closeness = f"Above is a visual representation of the subgraph with the top ten nodes based on closeness centrality.\n"
        message_closeness += f"The top ten nodes are: {', '.join(top_ten_nodes_by_closeness)}.\n"
        message_closeness += f"They have the following closeness centrality scores: {', '.join(map(str, top_ten_closeness_scores))}.\n"
        message_closeness += f"Closeness centrality is a measure of how quickly a node can reach other nodes in the network.\n"
        message_closeness += f"Higher closeness centrality indicates that the node is more central and well-connected in the network."


#########################################################################################
        colinstissviz1, colinstissviz2, colinstissviz3, colinstissviz4=st.tabs(['Social Network','Main Actors', 'Bridging Actors', 'Supporting Actors'])

        with colinstissviz1:
        # Display the graph in Streamlit
            with open('html_files/instaissue_social_network.html', 'r') as f:
                html_string = f.read()
            st.components.v1.html(html_string, height=810, scrolling=True)
            
        with colinstissviz2:
            # Display the graph in Streamlit
            with open("html_files/instaissue_degree_centrality_graph.html", "r") as f:
                html_string = f.read()
            st.components.v1.html(html_string, height=1000, scrolling=True)

            # Display the dynamic message using st.markdown to render formatted markdown
            st.subheader("Main Actors Network Analysis Details")
            st.markdown(message)

            # Display the DataFrame and chart side by side using st.columns
            col1, col2 = st.columns([1, 1])  # Adjust the column widths as needed

            # Display the DataFrame in the first column
            col1.subheader("Top Ten Main Actors Data")
            col1.dataframe(top_ten_degree_df.set_index("Actor"))

            # Display the chart in the second column
            col2.subheader("Top Ten Main Actors Chart")
            col2.bar_chart(top_ten_degree_df.set_index("Actor")["Degree Centrality Score"])

        with colinstissviz3:
            # Display the graph with betweenness centrality in Streamlit
            with open("html_files/instaissue_betweenness_centrality_graph.html", "r") as f:
                html_string_betweenness = f.read()
            st.components.v1.html(html_string_betweenness, height=1000, scrolling=True)

            st.subheader("Bridging Actors Network Analysis Details")

            # Display the dynamic message for betweenness centrality
            st.caption(message_betweenness)

            # Display the DataFrame and chart side by side using st.columns
            col1, col2 = st.columns([1, 1])  # Adjust the column widths as needed

            # Display the DataFrame in the first column for betweenness centrality
            col1.subheader("Top Ten Bridging Actors Data")
            col1.dataframe(top_ten_betweenness_df.set_index("Node"))

            # Display the chart in the second column for betweenness centrality
            col2.subheader("Top Ten Bridging Actors Chart")
            col2.bar_chart(top_ten_betweenness_df.set_index("Node")["Betweenness Centrality Score"])

        with colinstissviz4:

                # Display the graph with closeness centrality in Streamlit
            with open("html_files/instaissue_closeness_centrality_graph.html", "r") as f:
                html_string_closeness = f.read()
            st.components.v1.html(html_string_closeness, height=1000, scrolling=True)

            st.subheader("Supporting Actors Network Analysis Details")

            # Display the dynamic message for closeness centrality
            st.caption(message_closeness)

            # Display the DataFrame and chart side by side using st.columns
            col1, col2 = st.columns(2)  # Adjust the column widths as needed

            # Display the DataFrame in the first column for closeness centrality
            col1.subheader("Top Ten Supporting Actors Data")
            col1.dataframe(top_ten_closeness_df.set_index("Node"))

            # Display the chart in the second column for closeness centrality
            col2.subheader("Top Ten Supporting Actors Char")
            col2.bar_chart(top_ten_closeness_df.set_index("Node")["Closeness Centrality Score"])






#########################issu topic modeling ##########################

    import streamlit as st
    import os
    import json
    import pandas as pd
    import gensim
    from gensim import corpora
    import pyLDAvis.gensim_models as gensimvis
    import pyLDAvis
    from wordcloud import WordCloud
    import re

    # Function to load data from JSON files in the "instaposts" folder
    def load_data(folder_path):
        data = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                with open(os.path.join(folder_path, filename), "r") as file:
                    posts = json.load(file)
                    data.extend(posts)
        return data

    # Function to perform topic modeling
    def perform_topic_modeling(data):
        # Extract caption text from the data
        captions = [post["Caption"] for post in data]

        # Preprocess the captions - remove unwanted characters and "http"
        processed_captions = [
            re.sub(r'[\"@():,.#?!_*]', '', caption.lower())  # Remove unwanted characters
                .replace("http", "")  # Remove the word "http"
            for caption in captions
        ]

        # Tokenize the captions and remove stopwords
        stop_words = set()
        # Add your custom stopwords here if needed
        stop_words.update(["dan", "di", "ke", "dari", "untuk", "yang", "juga", "tetapi", "akan", "walau",
        "walaupun", "meski", "meskipun", "adapun", "jika", "maka", "karena", "sebab",
            "oleh", "dengan", "dalam", "ini", "itu", "pada", "atau", "ada", "hal", "baca", 
            "saat", "ketika", "tersebut", "apabila", "kita", "saya", "kami", "kamu", "aku",
                "bisa", "dapat", "bagaimana", "menjadi", "sebagai", "tidak", "iya", "antara",
                "atas", "bawah", "ini"])

        tokenized_captions = [
            [word for word in caption.lower().split() if word not in stop_words]
            for caption in processed_captions
        ]

        

        # Create a Gensim dictionary and corpus
        dictionary = corpora.Dictionary(tokenized_captions)
        corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_captions]

        # Perform LDA topic modeling
        num_topics = 5  # You can change the number of topics as needed
        lda_model = gensim.models.LdaModel(
            corpus, num_topics=num_topics, id2word=dictionary, passes=20
        )

        # Visualize the topics using PyLDAvis
        vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(vis_data, "topic_modeling_result.html")

        return lda_model, vis_data
    
    def display_wordcloud(lda_model, topic_num):
        words = dict(lda_model.show_topic(topic_num, topn=30))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(words)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    # Streamlit app
    st.title("Instagram Caption Topic Modeling")

    # Load data from the "instapost" folder
    folder_path = "instaposts"
    data = load_data(folder_path)

    # Create a list of filenames for the select box
    file_names = [os.path.splitext(filename)[0] for filename in os.listdir(folder_path) if filename.endswith(".json")]

    # User input: Select file and date range
    inskptmcol1, inskptmcol2, inskptmcol3=st.columns([2,1,1])
    with inskptmcol1:
        selected_file = st.selectbox("Select a keyword", file_names, index=0, key='selinstissuet')
    with inskptmcol2:
        start_date = st.date_input("Start Date", key='sttpinsissue')
    with inskptmcol3:
        end_date = st.date_input("End Date", key='3dtissue(tpin5t)')

    # Filter data based on selected date range
    filtered_data = [
        post for post in data
        if start_date <= pd.to_datetime(post["Posted on"]).date() <= end_date
    ]

    st.set_option('deprecation.showPyplotGlobalUse', False)

    if len(filtered_data) == 0:
        st.warning("No posts found within the specified date range.")
    else:
        # Perform topic modeling
        lda_model, vis_data = perform_topic_modeling(filtered_data)

        # Display the PyLDAvis visualization
        with open("topic_modeling_result.html", "r") as file:
            html = file.read()
        st.components.v1.html(html, width=1500, height=800)

        # Display selectable word cloud for each topic
        st.subheader("Word Cloud for Each Topic")
        topic_list = [f"Topic {topic_num + 1}" for topic_num in range(lda_model.num_topics)]
        selected_topic = st.selectbox("Select a topic", topic_list, index=0)

        if selected_topic:
            topic_num = int(selected_topic.split()[-1]) - 1
            display_wordcloud(lda_model, topic_num)


################################## HASHTAG ANALYSIS ###############################################

    import streamlit as st
    import json
    import os
    import pandas as pd

    def load_data(file_path):
        with open(file_path, "r") as file:
            return json.load(file)

    def filter_by_date(posts, start_date, end_date):
        filtered_posts = []
        for post in posts:
            posted_on = pd.to_datetime(post["Posted on"]).date()
            if start_date <= posted_on <= end_date:
                filtered_posts.append(post)
        return filtered_posts

    def extract_hashtags(caption):
        return [tag.strip("#") for tag in caption.split() if tag.startswith("#")]

    def convert_to_date(date_str):
        return pd.to_datetime(date_str[:10]).date()

    def count_hashtags(posts):
        hashtag_counts = {}
        for post in posts:
            hashtags = extract_hashtags(post["Caption"])
            for hashtag in hashtags:
                hashtag_counts[hashtag] = hashtag_counts.get(hashtag, 0) + 1
        return hashtag_counts

    def display_hashtag_chart(hashtag_counts):
        st.bar_chart(hashtag_counts)

    st.title("Instagram Hashtag Analysis")

    # Get all JSON files from the "instaposts" folder
    files = [os.path.splitext(f)[0] for f in os.listdir("instaposts") if f.endswith(".json")]
    selected_file = st.selectbox("Choose an", files)

    # Load selected JSON file
    file_path = os.path.join("instaposts", selected_file + ".json")
    posts = load_data(file_path)

    # Filter posts by date range
    start_date = st.date_input("Start date", pd.to_datetime(posts[0]["Posted on"]).date())
    end_date = st.date_input("End date", pd.to_datetime(posts[-1]["Posted on"]).date())

    filtered_posts = [
        post for post in posts
        if start_date <= convert_to_date(post["Posted on"]) <= end_date
    ]

    
    # Hashtag Analysis and Chart Display
    st.subheader("Hashtag Analysis Result")
    hashtag_counts = count_hashtags(filtered_posts)
    display_hashtag_chart(hashtag_counts)


     # st.subheader("Filtered Posts")
    # for post in filtered_posts:
    #     st.write("Posted on:", post["Posted on"])
    #     st.write("Caption:", post["Caption"])
    #     hashtags = extract_hashtags(post["Caption"])
    #     st.write("Hashtags:", ", ".join(hashtags))
    #     st.write("---")




##################################SENTIMENT ISSUE ANALYSIS ########################################
   
    from pathlib import Path
    from textblob import TextBlob
    import streamlit as st
    import json
    import os
    import pandas as pd
    import plotly.graph_objects as go

    # Function to read JSON files and extract data
    def read_json_file(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    # Function to perform sentiment analysis on Bahasa Indonesia text
    def analyze_sentiment(text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

    # Function to process the data and create pie chart
    def process_data(data):
        df = pd.DataFrame(data)
        df["Caption Sentiment"] = df["Caption"].apply(analyze_sentiment)
        df["Comments Sentiment"] = df["Comments"].apply(lambda comments: sum(analyze_sentiment(comment["Text"]) for comment in comments) / len(comments) if comments else 0)
        return df

    # Function to plot the pie chart
    def plot_pie_chart(values, labels, colors, title):
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))])
        fig.update_layout(title_text=title)
        return fig

    def display_pie_charts(files_list, start_date, end_date):
        num_files = len(files_list)
        num_columns = num_files * 2  # Set the number of columns based on the number of selected JSON files (2 columns for each file)
        columns = st.columns(num_columns)

        # Convert start_date and end_date to pandas Timestamp objects with UTC timezone
        start_date = pd.to_datetime(start_date).tz_localize('UTC')
        end_date = pd.to_datetime(end_date).tz_localize('UTC')

        for i, file_name in enumerate(files_list):
            file_path = f"instaposts/{file_name}.json"
            data = read_json_file(file_path)
            df = process_data(data)

            # Convert the "Posted on" column to pandas Timestamp objects with UTC timezone
            df["Posted on"] = pd.to_datetime(df["Posted on"])

            # Filter data based on date range
            mask = (df["Posted on"] >= start_date) & (df["Posted on"] <= end_date)
            df = df[mask]


            # Caption Sentiment
            positive_caption = df[df["Caption Sentiment"] > 0].shape[0]
            neutral_caption = df[df["Caption Sentiment"] == 0].shape[0]
            negative_caption = df[df["Caption Sentiment"] < 0].shape[0]
            total_caption = df.shape[0]
            caption_trace = plot_pie_chart([positive_caption, neutral_caption, negative_caption], ['Positive', 'Neutral', 'Negative'],
                                        ['#66CC99', '#FFCC99', '#FF9999'], f"{file_name} - Caption Sentiment")

            # Comments Sentiment
            positive_comments = df[df["Comments Sentiment"] > 0].shape[0]
            neutral_comments = df[df["Comments Sentiment"] == 0].shape[0]
            negative_comments = df[df["Comments Sentiment"] < 0].shape[0]
            total_comments = df.shape[0]
            comments_trace = plot_pie_chart([positive_comments, neutral_comments, negative_comments], ['Positive', 'Neutral', 'Negative'],
                                            ['#66CC99', '#FFCC99', '#FF9999'], f"{file_name} - Comments Sentiment")

            # Display the pie charts in the corresponding columns
            with columns[i * 2]:
                st.plotly_chart(caption_trace)
            with columns[i * 2 + 1]:
                st.plotly_chart(comments_trace)

    # Main Streamlit app
    st.title("Sentiment Analysis of Posts and Comments")
    files_list = [Path(file_name).stem for file_name in os.listdir("instaposts") if file_name.endswith(".json")]
    selected_files = st.multiselect("Select keyword", files_list)

    if selected_files:
        start_date = st.date_input("Select Start Date", value=pd.Timestamp('2023-06-01'))
        end_date = st.date_input("Select End Date", value=pd.Timestamp('2023-06-30'))

        filtered_files = [file_name for file_name in selected_files]
        display_pie_charts(filtered_files, start_date, end_date)


#######################SENTIMENT ANALYSIS ISSUE PER USER ###############
    # import json
    # import pandas as pd
    # import streamlit as st
    # from pathlib import Path
    # from nltk.sentiment import SentimentIntensityAnalyzer

    # # Download the Bahasa Indonesian lexicon (only needed once)
    # import nltk
    # nltk.download('stopwords')
    # nltk.download('punkt')
    # nltk.download('vader_lexicon')

    # # Function to read JSON files and extract data
    # def read_json_file(file_path):
    #     with open(file_path, "r") as f:
    #         data = json.load(f)
    #     return data

    # def analyze_sentiment(text):
    #     text = str(text)  # Convert to a string to handle possible encoding issues
    #     if not text:
    #         return None
    #     sid = SentimentIntensityAnalyzer()
    #     sentiment_scores = sid.polarity_scores(text)
    #     return sentiment_scores

    # # Streamlit app
    # st.title("Sentiment Analysis per Comment Owner")
    # # st.write("Select JSON files from 'instaposts' folder and choose the date range.")

    # files_list = [Path(file_name).stem for file_name in os.listdir("instaposts") if file_name.endswith(".json")]

    # # Find the newest file based on modification time
    # newest_file = max(files_list, key=lambda file: os.path.getmtime(f"instaposts/{file}.json"))

    # # Default date range (e.g., last 7 days)
    # default_end_date = pd.to_datetime("today").date()
    # default_start_date = default_end_date - pd.Timedelta(days=7)

    # # Multi-select files
    # selected_files = st.multiselect("Select keyword", files_list, default=[newest_file], key='selfil5entu5erinst4issue')

    # data = []
    # for file_name in selected_files:
    #     file_path = f"instaposts/{file_name}.json"
    #     data.extend(read_json_file(file_path))

    # df = pd.DataFrame(data)

    # # Step 2: Filter data based on start and end dates
    # start_date = st.date_input("Select start date", value=default_start_date, key='stdinstaissuesent111')
    # end_date = st.date_input("Select end date", value=default_end_date, key='endinstaissusent222')

    # filtered_data = df[(df["Posted on"] >= str(start_date)) & (df["Posted on"] <= str(end_date))]

    # # Initialize an empty list to store the comments data
    # comments_data = []

    # # Iterate through the rows of the filtered_data DataFrame
    # for _, row in filtered_data.iterrows():
    #     owner = row["Owner"]
    #     comments = row["Comments"]
        
    #     # Check if comments exist and are not empty
    #     if comments:
    #         for comment in comments:
    #             # Extract the comment text
    #             comment_text = comment["Text"]
    #             # Append the comment and its owner to the comments_data list
    #             comments_data.append({"Owner": owner, "Text": comment_text})

    # # Create the comments_by_owner DataFrame from the comments_data list
    # comments_by_owner = pd.DataFrame(comments_data)

    # # Step 3: Group comments by comment owner and calculate sentiment percentages
    # if not filtered_data.empty:
    #     comments_by_owner = filtered_data.explode("Comments").reset_index(drop=True)
    #     # comments_by_owner = pd.concat([comments_by_owner.drop(['Comments'], axis=1), comments_by_owner['Comments'].apply(pd.Series)], axis=1)
    #     comments_by_owner = comments_by_owner.explode("Comments")[['Owner', 'Text']].reset_index(drop=True)



    #     # Debugging statements to inspect the DataFrame structure and data types
    #     print(comments_by_owner.columns)
    #     print(comments_by_owner.dtypes)

    #     # Perform sentiment analysis on comments and filter out empty comments
    #     comments_by_owner["Sentiment Scores"] = comments_by_owner["Text"].apply(analyze_sentiment)
    #     comments_by_owner = comments_by_owner.dropna(subset=["Sentiment Scores"])

    #     if not comments_by_owner.empty:
    #         comments_by_owner["Compound Score"] = comments_by_owner["Sentiment Scores"].apply(lambda score: score["compound"])
    #         comments_by_owner["Sentiment"] = comments_by_owner["Compound Score"].apply(lambda score: "positive" if score >= 0 else "negative")

    #         # Calculate sentiment percentages per comment owner
    #         sentiment_counts = comments_by_owner.groupby(["Owner", "Sentiment"]).size().unstack(fill_value=0)
    #         sentiment_counts["Total"] = sentiment_counts.sum(axis=1)

    #         # Handle the case when there are no comments with negative sentiment
    #         if "negative" in sentiment_counts.columns:
    #             sentiment_counts["Negative Percentage"] = sentiment_counts["negative"] / sentiment_counts["Total"]
    #         else:
    #             sentiment_counts["Negative Percentage"] = 0

    #         # Handle the case when there are no comments with positive sentiment
    #         if "positive" in sentiment_counts.columns:
    #             sentiment_counts["Positive Percentage"] = sentiment_counts["positive"] / sentiment_counts["Total"]
    #         else:
    #             sentiment_counts["Positive Percentage"] = 0

    #         sentiment_counts = sentiment_counts[["Positive Percentage", "Negative Percentage"]]

    #         st.write("Sentiment Analysis per comment owner Results:")
    #         st.bar_chart(sentiment_counts)

    #         # Step 4: Display the comments table
    #         st.write("Filtered Comments:")
    #         st.dataframe(comments_by_owner[["Owner", "Text", "Sentiment"]])
    #     else:
    #         st.write("No comments with valid sentiment found within the selected date range.")


##############################################

with tab3:
    st.header("DATA")
    containsta=st.container()
    with containsta:

        client = httpx.Client(
            headers={
                "x-ig-app-id": "936619743392459",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept": "*/*",
            }
        )

        def scrape_user(username: str):
            result = client.get(
                f"https://i.instagram.com/api/v1/users/web_profile_info/?username={username}",
            )
            data = json.loads(result.content)
            return data["data"]["user"]

        def save_to_json(data, save_dir, username):
            file_path = os.path.join(save_dir, f"{username}_data.json")
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)

        def parse_data(data):
            result = jmespath.search(
                """{
                name: full_name,
                username: username,
                id: id,
                category: category_name,
                business_category: business_category_name,
                phone: business_phone_number,
                email: business_email,
                bio: biography,
                bio_links: bio_links[].url,
                homepage: external_url,        
                followers: edge_followed_by.count,
                follows: edge_follow.count,
                facebook_id: fbid,
                is_private: is_private,
                is_verified: is_verified,
                profile_image: profile_pic_url_hd,
                video_count: edge_felix_video_timeline.count,
                videos: edge_felix_video_timeline.edges[].node.{
                    id: id, 
                    title: title,
                    shortcode: shortcode,
                    thumb: display_url,
                    url: video_url,
                    views: video_view_count,
                    tagged: edge_media_to_tagged_user.edges[].node.user.username,
                    captions: edge_media_to_caption.edges[].node.text,
                    comments_count: edge_media_to_comment.count,
                    comments_disabled: comments_disabled,
                    taken_at: taken_at_timestamp,
                    likes: edge_liked_by.count,
                    location: location.name,
                    duration: video_duration,
                    comments: edge_media_to_comment.edges[].node.{
                        owner: owner.username,
                        comment_date: created_at,
                        comment_text: text
                    }
                },
                image_count: edge_owner_to_timeline_media.count,
                images: edge_felix_video_timeline.edges[].node.{
                    id: id, 
                    title: title,
                    shortcode: shortcode,
                    src: display_url,
                    url: video_url,
                    views: video_view_count,
                    tagged: edge_media_to_tagged_user.edges[].node.user.username,
                    captions: edge_media_to_caption.edges[].node.text,
                    comments_count: edge_media_to_comment.count,
                    comments_disabled: comments_disabled,
                    taken_at: taken_at_timestamp,
                    likes: edge_liked_by.count,
                    location: location.name,
                    accesibility_caption: accessibility_caption,
                    duration: video_duration,
                    comments: edge_media_to_comment.edges[].node.{
                        owner: owner.username,
                        comment_date: created_at,
                        comment_text: text
                    }
                },
                saved_count: edge_saved_media.count,
                collections_count: edge_saved_media.count,
                related_profiles: edge_related_profiles.edges[].node.username
            }""",
                data,
            )
            return result

        # Streamlit app title
        st.title('Instagram User Search')

        # Streamlit form to input username
        username = st.text_input('Enter Instagram username')

        # Streamlit button to submit the form
        if st.button('Search'):
            try:
                # Create directory if it doesn't exist
                save_dir = os.path.join(os.getcwd(), 'insperson')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Scrape user data
                user_data = scrape_user(username)

                account_data = parse_data(user_data)

                # Save data to JSON file
                save_to_json(account_data, save_dir, username)

                # Parse user data
                

                # Display parsed user data
                st.subheader('User Data')
                st.json(account_data)

            except Exception as e:
                st.error('An error occurred: ' + str(e))




#######################################################
    

    import streamlit as st
    import instaloader
    import json
    import os
    import pytz
    from datetime import datetime

    def save_session_to_json(username, password, json_file):
        L = instaloader.Instaloader()

        # Login with the provided username and password
        L.login(username, password)

        # Save the session cookies to a JSON file
        with open(json_file, "w") as file:
            json.dump(L.context._session.cookies.get_dict(), file)

        print("Session cookies saved to {}".format(json_file))


    def load_session_from_json(username, json_file):
        L = instaloader.Instaloader()

        # Load the session cookies from the JSON file
        with open(json_file, "r") as file:
            cookies = json.load(file)

        # Set the cookies in the Instaloader context
        L.context._session.cookies.update(cookies)

        # Test if the session is logged in
        if L.test_login():
            print("Logged in as {}".format(username))
        else:
            raise SystemExit("Cookie import failed: Unable to log in with the provided cookies.")

        return L

    def logout(L):
        L.close()
        print("Logged out successfully.")

    def scrape_instagram_posts(L, profile, start_date, end_date):
        # Get the profile of the target user
        try:
            profile = instaloader.Profile.from_username(L.context, profile)
        except instaloader.exceptions.ProfileNotExistsException:
            st.error("Error: The provided Instagram username does not exist.")
            return []

        results = []
        for post in profile.get_posts():
            if post.date_utc.date() < start_date:
                # Since the posts are in chronological order, we can break the loop when we reach posts older than the start date.
                break

            if start_date <= post.date_utc.date() <= end_date:
                post_data = {
                    'title': post.title,
                    'Posted on': post.date_local,
                    'user': post.owner_username,
                    'Image URL': post.url,
                    'Caption': post.caption,
                    'Likes': post.likes,
                    'Tagged Users': post.caption_mentions,  # Get tagged users in post caption
                    'Comments': [],
                }

                # Get comment details for the post
                for comment in post.get_comments():
                    comment_data = {
                        'Owner': comment.owner.username,
                        'Posted on': convert_to_local_time(comment.created_at_utc),  # Convert UTC to local time
                        'Text': comment.text,
                        'Answers': [],  # Initialize list to store answers
                    }

                    # Get answers to comments
                    for answer in comment.answers:
                        answer_data = {
                            'Owner': answer.owner.username,
                            'Posted on': convert_to_local_time(answer.created_at_utc),
                            'Text': answer.text,
                        }
                        comment_data['Answers'].append(answer_data)

                    post_data['Comments'].append(comment_data)

                results.append(post_data)

        return results

    def convert_to_local_time(utc_time):
        local_time = utc_time.replace(tzinfo=pytz.utc).astimezone()  # Convert UTC to local time
        return local_time

    class CustomJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)

    def save_data_to_json(data, folder_path, file_name):
        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save data to JSON file using the custom encoder
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4, cls=CustomJSONEncoder)

        print("Data saved to {}".format(file_path))

    def process_json_file(file_path):
        try:
            # Check if the file is empty before loading the JSON data
            if os.path.getsize(file_path) == 0:
                st.warning("The JSON file is empty.")
                return None

            # Debug print to check the file_path
            print("File path:", file_path)

            with open(file_path, "r") as file:
                data = json.load(file)

            # Debug print to check the loaded data
            print("Loaded data:", data)

            return data
        except FileNotFoundError:
            st.warning("The JSON file was not found.")
        except json.JSONDecodeError as e:
            st.error("Error decoding JSON file: {}".format(e))
        return None

       
    # Scrape posts and comments
    # Streamlit app
    st.title("Instagram User Search")

    # Input username and password
    username = st.text_input("Enter your Instagram login name", key='log1nsp5ta')
    password = st.text_input("Enter your Instagram password", key='pa55sp5ta', type='password')

    # Input Instagram username to scrape
    profile = st.text_input("Enter Instagram username", key='pr0)inp5ta')

    # Input date range
    start_date = st.date_input("Start Date", key='in5tp3rp0ststd')
    end_date = st.date_input("End Date", key='e4tp3rpo5tins')

    # Scrape posts and comments
    if st.button("Scrape"):
        if username and password and profile and start_date and end_date:
            if start_date <= end_date:
                # Save session to JSON
                session_json_file = "instagram_session.json"
                save_session_to_json(username, password, session_json_file)

                # Load the session from the JSON file
                L = load_session_from_json(username, session_json_file)

                # Once the Instaloader session is obtained, proceed with scraping
                results = scrape_instagram_posts(L, profile, start_date, end_date)

                if results:
                    # Save data to JSON file
                    file_name = "{}.json".format(profile)  # Use profile name as the filename
                    save_data_to_json(results, "inspersonpost", file_name)

                    for post_data in results:
                        st.image(post_data['Image URL'])
                        st.write('Caption:', post_data['Caption'])
                        st.write('Likes:', post_data['Likes'])
                        st.write('Comments:')
                        for comment_data in post_data['Comments']:
                            st.write("- Owner:", comment_data['Owner'])
                            st.write("  Text:", comment_data['Text'])
                            st.write("  Posted on:", comment_data['Posted on'])
                        st.write('Posted on:', post_data['Posted on'])
                        st.write('---')

                    # Logout after scraping
                    logout(L)
                else:
                    st.warning("No posts found within the specified date range.")
            else:
                st.warning("End date should be greater than or equal to the start date.")
        else:
            st.warning("Please fill in all the fields.")



####################################SCRAPER INSTAGRAM KEYWORD SEARCH ##################

    import instaloader
    import streamlit as st
    import pandas as pd
    import os
    import json
    import pytz
    from datetime import datetime

    # Function to save the session to a JSON file
    def save_session_to_json(username, password, session_json_file):
        L = instaloader.Instaloader()
        L.login(username, password)
        L.save_session_to_file(filename=session_json_file)

    # Function to load the session from a JSON file
    def load_session_from_json(username, session_json_file):
        L = instaloader.Instaloader()
        L.load_session_from_file(username, session_json_file)
        return L

    def logout(L):
        L.close()
        print("Logged out successfully.")

    # Function to scrape Instagram posts based on hashtags and date range
    def scrape_instagram_posts(L, keyword, start_date, end_date):
        # Get posts based on hashtags
        posts = L.get_hashtag_posts(keyword)

        results = []
        for post in posts:
            if post.date_utc.date() < start_date:
                # Since the posts are in chronological order, we can break the loop when we reach posts older than the start date.
                break

            if start_date <= post.date_utc.date() <= end_date:
                post_data = {
                    'title': post.title,
                    'Posted on': convert_to_local_time(post.date_utc).isoformat(),  # Convert UTC to local time and serialize to string
                    'user': post.owner_username,
                    'Image URL': post.url,
                    'Caption': post.caption,
                    'Likes': post.likes,
                    'Tagged Users': post.caption_mentions,  # Get tagged users in post caption
                    'Comments': [],
                }

                # Get comment details for the post
                for comment in post.get_comments():
                    comment_data = {
                        'Owner': comment.owner.username,
                        'Posted on': convert_to_local_time(comment.created_at_utc).isoformat(),  # Convert UTC to local time and serialize to string
                        'Text': comment.text,
                        'Answers': [],  # Initialize list to store answers
                    }

                    # Get answers to comments
                    for answer in comment.answers:
                        answer_data = {
                            'Owner': answer.owner.username,
                            'Posted on': convert_to_local_time(answer.created_at_utc).isoformat(),  # Convert UTC to local time and serialize to string
                            'Text': answer.text,
                        }
                        comment_data['Answers'].append(answer_data)

                    post_data['Comments'].append(comment_data)

                results.append(post_data)

        return results

    def convert_to_local_time(utc_time):
        local_time = utc_time.replace(tzinfo=pytz.utc).astimezone()  # Convert UTC to local time
        return local_time

    # Streamlit app
    st.title("Instagram Posts based on Keyword and Date Range")

    # Input username and password for Instagram login
    username = st.text_input("Instagram Username")
    password = st.text_input("Instagram Password", type="password")

    # Input keyword and date range
    keyword = st.text_input("Enter a keyword to search for Instagram posts:")
    start_date = st.date_input("Start Date", key="stsek3yin5t")
    end_date = st.date_input("End Date", key='eds3keyin5ta')

    # Button to log in to Instagram and trigger the scraping process
    if st.button("Scrape", key='instscrpkeywords'):
        if username and password and keyword and start_date and end_date:
            if start_date <= end_date:
                # Save session to JSON
                session_json_file = "sessionpath/instsearch_session.json"
                save_session_to_json(username, password, session_json_file)

                # Load the session from the JSON file
                L = load_session_from_json(username, session_json_file)

                # Once the Instaloader session is obtained, proceed with scraping
                results = scrape_instagram_posts(L, keyword, start_date, end_date)

                if results:
                    # Save the scraped data as a JSON file
                    output_folder = "instaposts"
                    os.makedirs(output_folder, exist_ok=True)
                    output_file = os.path.join(output_folder, f"{keyword}.json")
                    with open(output_file, 'w') as f:
                        json.dump(results, f)

                    st.success(f"Scraped data has been saved to {output_file}")

                    # Logout after scraping
                    logout(L)

                else:
                    st.warning("No posts found within the specified date range.")
            else:
                st.warning("End date should be greater than or equal to the start date.")
        else:
            st.warning("Please fill in all the fields.")

 