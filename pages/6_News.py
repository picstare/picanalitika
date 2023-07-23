import streamlit as st
import json
import pandas as pd
import numpy as np
from datetime import datetime
import nltk
from nltk.tree import Tree
import os
import re
import gensim
from nltk import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora
import folium
import plotly.graph_objs as go
import dateparser

# from gensim.utils import simple_preprocess
from pprint import pprint
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora, models
import pyLDAvis.gensim_models as gensimvis
# import PylDavis
import pyLDAvis.gensim
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import utils
from utils import logout
from streamlit_extras.switch_page_button import switch_page
import gensim.models.ldamodel as lda
import os
import json
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import streamlit as st
from nltk.corpus import stopwords
import Sastrawi

# Download the necessary NLTK resources
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('stopwords')


# Define the punctuation symbols
punctuation = "!\"$%&'()*+,-./:;<=>?[\\]^_`{|}~•@"


st.set_page_config(page_title='picanalitika | News Analysis', page_icon=':newspaper:', layout='wide')
a, b = st.columns([1, 10])
with a:
    st.image("img/newsicon.png", width=120)
    st.text("")
with b:
    st.title("News Analysis")

hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            div.vis-configuration-wrapper{display: block; width: 400px;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)



with st. sidebar:
    if st.button("Logout"):
        logout()
        switch_page('home')

# with st.sidebar:

#         filelist = []
#         for root, dirs, files in os.walk("newsdata"):
#             for file in files:
#                 filename = file
#                 filelist.append(filename)
#         # st.write(filelist)

#         optionfile = st.selectbox("Select file:", filelist, index=0)

#         newsdf = pd.read_json("newsdata/" + optionfile)
#         # articledf1['DateTime'] = pd.to_datetime(articledf1['DateTime'], unit='ms')

#         st.write("Choosen file to analyze:", optionfile)


listTabs = [
    "👨‍💼 News Analysis",
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
whitespace = 40
tab1, tab2= st.tabs([s.center(whitespace,"\u2001") for s in listTabs])

with tab1:
   





    # @st.cache_data
    # def analyze_data(json_data, file_name):
    #     df = pd.DataFrame(json_data)
    #     df['date'] = df['date'].astype(str).str.strip()  # Convert to string and remove leading/trailing spaces
    #     df['date'] = df['date'].apply(dateparser.parse)
    #     df['query'] = df['query'].str.strip()  # Remove leading/trailing spaces from the query column
    #     df['file_name'] = file_name  # Add file name as a column
    #     return df

    # # Get the list of files in the "newscraped" folder
    # folder_path = "newscraped"
    # files = os.listdir(folder_path)

    # # Initialize an empty DataFrame to store the data from all files
    # df = pd.DataFrame()

    # # Read and concatenate the data from all files
    # for file in files:
    #     file_path = os.path.join(folder_path, file)
    #     try:
    #         with open(file_path, "r") as f:
    #             data = json.load(f)
    #             temp_df = analyze_data(data, file)  # Call the analyze_data function
    #             df = pd.concat([df, temp_df], ignore_index=True)
    #     except json.JSONDecodeError:
    #         continue

    # # Convert the "date" column to datetime type with timezone-aware values
    # df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

    # # Display the query selection dropdown
    # queries = df["query"].unique()
    # default_queries = queries[:4]  # Select the first 4 queries as default
    # selected_queries = st.multiselect("Select Queries", queries, default=default_queries)

    # # Calculate the default start and end date as a 1-month range
    # default_start_date = pd.to_datetime(pd.Timestamp.now().normalize() - pd.DateOffset(months=1)).tz_localize('UTC')
    # default_end_date = pd.to_datetime(pd.Timestamp.now().normalize()).tz_localize('UTC')

    # # Display the date range selection inputs
    # start_date = st.date_input("Select Start Date", default_start_date.date())
    # end_date = st.date_input("Select End Date", default_end_date.date())

    # # Convert start_date and end_date to datetime objects with UTC timezone
    # start_date = pd.to_datetime(start_date).tz_localize('UTC')
    # end_date = pd.to_datetime(end_date).tz_localize('UTC')

    # # Filter the DataFrame based on the selected queries and date range
    # filtered_df = df[df["query"].isin(selected_queries)]
    # filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]

    # # Group the data based on time series interval and query, and calculate the counts
    # filtered_df['time_group'] = filtered_df['date'].dt.floor('H')
    # grouped_df = filtered_df.groupby(['query', 'time_group']).size().reset_index(name='count')

    # # Create the line chart using pandas plot function
    # fig, ax = plt.subplots(figsize=(10, 6))  # Set width: 10 inches, height: 6 inches

    # for query in selected_queries:
    #     query_data = grouped_df[grouped_df['query'] == query]
    #     ax.plot(query_data['time_group'], query_data['count'], label=query)

    # # Configure the chart
    # ax.set_xlabel('Date', fontsize=8)
    # ax.set_ylabel('Count', fontsize=8)
    # ax.legend(fontsize=10)

    # # Set the font size of tick labels on both axes
    # ax.tick_params(axis='x', labelsize=5)
    # ax.tick_params(axis='y', labelsize=5)

    # header_text = 'Time Series of News on ' + ', '.join(selected_queries)
    # st.header(header_text)

    # # Display the chart
    # st.pyplot(fig)



   ##################NER################################
    # @st.cache_data
    # def perform_ner(content):
    # # Tokenize the content into words
    #     words = word_tokenize(content)

    #     # Remove punctuation from the words
    #     words = [word for word in words if word not in punctuation]

    #     # Perform POS tagging on the words
    #     tagged_words = pos_tag(words)

    #     # Perform NER using the named entities chunker
    #     entities = ne_chunk(tagged_words)

    #     # Extract the entities of type PERSON, ORGANIZATION, and GPE
    #     extracted_persons = []
    #     extracted_organizations = []
    #     extracted_places = []
    #     for entity in entities:
    #         if hasattr(entity, 'label'):
    #             if entity.label() == 'PERSON':
    #                 extracted_persons.append(' '.join([child[0] for child in entity]))
    #             elif entity.label() == 'ORGANIZATION':
    #                 extracted_organizations.append(' '.join([child[0] for child in entity]))
    #             elif entity.label() == 'GPE':
    #                 extracted_places.append(' '.join([child[0] for child in entity]))

    #     return extracted_persons, extracted_organizations, extracted_places

    # # Define the path to the newscraped folder
    # folder_path = 'newscraped'

    # # Get the list of JSON files in the folder
    # json_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.json')]

    # # File selection using Streamlit
    # selected_files = st.multiselect("Select files to process:", json_files, default=json_files[:1])

    # # Create a list to store the extracted entities DataFrames
    # dataframes = []

    # # Process each selected JSON file
    # for filename in selected_files:
    #     file_path = os.path.join(folder_path, filename)

    #     with open(file_path, 'r') as file:
    #         # Load the JSON data
    #         data = json.load(file)

    #         # Process each article in the data
    #     for article in data:
    #         # Extract the content, date, and title
    #         content = article['content']
    #         date = article['date']
    #         title = article['title']

    #         # Perform NER on the content
    #         # persons, organizations, places = perform_ner(content)
    #         persons, organizations, places = perform_ner(str(content))

    #         # Create a DataFrame for the current article
    #         article_dataframe = pd.DataFrame({
    #             'Date': [date],
    #             'Title': [title],
    #             'Content': [content],
    #             'Person': [', '.join(persons)],
    #             'Organization': [', '.join(organizations)],
    #             'Place': [', '.join(places)]
    #         })

    #         # Append the article DataFrame to the list
    #         dataframes.append(article_dataframe)

    #     # Concatenate all DataFrames in the list
    #     concatenated_dataframe = pd.concat(dataframes, ignore_index=True)

    #     # Display the DataFrame
    #     st.write(concatenated_dataframe)

#########################SUMMARY#######################################
    
        # def clean_content(text):
        #     if pd.isnull(text) or not isinstance(text, str):
        #         return ""
        #     text = re.sub('<.*?>', '', text)
        #     text = re.sub('[^\\w\\s]', '', text)
        #     text = re.sub('\\s+', ' ', text)
        #     text = text.lower()
        #     text = text.replace("KOMPAS.com", "")  # Remove "KOMPAS.com" substring
        #     text = text.replace("kompascom", "")  # Remove "kompascom" substring
        #     return text

        # df['cleaned_content'] = df['content'].apply(clean_content)
        # df['tags'] = df['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

       
        # import openai

        # # Set up OpenAI API credentials
        # openai.api_key = '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'

        # # Create a function for generating summaries
        # @st.cache_data
        # def generate_summary(content):
        #     # Truncate the content to fit within the model's maximum context length
        #     max_context_length = 4097
        #     truncated_content = content[:max_context_length]
            
        #     # Generate the summary using OpenAI language model
        #     response = openai.Completion.create(
        #         engine='text-davinci-003',
        #         prompt=truncated_content,
        #         max_tokens=100,
        #         temperature=0.5,
        #         n=1,
        #         stop=None
        #     )
            
        #     # Extract the summarized text from the OpenAI response
        #     summary = response.choices[0].text.strip()
            
        #     return summary

        # # Assuming you have a DataFrame named 'df' with a 'cleaned_content' column

        # # Create an empty list to store the summaries
        # summaries = []

        # # Iterate over the 'cleaned_content' column and generate summaries
        # for content in df['cleaned_content']:
        #     # Generate the summary using the cached function
        #     summary = generate_summary(content)
            
        #     # Append the summary to the list
        #     summaries.append(summary)

        # # Add the summaries as a new column in the DataFrame
        # df['summary'] = summaries

        # # Print the DataFrame with summaries
        # st.dataframe(df)

        import streamlit as st
        import pandas as pd
        import glob
        import json
        import plotly.graph_objects as go
        import dateparser
        import datetime
        import pytz

        # Calculate the default start and end date as a 1-month range
        default_start_date = datetime.datetime.now().date() - datetime.timedelta(days=30)
        default_end_date = datetime.datetime.now().date()

        def load_data():
            file_list = glob.glob("newscraped/*.json")
            data = []
            for file_path in file_list:
                with open(file_path, 'r') as f:
                    content = json.load(f)
                    data.extend(content)
            return data

        def filter_data(data, start_date, end_date, keywords):
            filtered_data = []
            for item in data:
                date_string = item['date']
                if date_string is not None:
                    parsed_date = dateparser.parse(date_string)
                    if parsed_date is not None:
                        # Convert parsed_date to offset-aware datetime
                        parsed_date = parsed_date.astimezone(pytz.utc)

                        # Ensure start_date and end_date are offset-aware datetime
                        if not start_date.tzinfo:
                            start_date = start_date.astimezone(pytz.utc)
                        if not end_date.tzinfo:
                            end_date = end_date.astimezone(pytz.utc)

                        if start_date <= parsed_date <= end_date:
                            for keyword in keywords:
                                if keyword in item['content']:
                                    filtered_item = item.copy()
                                    filtered_item['date'] = parsed_date
                                    filtered_item['keyword'] = keyword

                                    # Check if 'tags' key exists in the item before adding it to filtered_data
                                    if 'tags' in filtered_item:
                                        filtered_item['tags'] = filtered_item['tags'] if isinstance(filtered_item['tags'], list) else []
                                    else:
                                        filtered_item['tags'] = []

                                    filtered_data.append(filtered_item)
                                    break  # Only add the article once for each keyword
            return filtered_data


        def plot_timeseries_chart(data, keywords, start_date, end_date):
            # Convert the "date" column to datetime type with timezone-aware values
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

            # Check if 'tags' column exists in the DataFrame
            if 'tags' in df.columns:
                # Apply lambda function only if 'tags' column exists
                df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])

            # Group the data based on time series interval, keyword, and calculate the counts
            grouped_df = df.groupby(['keyword', pd.Grouper(key='date', freq='D')]).size().reset_index(name='count')

            # Create the line chart using plotly.graph_objects
            fig = go.Figure()

            for keyword in keywords:
                query_data = grouped_df[grouped_df['keyword'] == keyword]
                fig.add_trace(go.Scatter(x=query_data['date'], y=query_data['count'], name=keyword))

            fig.update_layout(
                title="Time Series of News Articles by Keyword",
                xaxis_title="Date",
                yaxis_title="Count",
                showlegend=True,
                width=800,  # Set the width of the chart
                height=600  # Set the height of the chart
            )

            header_text = 'Time Series of News for Keywords: ' + ', '.join(keywords)
            st.subheader(header_text)

            # Display the chart
            st.plotly_chart(fig)


        # Load the data
        data = load_data()

        # Sidebar filters
        st.header('Time Series Analysis')

        keywords = st.text_input("Keywords (comma-separated)").split(',')
     
        start_date = st.date_input("Start Date", default_start_date)
        end_date = st.date_input("End Date", default_end_date)

        # Convert start_date and end_date to datetime objects with UTC timezone
        start_date = pd.to_datetime(start_date).tz_localize('UTC')
        end_date = pd.to_datetime(end_date).tz_localize('UTC')


        # Filter the data based on the filters
        filtered_data = filter_data(data, start_date, end_date, keywords)

        # Convert the filtered data to DataFrame and apply fixes for Arrow compatibility
        # Convert the filtered data to DataFrame and apply fixes for Arrow compatibility
        df = pd.DataFrame(filtered_data)
        st.dataframe(df.columns)
        print(df.columns)  # Print the columns of the DataFrame
        df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])

        # Plot the time series chart
        plot_timeseries_chart(df, keywords, start_date, end_date)


        # Display the DataFrame
        df = pd.DataFrame(filtered_data)
        st.dataframe(df)
        print(df)

###########################################TOPIC MODEL########################################################


        from gensim import corpora
        import Sastrawi.StopWordRemover.StopWordRemover
        import pyLDAvis.gensim_models as gensimvis
        import pyLDAvis

        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        
        # Define a list of Indonesian stopwords
        stopwords_id = [
            # Add your list of stopwords here
            "dan", "di", "ke", "dari", "untuk", "yang", "juga", "tetapi", "akan", "walau",
              "walaupun", "meski", "meskipun", "adapun", "jika", "maka", "karena", "sebab",
                 "oleh", "dengan", "dalam", "ini", "itu", "pada", "atau", "ada", "hal", "baca", 
                 "saat", "ketika", "tersebut", "apabila", "kita", "saya", "kami", "kamu", "aku",
                    "bisa", "dapat", "bagaimana", "menjadi", "sebagai", "tidak", "iya", "antara",
                       "atas", "bawah"   # Example stopwords
        ]

        def clean_content(text):
            if pd.isnull(text) or not isinstance(text, str):
                return ""
            text = re.sub('<.*?>', '', text)
            text = re.sub('[^\\w\\s]', '', text)
            text = re.sub('\\s+', ' ', text)
            text = text.lower()
            text = text.replace("KOMPAS.com", "")
            text = text.replace("kompascom", "")

            # Remove stopwords
            word_tokens = word_tokenize(text)
            filtered_text = [word for word in word_tokens if word.lower() not in stopwords_id]

            return " ".join(filtered_text)  
            
            # # Perform stemming
            # stemmer = StemmerFactory().create_stemmer()
            # stemmed_text = [stemmer.stem(word) for word in filtered_text]
            
            # return " ".join(stemmed_text)
        
     

        df['cleaned_content'] = df['content'].apply(clean_content)
        df['tags'] = df['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

        
        with st.expander("See Data"):
            st.dataframe(df)


        # Create a list of documents

        documents = df['cleaned_content'].values.tolist()

        # Tokenize the documents
        tokenized_docs = [doc.split() for doc in documents]

        # Create a dictionary representation of the documents
        dictionary = corpora.Dictionary(tokenized_docs)

        # Create a BoW representation of the documents
        bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

        # Set the number of topics
        num_topics = 5

        # Train the LDA model
        lda_model = gensim.models.LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=10)

        # Get the top keywords and probabilities for each topic
        topic_results = []
        for topic_id, topic in lda_model.show_topics(num_topics=num_topics, formatted=False):
            keywords = [word for word, prob in topic]
            probabilities = [prob for word, prob in topic]
            topic_result = {
                "Topic": f"Topic {topic_id + 1}",
                "Keywords": ", ".join(keywords),
                "Probabilities": probabilities  # Store the probabilities as a list
            }
            topic_results.append(topic_result)

        # Sort the topic results based on the maximum probability
        topic_results_sorted = sorted(topic_results, key=lambda x: max(x["Probabilities"]), reverse=True)

        # Prepare the data for the table
        table_data = []
        for topic_result in topic_results_sorted:
            table_row = {
                "Topic": topic_result["Topic"],
                "Keywords": topic_result["Keywords"],
                "Probabilities": ", ".join(map(str, topic_result["Probabilities"]))
            }
            table_data.append(table_row)

        # Display the topics, keywords, and probabilities in a table
        

        # Prepare the visualization data
        vis = gensimvis.prepare(lda_model, bow_corpus, dictionary)

        # Save the pyLDAvis visualization as an HTML file
        html_file_path = "pyldavis.html"
        pyLDAvis.save_html(vis, html_file_path)

        # Display pyLDAvis for each selected query
        html_content = open(html_file_path, 'r', encoding='utf-8').read()
        st.header(f'Topics of News')
        st.components.v1.html(html_content, width=1500, height=800)

        st.write("Topics and Keywords with Probabilities:")
        st.table(pd.DataFrame(table_data))

        
#######################SENTIMENTANALISIS#############################
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        st.header ('Sentiment Analysis')
        # Create an instance of the VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()

        # Perform sentiment analysis on each document in the cleaned_content column
        sentiments = []
        for doc in df['cleaned_content']:
            sentiment_scores = analyzer.polarity_scores(doc)
            sentiments.append(sentiment_scores)

        # Add the sentiment scores to the DataFrame
        df['Sentiment'] = sentiments

        # Display the DataFrame with sentiment scores
        st.subheader("Sentiment Analysis Results:")
        
       

        # Extract the compound scores from the 'Sentiment' column
        compound_scores = df['Sentiment'].apply(lambda x: x['compound'])

        col1, col2 =st.columns(2)

        with col1:

            # Calculate the average sentiment score
            average_sentiment = compound_scores.mean()
            # Display the average sentiment score
            st.write("Average Sentiment Score:", average_sentiment)

            st.dataframe(df[['cleaned_content', 'Sentiment']])

            

        import plotly.graph_objects as go

        with col2:

            # Calculate the count of sentiments
            sentiment_counts = df['Sentiment'].apply(lambda x: x['compound'] > 0).value_counts()
            positive_count = sentiment_counts.get(True, 0)
            negative_count = sentiment_counts.get(False, 0)
            neutral_count = len(df) - positive_count - negative_count

            # Create a pie chart
            fig = go.Figure(data=go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[positive_count, negative_count, neutral_count],
                hole=0.4
            ))

            # Set the chart title
            fig.update_layout(
                title="Sentiment Analysis of News",
                showlegend=True,
                
            )

            # Display the pie chart
            
            st.plotly_chart(fig, width=1000)


##############################################################################################

    

##########################################################
with tab2:
    import subprocess
    import streamlit as st
    import json
    import os
    import re
    from bs4 import BeautifulSoup

    SPIDERS = {
        "Antara News": "scrapeprojects/newsscraper/newsscraper/spiders/antarascraper.py",
        "Kompas.com":  "scrapeprojects/newsscraper/newsscraper/spiders/kompascomscraper.py",
        "Detik News":  "scrapeprojects/newsscraper/newsscraper/spiders/detiknews.py",
        "Tempo.co": "scrapeprojects/newsscraper/newsscraper/spiders/tempo.py",
        "Jawa Pos" :  "scrapeprojects/newsscraper/newsscraper/spiders/jawaposscraper.py",
        "Berita Jatim" :  "scrapeprojects/newsscraper/newsscraper/spiders/beritajatim.py",
        "Gresik Satu" :  "scrapeprojects/newsscraper/newsscraper/spiders/gresiksatu.py",
        "Info Gresik" :  "scrapeprojects/newsscraper/newsscraper/spiders/infogresik.py",
        # Add more spiders here if needed
    }

    SCRAPY_EXECUTABLE = r".venv/bin/scrapy"  # Specify the absolute path to the Scrapy executable
    
    @st.cache_resource
    def run_spider(spider_name, query):
        output_folder = "newscraped"
        output_file = f"{output_folder}/{spider_name}-{query}.json"  # Specify the desired output file path with the spider name and query in the file name

        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        if os.path.exists(output_file):
            # If the file exists, load the existing data
            with open(output_file, "r") as file:
                existing_data = json.load(file)
        else:
            # If the file does not exist, initialize an empty list for data
            existing_data = []

        # Temporary output file path
        tmp_output_file = "newscraped/tmp.json"

        # Run the spider
        command = [
            SCRAPY_EXECUTABLE,
            "runspider",
            SPIDERS.get(spider_name),
            "-a",
            f"query={query}",
            "-t",
            "jsonlines",
            "-o",
            tmp_output_file,
        ]

        if command:
            subprocess.run(command)
            st.success(f"Spider '{spider_name}' finished running.")

            if os.path.exists(tmp_output_file) and os.path.getsize(tmp_output_file) > 0:
                new_data = []

                # Load the newly scraped data from the temporary output file line by line
                with open(tmp_output_file, "r") as file:
                    for line in file:
                        # Process each line as a separate JSON object
                        try:
                            data = json.loads(line)
                            new_data.append(data)
                        except json.JSONDecodeError as e:
                            st.warning(f"Invalid JSON data in the temporary output file: {str(e)}")
                            continue

                # Append the newly scraped data to the existing data
                existing_data.extend(new_data)

                # Write the combined data to the output file
                with open(output_file, "w") as file:
                    json.dump(existing_data, file)

                # Close the temporary output file
                file.close()

                # Remove the temporary output file
                os.remove(tmp_output_file)

                st.success("New data was scraped and saved.")
            else:
                st.warning("No new data was scraped.")
        else:
            st.error(f"Spider '{spider_name}' not found.")


    st.title("📰 Search News")

    # Form inputs
    with st.form("Spider Form"):
        query = st.text_input("Search:")
        spider_name = st.selectbox("Select Media Outlets:", list(SPIDERS.keys()))
        submitted = st.form_submit_button("Search")

    # Run button
    if submitted:
        if not query:
            st.error("Please enter a query.")
        else:
            st.info(f"Running spider '{spider_name}'...")
            run_spider(spider_name, query)


    @st.cache_data(ttl=3600)
    def load_json_data(file_path):
        # Load the JSON data from the file
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
            return data
        except (IOError, json.JSONDecodeError) as e:
            st.error(f"Error loading JSON file: {file_path} - {str(e)}")
            return None


    folder_path = "newscraped"

    # Get all the files in the folder
    files = os.listdir(folder_path)

    # Filter out only the JSON files
    json_files = [file for file in files if file.endswith(".json")]

    # List to store all articles
    all_articles = []

    # Loop over the JSON files
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        # st.write(f"Processing file: {file_path}")
        st.write(f"")

        # Load the JSON data from the file
        data = load_json_data(file_path)
        if data is None:
            st.warning(f"No data loaded for file: {file_path}")
            continue

        # Append articles to the list
        all_articles.extend(data)

    # Sort articles by date in descending order
    sorted_articles = sorted(all_articles, key=lambda x: x.get("date") or "", reverse=True)

    # Display the sorted articles in the Streamlit app
    for i, article in enumerate(sorted_articles[:100]):
        query = article.get("query")
        title = article.get("title")
        date = article.get("date")
        image_url = article.get("image_url")
        content = article.get("content")
        author = article.get("author")
        tags = article.get("tags")
        url = article.get("url")

        # Handle missing or empty fields
        if title is None or title.strip() == "":
            title = "No title available"
        if author is None or author.strip() == "":
            author = "No author available"
        if date is None or date.strip() == "":
            date = "No date available"
        if content is None or content.strip() == "":
            content = "No content available"
        if tags is None or len(tags) == 0:
            tags = "No tags available"
        if url is None or url.strip() == "":
            url = "No URL available"

        # Remove HTML tags from the content
        soup = BeautifulSoup(content, "html.parser")
        clean_content = re.sub(r"<.*?>", "", soup.get_text())

        # Display the data
        st.write("Keyword:", query)
        st.write(title)
        st.write(author)
        st.write(date)

        # Check if the image URL is not None
        if image_url is not None:
            try:
                st.image(image_url)
            except Exception as e:
                st.warning("Error loading image.")
        else:
            st.warning("No image available.")

        # Handle the content field when it's a list
        if isinstance(content, list):
            content = " ".join(content)

        # Check if content is None or an empty string
        if content is None or content.strip() == "":
            content = "No content available"

        st.write(content)
        st.write("Tags:", tags)
        st.write(url)
        st.write("---")








