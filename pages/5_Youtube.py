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


with tab2:
    st.header("ISSUE")

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
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")

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
            response = youtube.search().list(part='id', q=channel_name, maxResults=1, type='channel').execute()
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
        def save_result(result, channel_name, start_date, end_date):
            folder_name = "ytaccscrap"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # Generate the file title
            file_title = f"{channel_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"

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





