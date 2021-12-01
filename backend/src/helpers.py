import base64
import requests
import json
from environs import Env


env = Env()
env.read_env()

client_key = env.str("TWITTER_CLIENT_KEY")
client_secret = env.str("TWITTER_CLIENT_SECRET")


def get_tweet_content(tweet_id):
    # Define your keys from the developer portal
    # Reformat the keys and encode them
    key_secret = '{}:{}'.format(client_key, client_secret).encode('ascii')

    # Transform from bytes to bytes that can be printed
    b64_encoded_key = base64.b64encode(key_secret)
    # Transform from bytes back into Unicode
    b64_encoded_key = b64_encoded_key.decode('ascii')

    base_url = 'https://api.twitter.com/'
    auth_url = '{}oauth2/token'.format(base_url)
    auth_headers = {
        'Authorization': 'Basic {}'.format(b64_encoded_key),
        'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
    }
    auth_data = {
        'grant_type': 'client_credentials'
    }
    auth_resp = requests.post(auth_url, headers=auth_headers, data=auth_data)

    access_token = auth_resp.json()['access_token']

    search_headers = {
        'Authorization': 'Bearer {}'.format(access_token)    
    }
    # Create the URL
    search_url = '{}1.1/statuses/show.json?id={}'.format(base_url, tweet_id)

    # Execute the get request
    search_resp = requests.get(search_url, headers=search_headers)
    # Get the data from the request
    Data = json.loads( search_resp.content )

    # Print out the data!
    # print(Data)
    text = Data.get('text')
    img_url = ''
    try: img_url = Data['entities']['media'][0]['media_url']
    except KeyError: pass
    return text, img_url
