import json
import pandas as pd
from textblob import TextBlob


def read_json(json_file: str)->list:
    """
    json file reader to open and read json files into a list
    Args:
    -----
    json_file: str - path of a json file
    
    Returns
    -------
    length of the json file and a list of json
    """
    
    tweets_data = []
    for tweets in open(json_file,'r'):
        tweets_data.append(json.loads(tweets))
    
    
    return len(tweets_data), tweets_data

class TweetDfExtractor:
    """
    this function will parse tweets json into a pandas dataframe
    
    Return
    ------
    dataframe
    """
    def __init__(self, tweets_list):
        
        self.tweets_list = tweets_list

    # an example function
    def find_statuses_count(self)->list:
        statuses_count = []
        for tweet in self.tweets_list:
            statuses_count.append(tweet["user"]["statuses_count"])
        return statuses_count
        
    def find_full_text(self)->list:
        text = []
        for tweet in self.tweets_list:
            text.append(tweet["full_text"])
        return text
    
    #still not fixed
    def find_sentiments(self, text)->list:
        #text is a list
        blobs = [TextBlob(tx) for tx in text]
        polarity = [blb.sentiment.polarity for blb in blobs ]
        subjectivity = [blb.sentiment.subjectivity for blb in blobs]
        
        return polarity, subjectivity

    def find_created_time(self)->list:
        created_at = []
        for tweet in self.tweets_list:
            created_at.append(tweet['created_at'])
        return created_at

    def find_source(self)->list:
        source = [x['source'] for x in self.tweets_list]
        return source

    def find_screen_name(self)->list:
        screen_name = [x['user']['screen_name'] for x in self.tweets_list]
        return screen_name

    def find_followers_count(self)->list:
        followers_count = [x['user']['followers_count'] for x in self.tweets_list]
        return followers_count

    def find_friends_count(self)->list:
        friends_count = [x['user']['friends_count'] for x in self.tweets_list]
        return friends_count        

    #example
    def is_sensitive(self)->list:
        is_sensitive = [x['possibly_sensitive'] if 'possibly_sensitive' in x else None for x in self.tweets_list]

        return is_sensitive

    def find_lang(self)->list:
        lang = [x["lang"] for x in self.tweets_list]
        return lang
    
    def find_favourite_count(self)->list:
        favourite_count = [x['user']['favourites_count'] for x in self.tweets_list]
        return favourite_count
    
    def find_retweet_count(self)->list:
        retweet_count = [x['retweet_count'] for x in self.tweets_list]
        return retweet_count

    def find_hashtags(self)->list:
        hashtags = [x["entities"]['hashtags'] for x in self.tweets_list]
        return hashtags

    def find_mentions(self)->list:
        mentions = [x['entities']['user_mentions'] for x in self.tweets_list]
        return mentions


    def find_location(self)->list:
        locations = []
        for tweet in self.tweets_list:
            try:
                locations.append(tweet['user']['location'])
            except TypeError:
                locations.append('')
                
        return locations

    def get_tweet_df(self, save=False, file_name = 'processed_tweet_data.csv')->pd.DataFrame:
        """required column to be generated you should be creative and add more features"""
        
        columns = ['statuses_coun','created_at', 'source', 'original_text','polarity','subjectivity', 'lang', 'favorite_count', 'retweet_count', 
            'original_author', 'followers_count','friends_count','possibly_sensitive', 'hashtags', 'user_mentions', 'place']
        
        statuses_count = self.find_statuses_count()
        created_at = self.find_created_time()
        source = self.find_source()
        text = self.find_full_text()
        polarity, subjectivity = self.find_sentiments(text)
        lang = self.find_lang()
        fav_count = self.find_favourite_count()
        retweet_count = self.find_retweet_count()
        screen_name = self.find_screen_name()
        follower_count = self.find_followers_count()
        friends_count = self.find_friends_count()
        sensitivity = self.is_sensitive()
        hashtags = self.find_hashtags()
        mentions = self.find_mentions()
        location = self.find_location()
        data = zip(statuses_count, created_at, source, text, polarity, subjectivity, lang, fav_count, retweet_count, screen_name, follower_count, friends_count, sensitivity, hashtags, mentions, location)
        df = pd.DataFrame(data=data, columns=columns)

        if save:
            df.to_csv(file_name, index=False)
            print('File Successfully Saved.!!!')
        
        return df

                
if __name__ == "__main__":
    # required column to be generated you should be creative and add more features
    #columns = ['created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity', 'lang', 'favorite_count', 'retweet_count', 'original_author', 'screen_count', 'followers_count','friends_count','possibly_sensitive', 'hashtags', 'user_mentions', 'place', 'place_coord_boundaries']
    _, tweet_list = read_json("/global_twitter_data.json")
    tweet = TweetDfExtractor(tweet_list)
    tweet_df = tweet.get_tweet_df() 

    # use all defined functions to generate a dataframe with the specified columns above