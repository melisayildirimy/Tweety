import tweepy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib as plt
from datetime import datetime, timedelta
import tkinter as tk

# Twitter API anahtarları
consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"
access_token = "your_access_token"
access_token_secret = "your_access_token_secret"

# Doğrulama işlemi
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# API objesi oluşturma
api = tweepy.API(auth)

# Hashtag listesi
hashtags = ["#hashtag1", "#hashtag2", "#hashtag3"]

# SentimentIntensityAnalyzer nesnesi oluşturma
analyzer = SentimentIntensityAnalyzer()

# MultinomialNB sınıflandırıcısı oluşturma
nb = MultinomialNB()

# CountVectorizer nesnesi oluşturma
vectorizer = CountVectorizer()

# Bugünün tarihini al
today = datetime.today().date()

# Geçen günlerin sayısı, ör. 1 gün önce
days_ago = 1

# Tarihi belirli sayıda gün öncesine ayarla
search_date = today - timedelta(days=days_ago)

# Bugün atılan tweetleri alma ve analiz etme fonksiyonu
def analyze_tweets(hashtag):
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    total_count = 0
    tweets = []
    last_tweet_id = None  # Son çekilen tweet'in ID'si
    
    # Tweetleri alma
    while True:
        # Arama sorgusunu oluştur
        query = f"{hashtag} since:{search_date} -filter:retweets"
        if last_tweet_id is not None:
            query += f" max_id:{last_tweet_id}"
        
        # Tweetleri çek
        search_results = api.search(q=query, lang="en", tweet_mode="extended", count=100)
        
        # Tweet yoksa döngüyü kır
        if len(search_results) == 0:
            break
        
        # Tweetleri analiz et
        for tweet in search_results:
            # Analiz edilecek tweet metnini alma
            tweet_text = tweet.full_text
            
            # Tweet metnini VADER ile analiz etme
            vs = analyzer.polarity_scores(tweet_text)
            
            # Tweet metnini TextBlob ile analiz etme
            tb = TextBlob(tweet_text)
            
            # Tweet metnini Naive Bayes ile analiz etme
            nb_score = nb.predict(vectorizer.transform([tweet_text]))[0]
            
            # Toplam tweet sayısını arttırma
            total_count += 1
            
            # Pozitif, negatif veya nötr olup olmadığını belirleme
            if vs["compound"] > 0.05 and tb.sentiment.polarity > 0 and nb_score == 1:
                sentiment = "positive"
                positive_count += 1
            elif vs["compound"] < -0.05 and tb.sentiment.polarity < 0 and nb_score == 0:
                sentiment = "negative"
                negative_count += 1
            else:
                sentiment = "neutral"
                neutral_count += 1
            
            # Tweet metnini, analiz sonucunu ve sentiment durumunu listeye ekleme
            tweets.append([tweet_text, vs["compound"], tb.sentiment.polarity, nb_score, sentiment])
            
            # Son çekilen tweet'in ID'sini güncelleme
            if last_tweet_id is None or tweet.id < last_tweet_id:
                last_tweet_id = tweet.id
    
    # Tweet sayılarını ekrana yazdırma
    print(f"Hashtag: {hashtag}")
    print(f"Total tweets: {total_count}")
    print(f"Positive tweets: \033[92m{positive_count}\033[0m") # Yeşil renkli yazdırma
    print(f"Negative tweets: \033[91m{negative_count}\033[0m") # Kırmızı renkli yazdırma
    print(f"Neutral tweets: {neutral_count}")
def show_results(hashtag, positive_count, negative_count, neutral_count):
    # Pencere oluşturma
    window = tk.Tk()
    window.title(f"Hashtag Results: {hashtag}")
    window.geometry("400x200")
    
    # Etiketleri oluşturma
    tk.Label(window, text="Hashtag:").grid(row=0, column=0, padx=5, pady=5)
    tk.Label(window, text=hashtag).grid(row=0, column=1, padx=5, pady=5)
    
    tk.Label(window, text="Positive Tweets:").grid(row=1, column=0, padx=5, pady=5)
    tk.Label(window, text=positive_count, fg="green").grid(row=1, column=1, padx=5, pady=5)
    
    tk.Label(window, text="Negative Tweets:").grid(row=2, column=0, padx=5, pady=5)
    tk.Label(window, text=negative_count, fg="red").grid(row=2, column=1, padx=5, pady=5)
    
    tk.Label(window, text="Neutral Tweets:").grid(row=3, column=0, padx=5, pady=5)
    tk.Label(window, text=neutral_count, fg="gray").grid(row=3, column=1, padx=5, pady=5)
    
    # Pencereyi gösterme
    window.mainloop()
if __name__ == '__main__':
    for hashtag in hashtags:
        # Tweet analizi
        tweets_df = analyze_tweets(hashtag)
        
        # Tweetlerin sentiment durumlarına göre renklerin belirlenmesi
        colors = ['red' if sentiment == 'negative' else 'green' if sentiment == 'positive' else 'gray' for sentiment in tweets_df['Sentiment']]
        
        # Sonuçların bar grafiğiyle gösterilmesi
        ax = tweets_df.plot(kind='bar', x='Tweet', y=['Vader Score', 'TextBlob Polarity', 'Naive Bayes Score'], color=colors, title=hashtag)
        ax.set_xlabel("Tweet")
        ax.set_ylabel("Score")
        #plt.show()
        analyze_result = analyze_tweets(hashtag)
        total_count = analyze_result[0]
        positive_count = analyze_result[1]
        negative_count = analyze_result[2]
        neutral_count = analyze_result[3]
        #show_results(hashtag, positive_count, negative_count, neutral_count)
        while True:
            cho = input("Görüntüleme şekli giriniz:\n1)Bar Grafği(Terminal)\n2)Pencere\n3)Çıkış)")
            if cho == 1:
                plt.show()
            elif cho == 2:
                show_results(hashtag, positive_count, negative_count, neutral_count)
            elif cho == 3:
                break
            else:
                print("Geçersiz seçim. Tekrar girmek için enter tuşlayınız") 

