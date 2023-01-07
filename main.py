
# System: ASX Sentiment Analysis

# Author: Vidit Patel
# Last edited: 07/01/2022

#import the library required

from urllib.request import urlopen, Request #used for getting req and res
from bs4 import BeautifulSoup #need this for web scrapping
from nltk.sentiment.vader import SentimentIntensityAnalyzer #need this for sentiment analysis
import pandas as pd #provides useful data structures 
import matplotlib.pyplot as plt #used for presenting the results, visualising data 

#SECTION 1: Collecting the Data
#-------------------------------

#General URL from FinViz

general_finviz_url = 'https://finviz.com/quote.ashx?t='

#list of stocks to check

stocks = ['AMZN', 'META', 'TSLA', 'GOOG']

#Generate the url for each of the stocks above

news_tables = {}

for stock in stocks:
    
    new_url = general_finviz_url + stock

    #Request HTML data 
    req = Request(url=new_url,  headers={'user-agent' : 'my-app'})
    response = urlopen(req)
    
    #Scrap the html code for this stock 
    html_code = BeautifulSoup(response, 'html.parser') #get BeatifulSoup to parse this html website for current stock 

    #parse the html to get the id of the news table (id = 'news-table') for this stock

    link_for_the_news_table = html_code.find(id='news-table');
    news_tables[stock] = link_for_the_news_table;


#SECTION 2: Manipulating and formatting the data 
#------------------------------------------------

#goal is to get the timestamp, article text, and link

#the title is stored in an a-tag, with class "tab-link-news"

#need to store stock_ticker, date/time, title of article

parsed_data = []

for stock, news_tables in news_tables.items():

    # go through all articles for the current stock

    for row in news_tables.findAll('tr'):

        title = row.a.get_text() #grabs the title of the article

        date_data = row.td.text.split(' ') #check for date-time or just time

        if len(date_data) == 1:

            time = date_data[0]

        else:
            date = date_data[0]
            time = date_data[1]

        parsed_data.append([stock, date, time, title])

#SECTION 3: Apply Sentiment Analysis
#------------------------------------------------

# Background information on NLTK.Sentiment.Vader - It is model used for text sentiment analysis

# This model applies polarity scores on the input text and produces positive, negative, neutral scores (range is from -1 to 1)

# Negative, Nuetral, Positive describe the scores which fall into each category

# This can be used to check if the article will positevly or negatively impact the company reputation

panda_data_frame = pd.DataFrame(parsed_data, columns=['stock','date', 'time', 'title']) #better format for holding the data using pandas 

#initalise the sentiment analyser

vader = SentimentIntensityAnalyzer();

#apply polarity scores on the titles 

get_score = lambda title: vader.polarity_scores(title)['compound'] #this function is responsible for just getting back the compound score and no other score from the vader.polarity_scores() method 

panda_data_frame['compound'] = panda_data_frame['title'].apply(get_score)

#SECTION 4: Visualising the data 
#------------------------------------------------

#convert the dates into a more appropriate format

panda_data_frame['date'] = pd.to_datetime(panda_data_frame.date).dt.date

plt.figure(figsize=(10,8))

#get the average of all compound scores of a company to check if today was a positive or negative day for a company 

mean_compound = panda_data_frame.groupby(['stock', 'date']).mean()  #This will only look for integer values, which in this case, is the compound scores 

#grab the key value pairs 
mean_compound = mean_compound.unstack()
mean_compound = mean_compound.xs('compound', axis = "columns").transpose()

#plot the data
mean_compound.plot(kind='bar')
plt.show()


