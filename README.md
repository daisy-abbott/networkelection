# 2020 Election Poll and Analysis

## Introduction 

The purpose of our project is to better understand how the public feels about key issues in regards to Trump and Biden using tweets regarding the 2020 election. The model we designed could be used to measure support for either Trump and Biden in the upcoming election, giving insight into how different demographics of Americans feel about certain issues.

 Using a kaggle dataset from the 2020 election, we will measure public sentiment for three of the most divisive issues in the US today: abortion, immigration, and gun control. We will visualize a percentage of the tweets that support Trump and Biden for each of the respective topics. 


In typical polls, there is a large degree of self-selection bias, leading to inaccurate polls that fail to represent the true feelings of the American public. So if we are able to accurately predict which candidate a tweet supports, we can then measure public sentiment on a much larger scale, which could potentially be more representative of public opinion than traditional polls. 


There has also been a surprising lack of social media sentiment analysis to help predict and analyze elections. Additionally, there are no large datasets with labeled data on political sentiment, so our model could ideally provide accurate labeling for large datasets. 

### Problem Statement:

* Traditional polls have shown to be inadequate, so are we able to gain more accurate insights into public political sentiment on a larger scale through analysis of Twitter data? 


## Methodology and Data
### Resources Used 
* Dataset: https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets
* Word2Vec Model: word2vec-google-news-300 from the ```gensim.downloader```
* ```Vader``` (an NLTK module that provides sentiment scores based on the words used)

### Pre-Processsing the Training Dataset
The kaggle dataset that we worked with consisted of two files, one containing trump related tweets, and the other containing biden related tweets.  They are not pro one candidate or the other. We began by loading the files and preprocessing the data. Because we wanted our model to predict whether the sentiment of the  tweet was pro trump or pro biden, we first had to label our data as pro trump or pro biden. We did this by creating two data frames, and filtering the data by pro trump or pro biden hashtags. The filtering definition is  as follows: 

```
# Define hashtags and topics
trump_hashtags = ["#MAGA", "#KAG", "#FourMoreYears", "#SleepyJoe", "#BlacksForTrump", "#Trump2020", "#VoteRed", "#WomenForTrump", "#LatinosForTrump", "#AmericaFirst", "#BuildTheWall"]
biden_hashtags = ["#Biden2020", "#BidenHarris2020", "#VoteBlue", "#NotMyPresident", "#BlueWave2020", "#VoteBiden", "#VoteBlueToSaveAmerica", "#BlacksForBiden", "#WomenForBiden", "#LatinosForBiden"]
```

We looked at the trump file first, and if a tweet in this file contained a hashtag from the pro trump hashtags list, we added it to the pro trump dataframe.  We repeated the process for biden. We then added a label column with the candidate corresponding to the dataframe. 

Once we had the two data frames, we ran the tweets from each dataframe through Vader to analyze the sentiment of the tweets and obtain a sentiment score. Next, we merged the dataframes into one, and dropped all duplicates.  However, before training the model, we removed all of the pro-trump and pro-biden hashtags from the tweets so that the model would not only learn to classify the tweets based on those hashtags, making it better able to generalize the larger dataset. From here, we were able to feed this data into our model. 


Our final training dataset consisted of:
* 204182 pro trump tweets
* 85406 pro biden tweets

