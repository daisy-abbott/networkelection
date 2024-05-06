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


### Training our model

We designed our own neural network that took as input the tweet text and vader score, and output a prediction that was either pro biden or pro trump.  


In order to process the text as input, we used tokenization, a technique that converts each word in the tweet text into a numerical representation, meaning each word is assigned a unique numerical value. We set a maximum number of words to consider as features to 1,000, in order to manage computational complexity effectively. This means that the network only considered these 1,000 unique words when making decisions. This process allowed us to represent the text data in a way that the network could understand. 

```
# Tokenize text
max_words = 1000  # maximum number of words to consider as features
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train['tweet_text'])
sequences = tokenizer.texts_to_sequences(X_train['tweet_text'])
word_index = tokenizer.word_index
maxlen = 280 # max length of a tweet
data_text = pad_sequences(sequences, maxlen=maxlen)
```


In addition to tokenization, we added an embedding layer, which translates the numerical representations of words into dense vectors. This embedding layer is supposed to be able to learn relationships between words based on their contextual usage within the dataset. We then flattened the  output of the embedding layer to transform it into a one-dimensional array.  We did this because in our reading, we found that this step is often necessary when transitioning from convolutional or recurrent layers to fully connected layers in a neural network architecture. 

```
# Define text input
text_input = Input(shape=(maxlen,), name='text_input')
text_embedding = Embedding(max_words, 32, input_length=maxlen)(text_input)
text_flatten = Flatten()(text_embedding)
```

Because we were also using the vader score as input into our network, we concatenated them into a single input layer so that our model could use both of them in its decision making process. This input was then passed through several dense layers, each with RelU activation functions. We also added a dropout layer to prevent overfitting. In addition to a dropout layer, we also implemented early stopping as another form of regularization. 

Finally, we compiled the model using the Adam optimizer and ```binary cross-entropy loss function```, and evaluated our model with accuracy. We hoped that our model would be able to accurately predict whether a given tweet was supportive of Biden or Trump, and provide valuable insights into public sentiment on social media.


## Pre-Processing Larger Dataset

One goal coming out of this project was to visualize a poll on the nation regarding certain topics of contention. To accomplish this, we loaded the pre-tr ained model and decided to run it on a much larger dataset. 

The much larger dataset consisted of a less filtered version of our original dataset. We still preprocessed our data, but instead of filtering by sentiment of hashtag (pro-biden vs pro-trump), we filtered by topic of contention.

<img width="609" alt="topictolabel" src="https://github.com/daisy-abbott/networkelection/assets/112681549/650b50dc-21bc-436e-ae39-265a0d8428f1">



We originally had an issue getting enough data, and started to pivot to using cosine similarity between keywords in our tweets which we will get into later, but ultimately decided to use this mapping, as it gave us enough data. The reason this ended up giving us enough data was because we expanded our dictionary to include capitalization, hyphens, and more related keywords.

We then loaded  the tweets in each file by chunks, and filtered the data so that we only kept the tweets if they contained one of these words. If a tweet contained a word in the dictionary, we assigned the related topic as a label to this tweet. 

Next, we ran vader on the tweet texts to obtain a sentiment score for each tweet. We then merged the biden and trump dataframes into one dataframe and ended up with the following amounts of data per topic per candidate: 

<img width="359" alt="testdata" src="https://github.com/daisy-abbott/networkelection/assets/112681549/427b46dd-caa3-4b4e-ad13-b9928817be7c">

As you can see, we have ```25,000``` tweets from the ```Biden``` file and ```15,000``` tweets from the ```trump``` file. ```21,000``` tweets related to ```Gun Control``` (13,000 from the biden file, 7,000 from the trump file), ```16,000``` tweets related to ```immigration``` (10,000 from biden and 6,000 from trump), and ```4,000``` tweets related to abortion (2400 from biden and 1400 from trump). 

We then ran our pre-trained model on this larger cleaned dataset to give us a visualization on a poll of the nations viewpoints through tweet analysis. 


### A note on cosine similarity

At the very beginning of the data preprocessing, we were filtering our data by both topic and hashtag sentiment. Our topic dictionary was a lot smaller, it didn’t include edge cases of capitalization and other keywords, and we didn’t have enough data so we were overfitting our model. In order to avoid this, we decided to change how we were filtering by topic and instead of using a dictionary containing topics, we decided to do word embedding on our tweets and keywords, and find the cosine similarity between them.  

We used the ```word2vec-google-news-300``` which was listed as one of the top five embedding models and was trained on news related text, so we figured that this would be perfect for our purposes. 


<img width="346" alt="describe" src="https://github.com/daisy-abbott/networkelection/assets/112681549/4ef7a4bc-74c6-44d7-be24-eba993803d48">

Unfortunately, as you can see from the describe(), there is almost no similarity at all between the vectors. Because this did not prove to be useful, we changed our direction to what we talked about earlier: Training a model on filtered text by hashtag and sentiment score to predict a candidate, and then loading the pretrained model, and feeding the filtering the data solely by topic, and feeding that into the model. 

We did further investigation as to why the cosine similarity scores were so low, and realized that the function was taking the average cosine similarity of every word in the tweet, instead of grabbing out the important ones. 

We were able to modify our code to loop through each word in the tweet, take the word with the highest cosine similarity, and save that as the score for the entire tweet. This did work and we got incredibly high scores that were relevant and fit with our data. 

<img width="335" alt="cosine_sim" src="https://github.com/daisy-abbott/networkelection/assets/112681549/0de07939-5a7a-4808-be20-92f11b9324a7">

In the future, we’d like to explore how the model would perform, but since at this point, we already had enough data, we decided to proceed with our original plan. 


# Results: 
* Training and Validation Accuracy
<img width="608" alt="trainingAccuracy" src="https://github.com/daisy-abbott/networkelection/assets/112681549/5e5bd026-bfb6-4567-9be2-0dfd7099a26d">

* Training and Validation Loss
<img width="1382" alt="trainingLoss" src="https://github.com/daisy-abbott/networkelection/assets/112681549/e28cff66-4248-4a61-9415-25ade3e47d7c">

* Classification Report
![classificationreport](https://github.com/daisy-abbott/networkelection/assets/112681549/0dcf66d6-2fd8-45f9-8856-6ea18a52d1ee)



The classification report demonstrates that the model is predicting both candidates very accurately, with 0 being Biden and 1 being Trump. However, the f1-score for classifying pro-Biden tweets is lower than pro-Trump tweets (especially when looking at recall). This is most likely because two thirds of the training data was pro-Trump tweets, the other third being pro-Biden. This bias is reflected in the model, making it more likely to predict a tweet as pro-Trump. 

* Confusion Matrix
![confusionmatrix](https://github.com/daisy-abbott/networkelection/assets/112681549/c4f26238-d2e2-432f-9fac-eb6d482a015c)

* Distribution of predictions on abortion
<img width="250" alt="dist_abortion" src="https://github.com/daisy-abbott/networkelection/assets/112681549/7369e2db-3524-46b9-b312-473e74a21df7">

* Distribution of predictions on immigration
<img width="257" alt="dist_immigration" src="https://github.com/daisy-abbott/networkelection/assets/112681549/5f55af6e-8803-42c9-8e09-210635bb6bdc">

* Distribution of predictions on gun control
<img width="293" alt="dist_guncontrol" src="https://github.com/daisy-abbott/networkelection/assets/112681549/31ef0899-a93c-4717-81eb-ab6e149ff926">

According to this confusion matrix, the model is making the most incorrect predictions when it is actually pro-Biden, but it confuses it as pro-Trump. Like we mentioned before, this is because of the imbalance in the training data, which contained more pro-Trump data. 
