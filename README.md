# 2020 Election Poll and Analysis

## Introduction 

The purpose of our project is to better understand how the public feels about key issues in regards to Trump and Biden using tweets regarding the 2020 election. The model we designed could be used to measure support for either Trump and Biden in the upcoming election, giving insight into how different demographics of Americans feel about certain issues.

 Using a kaggle dataset from the 2020 election, we will measure public sentiment for three of the most divisive issues in the US today: abortion, immigration, and gun control. We will visualize a percentage of the tweets that support Trump and Biden for each of the respective topics. 


In typical polls, there is a large degree of self-selection bias, leading to inaccurate polls that fail to represent the true feelings of the American public. So if we are able to accurately predict which candidate a tweet supports, we can then measure public sentiment on a much larger scale, which could potentially be more representative of public opinion than traditional polls. 

*  <img width="299" alt="pic_trump_biden" src="https://github.com/daisy-abbott/networkelection/assets/112681549/76010965-c480-446c-b11e-b822fd5d6c95">


There has also been a surprising lack of social media sentiment analysis to help predict and analyze elections. Additionally, there are no large datasets with labeled data on political sentiment, so our model could ideally provide accurate labeling for large datasets. 

### Problem Statement:

* Traditional polls have shown to be inadequate, so are we able to gain more accurate insights into public political sentiment on a larger scale through analysis of Twitter data? 


## Methodology and Data
### Resources Used 
* Dataset: https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets
* Word2Vec Model: word2vec-google-news-300 from the ```gensim.downloader```
* ```Vader``` (an NLTK module that provides sentiment scores based on the words used)

### Pre-Processsing the Training Dataset
The kaggle dataset that we worked with consisted of two files, one containing Trump related tweets, and the other containing Biden related tweets.  They are not pro one candidate or the other. We began by loading the files and preprocessing the data. Because we wanted our model to predict whether the sentiment of the  tweet was pro-Trump or pro-biden, we first had to label our data as pro trump or pro biden. We did this by creating two data frames, and filtering the data by hashtags that are representative of pro-Trump or pro-Biden sentiment. The filtering definition is  as follows: 

```
# Define hashtags and topics
trump_hashtags = ["#MAGA", "#KAG", "#FourMoreYears", "#SleepyJoe", "#BlacksForTrump", "#Trump2020", "#VoteRed", "#WomenForTrump", "#LatinosForTrump", "#AmericaFirst", "#BuildTheWall"]
biden_hashtags = ["#Biden2020", "#BidenHarris2020", "#VoteBlue", "#NotMyPresident", "#BlueWave2020", "#VoteBiden", "#VoteBlueToSaveAmerica", "#BlacksForBiden", "#WomenForBiden", "#LatinosForBiden"]
```

We looked at the trump file first, and if a tweet in this file contained a hashtag from the pro trump hashtags list, we added it to the pro trump dataframe.  We repeated the process for biden. We then added a label column with the candidate corresponding to the dataframe. 

Once we had the two data frames, we ran the tweets from each dataframe through Vader (a rule-based sentiment analysis engine) to analyze the sentiment of the tweets and obtain a sentiment score. Next, we merged the dataframes into one, and dropped all duplicates.  However, before training the model, we removed all of the pro-trump and pro-biden hashtags from the tweets so that the model would not only learn to classify the tweets based on those hashtags, making it better able to generalize the larger dataset. From here, we were able to feed this data into our model. 



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

We will discuss this further in our results section, but because our model was predicting way too accurately, we decided to try one more approach. We filtered the data both by topic and by sentiment (pro biden or pro trump) and instead of feeding the tweet text and vader score into our model we fed the topic and the vader score into our model and got the prediction of the candidate. This unfortunately didn’t prove to be successful. We tried many different learning rates as well as different types of regularization(early stopping and dropout), but the model did not learn very successfully. This is likely due to the fact that although Vader is great at finding the overall sentiment of a text, it doesn’t find the sentiment of one candidate or the other. 

## Pre-Processing Larger Dataset

One goal coming out of this project was to visualize a poll on the nation regarding certain topics of contention. To accomplish this, we loaded our trained model and decided to run it on a much larger dataset. 

The much larger dataset consisted of a less filtered version of our original dataset. We still preprocessed our data, but instead of filtering by sentiment of hashtag (pro-biden vs pro-trump), we filtered by topic of contention. 

<img width="609" alt="topictolabel" src="https://github.com/daisy-abbott/networkelection/assets/112681549/650b50dc-21bc-436e-ae39-265a0d8428f1">



We originally had an issue getting enough data, and started to pivot to using cosine similarity between keywords in our tweets which we will get into later, but ultimately decided to use this mapping, as it gave us enough data. The reason this ended up giving us enough data was because we expanded our dictionary to include capitalization, hyphens, and more related keywords.

We then loaded  the tweets in each file by chunks, and filtered the data so that we only kept the tweets if they contained one of these words. If a tweet contained a word in the dictionary, we assigned the related topic as a label to this tweet. 

Next, we ran vader on the tweet texts to obtain a sentiment score for each tweet. We then merged the biden and trump dataframes into one dataframe and ended up with the following amounts of data per topic per candidate: 

<img width="359" alt="testdata" src="https://github.com/daisy-abbott/networkelection/assets/112681549/427b46dd-caa3-4b4e-ad13-b9928817be7c">

As you can see, we have ```25,000``` tweets from the ```Biden``` file and ```15,000``` tweets from the ```trump``` file. ```21,000``` tweets related to ```Gun Control``` (13,000 from the biden file, 7,000 from the trump file), ```16,000``` tweets related to ```immigration``` (10,000 from biden and 6,000 from trump), and ```4,000``` tweets related to abortion (2400 from biden and 1400 from trump). 

We then ran our trained model on this larger cleaned dataset to give us a visualization on a poll of the nations viewpoints through tweet analysis. 


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
* Figure 1: Training and Validation Accuracy
<img width="608" alt="trainingAccuracy" src="https://github.com/daisy-abbott/networkelection/assets/112681549/5e5bd026-bfb6-4567-9be2-0dfd7099a26d">

* Figure 2: Training and Validation Loss
<img width="608" alt="trainingLoss" src="https://github.com/daisy-abbott/networkelection/assets/112681549/e28cff66-4248-4a61-9415-25ade3e47d7c">

* Figure 3: Classification Report
![classificationreport](https://github.com/daisy-abbott/networkelection/assets/112681549/0dcf66d6-2fd8-45f9-8856-6ea18a52d1ee)
The classification report shows the models performance by evaluating precision, recall, f1 score, and accuracy. 




* Figure 4: Confusion Matrix
![confusionmatrix](https://github.com/daisy-abbott/networkelection/assets/112681549/c4f26238-d2e2-432f-9fac-eb6d482a015c)
This confusion matrix shows the actual and predicted values for Trump and Biden. 




* Figur 5: 
Distribution of predictions on abortion
<img width="250" alt="dist_abortion" src="https://github.com/daisy-abbott/networkelection/assets/112681549/7369e2db-3524-46b9-b312-473e74a21df7">
Distribution of predictions on immigration
<img width="257" alt="dist_immigration" src="https://github.com/daisy-abbott/networkelection/assets/112681549/5f55af6e-8803-42c9-8e09-210635bb6bdc">
Distribution of predictions on gun control
<img width="293" alt="dist_guncontrol" src="https://github.com/daisy-abbott/networkelection/assets/112681549/31ef0899-a93c-4717-81eb-ab6e149ff926">
After running our model on the larger dataset, inputting the tweet text and vader score to predict the candidate, the model gave us predictions that are difficult to interpret. As you can see from the distribution of prediction scores, they only range from 0.24 to 0.45 for all three topics, seemingly only predicting all of the tweets as closer to Biden. 


 * Figure 6: Percentage of Tweets in Support of Biden/Trump
<img width="666" alt="abortion" src="https://github.com/daisy-abbott/networkelection/assets/112681549/2748f580-51e7-4ee8-9252-f8dd16e264e0">
<img width="666" alt="guncontrol" src="https://github.com/daisy-abbott/networkelection/assets/112681549/2ff06bbd-3e8b-4363-a5cd-3d549b43838c">
<img width="657" alt="immigration" src="https://github.com/daisy-abbott/networkelection/assets/112681549/1eb80360-341c-4388-ba2d-8bfcb6d2800d">


* Figure 7: Topic and Vader Score
<img width="605" alt="vadertopicacc" src="https://github.com/daisy-abbott/networkelection/assets/112681549/715149aa-4198-4905-a746-abba9312d8f1">
These are the results from an additional model that we attempted to train using only vader score and the topic as features to predict the candidate. 


# Discussion

## Figures 1 & 2 visualize the model’s performance on our dataset.

As you can see from the Accuracy and Loss, our model performs extremely well, too well. We originally thought that our model was overfitting, so we added in a dropout layer and early stopping as forms of regularization, but if you look at the validation accuracy, it is equally high and does not decrease, indicating no signs of overfitting. Our hypothesis is that because of the way we embedded and tokenized our text, it’s possible that our model learned the numerical values for the words biden and trump as well as any other associations, even though we removed the explicit hashtags that were indicative of the tweet’s corresponding label. The issue may also lie in the way we preprocessed the data. All of our pro-Trump tweets come from the trump csv file, where each tweet contains a reference towards Trump. We did not scrape the Biden csv file for pro-Trump tweets. This means that the model may just be recognizing any reference towards Trump and learning that as pro-Trump sentiment because all our pro-Trump labeled tweets have a direct reference to Trump. This issue is reflected later in our predictions, which generally does not separate pro-Trump and pro-Biden tweets very accurately. If we were to do this project again, it would have been better to remove #Trump from all pro-Trump tweets, and the same for Biden. We should have filtered out the pro-Trump tweets from the biden csv file as well, so that some of the pro-Trump tweets have references to Biden too (and vice versa). However, it would have been difficult to label pro-Trump sentiment in a file that contains only references to biden. 

## Figure 3: Classification Report 
The classification report demonstrates that the model is predicting both candidates very accurately, with 0 being Biden and 1 being Trump. However, the f1-score for classifying pro-Biden tweets is lower than pro-Trump tweets (especially when looking at recall). This is most likely due to the data imbalance as  two thirds of the training data was pro-Trump tweets, the other third being pro-Biden. This bias is reflected in the model, making it more likely to predict a tweet as pro-Trump.  

## Figure 4: Confusion Matrix 
According to this confusion matrix, the model is performing very well but is making the most incorrect predictions when the true label is pro-Biden, but the model confuses it as pro-Trump. Like we mentioned before, this is likely because of the imbalance in the training data, which contained more pro-Trump tweets. 

## Figure 5: Distribution on Gun Control, Abortion, and Immigration
We are not completely sure as to why the predictions came out this way, so we looked at the tweets with the highest and lowest prediction scores to see if it was able to separate pro-Biden and pro-Trump tweets. When looking at the top 20 highest prediction scores for abortion-related tweets, 17 out of 20 were pro-Trump. When looking at the lowest 20 prediction scores, 19 out of 20 were pro-Biden. Despite the strange output of the model, it was still able to somewhat accurately classify these abortion-related tweets. 

However, when looking at the immigration and gun control tweets, the results become more puzzling. For both of these topics, the model was not able to accurately separate the pro Trump or Biden sentiment when looking at the highest and lowest 20 prediction scores. We discuss our hypothesis for why this might be the case later in the discussion section.

To create visualizations for this data, we decided to make the median of the prediction scores the cutoff for either pro-Trump or pro-Biden sentiment. Any tweet above the median would be classified as pro-Trump and everything below would be pro-Biden. We then plotted this as the percentage of tweets supporting each candidate related to one of the three topics.

## Figure 6: Percentage of Tweets in support of Candidate on pro biden / pro trump 
As you can see from the graphs, there is much more pro-Biden sentiment in the tweets related to these topics. The greatest disparity is in gun control sentiment, with 75% of the tweets supporting Biden. The pro-Biden sentiment across the key issues may reflect the results of the 2020 election, where Biden won a majority of the popular vote. However, these results are most likely not accurate for immigration and gun control, as the model did not give accurate prediction scores.

## Figure 7: Model Vader and Topic Accuracy
We attempted to try another angle where we trained the model on the topic and the vader score rather than the tweet text and the vader score in the hopes that we would avoid the insanely correct model. However, when training this model, we only had around 3,000 tweets in total for all of the topics. Due to the lack of data and the fact that vader score was not very indicative of tweet sentiment, the accuracy for this model was poor and was not able to learn without the tweet itself (did not improve past 54% accuracy). 

## Issues with our model / data:
* Our model gives out a binary prediction of either support for Trump or Biden, so it will classify tweets that are neutral as support for one candidate or the other. In theory, the model should randomly assign these tweets between Trump and Biden. However, since the model is trained on more pro Trump tweets than pro Biden tweets, there may be a bias towards classifying these neutral tweets as pro Trump more often. 

* There are also more Trump related tweets than Biden related tweets, and even though these tweets are not necessarily in favor of one candidate or the other, there was a skew towards more pro Trump predictions in general, as shown in the confusion matrix. 

* Our labeling is not 100% accurate, because some of the hashtags may not completely represent pro-Biden or pro-Trump sentiment. For example a tweet that says “Trump is a liar #Trump2020” would be classified as pro-Trump.
We initially wanted to use the topics as an extra feature to train the model on, but we had issues filtering out the topics from the tweets, so we decided to train the model without including the topic as a feature. We tried using a pre-trained word2vec model to create word embeddings and then labeling the topic of tweets based on the tweet’s cosine similarity to specific keywords, but the cosine similarity was very low. We believe that the single keyword did not map well to an entire tweet. We eventually fixed this issue, but decided to move forward with our original plan.

* We would have liked to compare the sentiment towards the key issues from the 2020 election to new data regarding the upcoming 2024 election by scraping tweets from Twitter, but given the time constraints for this project, we were not able to do this. 

* Since vader is a strict rule-based model, its sentiment scores were not able to pick up on the complexity of politically charged tweets very well, so adding the scores as an extra feature may not have been very helpful to the learning of the model. This is reflected in our results when trying to predict the candidate with just the vader score and topic, as the model did not get better than 50% accuracy. 



# Conclusion

Our project aimed to explore sentiment analysis on Twitter data regarding the 2020 US election. We attempted to address the shortcomings of traditional polling methods by leveraging machine learning techniques. While our model demonstrated high accuracy during training, it faced challenges in accurately predicting sentiment on a larger scale, particularly regarding immigration and gun control when run on the entire dataset.


Although we preprocessed the data to the best of our abilities, we still had imbalances in the training data which lead to the model having a bias towards predicting trump. Additionally, because of the binary predictions combined with the complexity of politically charged tweets, the model struggled to generalize accurately to the entire dataset. We believe this is due to the fact that we didn’t fully preprocess our data by including pro biden/pro trump sentiment from both csv files. Furthermore, the use of Vader sentiment analysis, while informative, lacked the nuance required for comprehensive political sentiment analysis. 


Overall, while our project touches on the potential of machine learning in social media for political sentiment analysis, there definitely needs to be further refinement and exploration to accurately capture the complexities of public opinion in the digital age.


## References: 

* https://python.plainenglish.io/how-to-extract-tweets-on-a-hashtag-from-twitter-api-using-tweepy-34d697ecba21
https://www.pewresearch.org/short-reads/2022/09/21/does-public-opinion-polling-about-issues-still-work/
https://www.theguardian.com/us-news/2024/mar/05/2024-election-campaign-issue-tracker


* https://www.sciencedirect.com/science/article/pii/S1877050920306669

* https://github.com/mgierlach/nlp-twitter-political-sentiment-analysis

* https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00633-z

* https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets

* https://python.plainenglish.io/how-to-extract-tweets-on-a-hashtag-from-twitter-api-using-tweepy-34d697ecba21

* https://www.diva-portal.org/smash/get/diva2:1335995/FULLTEXT02#:~:text=CNN%2DLSTM%20model%20is%20best,respect%20to%20the%20selected%20dataset.

* https://www.kaggle.com/datasets/kazanova/sentiment140

* https://www.kaggle.com/code/varun08/sentiment-analysis-using-word2vec

* https://www.semanticscholar.org/paper/Analysis-on-Audio-and-Video-using-Vader-Algorithm-Yadav-Raskar/f3dbdee338de75106e2a765766c92f8563920f84

* https://medium.com/semi-supervised-sentiment-analysis-and-language/plenty-of-advancements-happened-in-language-model-at-the-end-of-2018-with-the-arrival-of-cd0c5dc7ca57



