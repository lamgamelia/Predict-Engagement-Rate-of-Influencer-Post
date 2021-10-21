# Predict Engagement Rate of Influencer Post

The dataset (Source: https://sites.google.com/site/sbkimcv/dataset) consists of over 10 million posts from 33,935 instagram influencers saved in separate JSON files which contain information: caption, usertags, hashtags, likes, comments, etc.

The users have been classified into nine categories: beauty, family, fashion, fitness, food, interior, pet, travel and other.

A mapping file is provided to map the users to the posts (i.e. name of JSON file)

## Project objective

This project aims to explore NLP and other features available from influencer posts to predict engagement rate of influencer posts. This will provide a systematic method for businesses to form its influencer marketing strategy.

## Methods Used

* Data Munging
* EDA
* Feature Engineering
* Natural Language Processing
* Multi-class classification
* Dimensionality reduction
* Predictive modelling using regression

## Technologies

* Python
* pandas, jupyter
* sklearn
* NLP libraries: spacy/nltk/emot

## Project description

1. Preprocessing

    Dataset **influencer_data** and **df_filenames** have been converted from .txt source files into pandas dataframe\
    **df_filenames** contains file names of each posts in each row. 10 post samples are chosen from each influencer for the purpose of this project, using function *sample*\
    Using function *run_content* details of each sampled post is populated in 'contents' column of dataframe **dfs10**\
    NA values are dropped and dataframe is then exported as .csv file for next-step preprocessing.

    Extracting features from 'contents' of JSON files to be added to **dfs10** dataframe

    Below functions are created:
    * *comment_count* : extract no. of comments for a post
    * *comment* : extract all the comments on related post in text
    * *count_tags* : extract the no. of other users tagged in a post
    * *verified_tags* : extract no. of other verified users tagged in a post
    * *likes* : extract no. of likes on main post
    * *caption* : extract the caption of each post, i.e. text form which will be used for NLP 
    
    Additional functions/steps taken to fine-tuned the captions:
    * clean text using defined function *clean_re*
    * detect language of post using function *language* which will return 'en' for english posts, non-english posts are removed from dataset
    * stopwords are removed from captions and then lemmatised and tokenised, stored in column 'tokens'

2. Feature engineering

    a. 'en_score' - measures the score 0-1 of the use of english in each caption derived using spacy's LanguageDetector
    b. 'comments' - emojis in comments are retained and converted to text for sentiment analysis
    c. 'comment_counts'
    d. 'likes'
    e. 'count_tags'
    f. 'verified_tags'
    g. 'tokens' : as text for classification and scalar for regression model
    h. 'no_hashtag' - consists of caption text without hashtags
    i. 'hashtag' - consists of only hashtags
    j. 'sentiment' - score from -1 to 1 using nltk's SentimentIntensityAnalyzer
    k. 'Category' - being area of interest of influencer obtained from **influencer_data**
    l. Series of NER Tag counts - counts of no of ["ORG","PERSON","NORP","GPE","LOC","FAC","EVENT","PRODUCT","WORK_OF_ART"] tags based on spacy Named Entity Recognition in a post
    m. 'tier' - specifies influencers as ["Micro","Nano","Mid-tier","Macro","Mega"] based on no of followers
    n. 'engagement_rate' - calculated based on ('comment_counts' + 'likes')/no of followers of each influencer
    o. 'count_image' - counts no of image in a post
    p. 'wordcount' - counts of no of words in 'tokens'

3. Exploratory Data Analysis

    a. Using panda's barplot notice that there are imbalance in the distribution of users across category of interest. Fashion is the top common interest across users. Top 5 categories will be used in this project
    b. Using panda's barplot notice that there are disproportionately lower number of Mega and Macro influencers with no of followers > 500k
    c. Using seaborn's heatmap, illustrate the correlation of features
    d. Using seaborn's histplot, illustrate the distribution of posts across different engagement rate. We note a left skewed distribution as most posts has relatively low engagement rate with a few with very high rate
    e. Using boxplot, illustrate  the distribution of engagement rates across different influencer tier based on no of followers
    f. Using wordcloud, visualise the most common words used for each category of interest

4. Multi-class Classification model based on posted captions

    'tokens' text are used to train a supervised classification model to categorise posts into 5 main categories, i.e. fashion, travel, family, food and beauty. 
    For each models trained tokens are converted into count vectors, word level TF-IDF and N-Grame vectors:
    - Naive Bayes
    - SVM
    - Random Forest
    - XGBoost
    Hyper-parameter tuning using GridSearch and RandomizedSearchCV method is performed.

    Evaluation of the model is done using the precision, recall, f1-score, confusion matrix and also auc-roc curve.

    Visualisation of the clusters is also done using gensim's Word2Vec vocabulary and using TSNE to reduce dimensionality for a 2 dimensional plot. This allows us to visualise the similarity and exclusivity of words used in caption in relation to the area of interest. Well separated clusters pre-liminarily suggest the relevance of caption text in determining area of interests of the influencer.

5. Regression model for prediction of engagement rate

    Some features are added and derived from the trained classification model above, i.e. .predict_proba. 
    - 'cat_prob'  represents the probability that the post is in the same area of interest as the user\
        This feature represents the extent to which a post is related to the perceived main area of interest of the influencer. Posts that are outside the main area of interest may lead to lower engagement rate.
    - Probabilities of the posts falling under each categories: 'beauty', 'family', 'fashion', 'food' and 'travel'

    With the relevant features, below regression models are trained. In consideration of the skewness of the features, box-cox transformation has also been performed on the dataset. Min-max scaling has also been done on separately in consideration of extreme difference in ranges of the numerical features.
    - Ridge, Lasso Regression
    - SVR Linear and  Rbf kernel
    - Random Forest
    - MLPRegressor (i.e. ANN)

    Evaluating scores usd are MAE, MSE, RMSe and R2 scores.

Conclusion:\
The regression model give rise to low error but also low R2 scores. Further optimisation of features selected and extension of the samples used for training can be considered to improve the model.

Please refer to attached powerpoint slides and word documentation for further business applications and references used.