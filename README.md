# Sentiment-Analyser

It is Web application for sentiment analysis of tweet. User can directly check the sentiment of the tweet in just one click.

# Working:

User had to follow some of the steps to get proper sentiment from by using this app:
1. Start the app.
2. Write the tweet in the entry box.
3. Click on the Sentiment button.
4. After clicking the model test the tweet and give the output in a popup box.
5. Click on Ok button. Use the same for other tweets.

# Implementation:

import nltk - For text processing.
Remove noise, and filter the meaningful words.
Then make the token of words by using the above imported module.
Use the NaiveBayesClassifier to train the model.
Check the accuracy of the model if not accurate then check some other algorithm.
Now test the model.
Deploy the model.
