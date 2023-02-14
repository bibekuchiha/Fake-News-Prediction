# Fake-News-Prediction
This code is an implementation of a machine learning model that classifies news articles as real or fake. The code is organized into several sections, each with a specific purpose. The following is a documentation of the code:

### Importing necessary dependencies:
The first section of the code imports the required Python libraries for this project. These libraries include NumPy and Pandas for data manipulation, the regular expressions library 're', NLTK (Natural Language Toolkit) for text processing, TfidfVectorizer from Scikit-learn for feature extraction, and Logistic Regression for machine learning.

### Data Preprocessing:
This section is responsible for loading the dataset, counting missing values in the news dataset, and replacing the null values with empty strings. It also merges the title and author columns and separates the data and label.

### Stemming:
Stemming is the process of reducing a word to its root word. In this section, the code defines a stemming function using Porter Stemmer from NLTK. This function takes in text as input and outputs the stemmed version of the text.

### Tfidfvectorizer:
The TfidfVectorizer function from Scikit-learn is used to convert textual data to numerical data. The TfidfVectorizer function transforms text into a matrix of numbers that can be used as input for machine learning models.

### Splitting training and test data:
In this section, the code splits the data into training and testing sets. The data is split into an 80-20 ratio, with 80% used for training and 20% for testing. The stratify parameter is used to ensure that the distribution of the labels in the training and testing data sets is the same.

### Training the model using logistic regression:
The code trains a logistic regression model using the training data. Logistic Regression is a popular classification algorithm used in machine learning.

### Evaluation:
This section evaluates the performance of the model by computing the accuracy score for both the training and testing datasets.

### Making predictions:
Finally, the code makes a prediction using a new test data point and prints whether the news is fake or real based on the prediction.

Overall, this code demonstrates how to preprocess text data, extract features using TfidfVectorizer, train a machine learning model, and evaluate its performance.


### Dataset link:
https://www.kaggle.com/competitions/fake-news/data
