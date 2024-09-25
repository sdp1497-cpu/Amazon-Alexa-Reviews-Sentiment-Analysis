

# 🔍 Amazon Alexa Reviews Sentiment Analysis 📊

This project focuses on performing **Sentiment Analysis** on Amazon Alexa product reviews, helping to uncover insights into customer opinions and sentiments. By analyzing customer feedback, we can gauge overall satisfaction, identify potential areas of improvement, and better understand user behavior towards the Alexa product line.

[Explore the project on Kaggle](https://www.kaggle.com/code/sonia1497/amazon-alexa-reviews-sentiment-analysis)

## 🌟 Project Overview

This **Natural Language Processing (NLP)** project uses sentiment analysis techniques to categorize Amazon Alexa reviews into positive and negative sentiments. By applying machine learning models to the dataset, this project seeks to:
- Classify customer reviews into positive and negative categories.
- Visualize the sentiment distribution across the dataset.
- Provide insights into customer feedback on Amazon Alexa products.

## 🛠️ Tools and Libraries Used

- **Python** 🐍
- **Pandas**: For data loading, cleaning, and manipulation.
- **NumPy**: For numerical operations.
- **NLTK**: For text preprocessing (tokenization, stopword removal, etc.).
- **scikit-learn**: For building and evaluating machine learning models.
- **Matplotlib & Seaborn**: For visualizing sentiment distributions and results.
- **WordCloud**: To visualize the most frequent words in the reviews.

## 📂 Dataset

The dataset used for this project consists of **Amazon Alexa customer reviews**, which includes information such as:
- **Rating**: The overall rating provided by the customer.
- **Review Text**: The written feedback from the customer.
- **Date**: The date the review was posted.
- **Feedback**: Whether the feedback was positive or negative.

The dataset is available on Kaggle, and can be accessed in the project.

## 🚀 Project Workflow

1. **Data Loading & Exploration**:
   - Load the dataset using Pandas and explore its structure and contents.
   - Perform **exploratory data analysis (EDA)** to gain an understanding of the key features and distributions.

2. **Text Preprocessing**:
   - Clean the review text by removing punctuation, special characters, and stopwords.
   - Tokenize and convert text into lowercase.
   - Apply **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization to convert text data into numerical format for model training.

3. **Sentiment Analysis Model**:
   - Build and train several machine learning models, including:
     - **Logistic Regression**
     - **Random Forest**
     - **Naive Bayes**
   - Evaluate the models using accuracy, precision, recall, and F1-score metrics.

4. **Data Visualization**:
   - Generate **WordClouds** to visualize common words in both positive and negative reviews.
   - Plot **sentiment distribution** to understand the overall feedback towards Alexa products.

5. **Model Evaluation & Results**:
   - Evaluate the performance of each model using various metrics to determine which model performs best for sentiment classification.

## 📊 Visualizations

- **Sentiment Distribution**: Shows the overall ratio of positive and negative reviews in the dataset.
- **WordCloud**: Visualizes the most frequently used words in both positive and negative reviews.
- **Confusion Matrix**: Displays the model performance in terms of correctly and incorrectly classified sentiments.

## 🎯 Key Insights

By analyzing the sentiment of Amazon Alexa reviews, this project provides valuable insights, such as:
- The majority of customers have **positive feedback** about Alexa products.
- Common positive keywords include "love," "easy," and "great," while negative feedback often mentions "problem," "stop," and "issue."
- The best-performing model for sentiment classification was found to be **Logistic Regression** with high accuracy and precision.
![image](https://github.com/user-attachments/assets/d3e6eeae-02b0-41cc-aa39-07fcd06d1739)

## 🔧 How to Run the Project

1. **Fork the Kaggle Notebook**:
   Visit the [Kaggle Project Link](https://www.kaggle.com/code/sonia1497/amazon-alexa-reviews-sentiment-analysis) and fork the notebook to run the code.

2. **Explore the Dataset**:
   Access the dataset directly in the Kaggle environment to explore the data and visualizations.

3. **Modify and Experiment**:
   Try different machine learning models or modify the parameters to experiment with the results.

