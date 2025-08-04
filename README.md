## 🛡️ SPAM_EMAIL DETECTOR ANALYSIS

## 📌 Project Overview
This capstone project, SpamDetector, is an end-to-end solution for detecting and visualizing spam emails. The project involves a machine learning model, data analysis with Python, and an interactive dashboard built with Power BI to provide a comprehensive understanding of what makes an email spam.

The core goal was to build a predictive model capable of accurately classifying emails as either "spam" or "not spam." This was achieved using the **spambase** dataset, a classic dataset in machine learning for email classification.

### 🔍 Dataset Information

The data for this project comes from the UCI Machine Learning Repository. The spambase dataset is a collection of 4,601 emails, with 57 continuous features that describe various characteristics of each email.

##### The features are categorized into three main groups:

  **Word Frequency:** 48 features representing the percentage of words in the email that match a specific key word (e.g., word_freq_free, word_freq_money).

  **Character Frequency:** 6 features representing the percentage of characters in the email that match a specific punctuation mark (e.g., char_freq_exclamation, char_freq_dollar).

  **Capital Run Length:** 3 features measuring the length of uninterrupted sequences of capital letters (capital_run_length_longest, capital_run_length_average, capital_run_length_total).

The final column, is_spam, is the target variable that classifies the email as either 1 (spam) or 0 (not spam).

### 🔬 Data Analysis

After cleaning and processing the data, a thorough analysis was conducted to uncover patterns that differentiate spam from legitimate emails. Key insights from this analysis include:

**1.High-Value Spam Words:** Words such as free, business, credit, and money showed significantly higher average frequencies in spam emails compared to non-spam.

**2.Punctuation as an Indicator:** Spam emails frequently used punctuation marks like $ and ! with a much higher frequency than legitimate emails.

**3.Capitalization Trends:** Spam emails contained, on average, much longer sequences of capitalized letters (capital_run_length_longest), likely used to grab a user's attention.

These insights were crucial for understanding the data and informed the training of the machine learning model.

### 🚀 The Solution

The project is built on three key components:

**1.Data Preprocessing and Modeling:** A Python script was used to clean the raw data, train a Logistic Regression model, and save the model for use in a web application.

**2.Web Application:** A Flask application provides a user-friendly interface for real-time spam detection. Users can input a message and receive a classification instantly.

<img width="886" height="767" alt="UI_spam email" src="https://github.com/user-attachments/assets/1d4850a9-26bc-4cb9-bd85-96dd98cf4461" />


**3.Data Visualization:** An interactive Power BI dashboard was created to visualize key insights from the data, making the model's findings easy to understand.

### 📊 Power BI Dashboard
The interactive Power BI dashboard provides a visual summary of the data and model findings. It includes several visualizations, such as:

**A Doughnut Chart** showing the distribution of spam vs. non-spam emails.

**Bar Charts** visualizing the average frequency of key words like free, money, and business in spam emails.

**Scatter Plots and Slicers** for interactive exploration of relationships between different features.

A custom "Spam Rate" **DAX measure** to provide an at-a-glance view of the overall spam percentage.

<img width="1048" height="547" alt="spam dashboard" src="https://github.com/user-attachments/assets/e5f95085-7b8c-4ce7-997f-20a1446f5fa5" />




