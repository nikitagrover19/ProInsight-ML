🧠 ProInsight - Email Classifier

This project leverages Machine Learning, NLP, and the Gemini API to analyze and derive insights from corporate email communications using the Enron Email Dataset
.
It processes, cleans, and analyzes large volumes of text to uncover communication patterns and key semantic relationships.

📂 Dataset

Source: Enron Email Dataset (Kaggle)

Cleaning: Performed using Python’s email module to extract fields like:

Message-ID

Date

From

To

Subject

Body

The cleaned dataset was stored as emails_clean.csv for further NLP and ML analysis.

⚙️ Preprocessing & Feature Engineering

Data Cleaning: Parsed raw .csv messages and removed malformed entries

Text Normalization: Stopword removal, lemmatization, and tokenization using SpaCy

Feature Extraction: TF-IDF vectorization and keyword analysis

Network Graphs: Constructed using NetworkX to visualize communication flow

🤖 ML & Gemini API Integration

Gemini API: Used for semantic enrichment, summarization, and extracting contextual insights from email bodies.

ML Models: Implemented clustering and classification to detect communication trends and thematic groupings.

Text Analysis: Combined Gemini-powered embeddings with traditional NLP features for improved interpretability.

🧰 Tech Stack

Languages: Python

Libraries: Pandas, NumPy, SpaCy, TextBlob, scikit-learn, NetworkX, Matplotlib

External API: Gemini API (Google Generative AI)

🚀 How to Run

Clone the repository

git clone https://github.com/<your-username>/email-insight-classifier.git
cd email-insight-classifier


Install dependencies

pip install -r requirements.txt


Run the cleaning script

python clean_emails.py


Analyze and generate insights

jupyter notebook analysis.ipynb

📊 Output

Cleaned dataset (emails_clean.csv)

Visualized communication network graph

Summarized email insights generated via Gemini API

Keyword and sentiment analysis reports
