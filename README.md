

# 🧠 ProInsight - Email Insight Classifier

This project applies **Machine Learning** and **Natural Language Processing (NLP)** techniques to analyze and classify corporate emails from the **[Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)**.  
It extracts semantic patterns, relationships, and insights from large-scale email communication data to help visualize and interpret professional correspondence.

---

## 🌐 Live Links

- **Frontend (Website):** [https://proinsight-frontend.vercel.app](https://pro-insight-frontend-18ku.vercel.app/)  
- **Backend API:** [https://proinsight-backend.onrender.com](https://proinsight-backend.onrender.com)

---

## 📂 Dataset

**Source:** [Enron Email Dataset (Kaggle)](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)

**Cleaning Process:**  
Raw email data was parsed using Python’s `email` module to extract:
- `Message-ID`  
- `Date`  
- `From`  
- `To`  
- `Subject`  
- `Body`

The cleaned dataset was saved as **`emails_clean.csv`** for downstream NLP and ML analysis.

---

## ⚙️ Preprocessing & Feature Engineering

- **Data Cleaning:** Removal of stopwords, punctuation, and non-ASCII characters.  
- **Tokenization & Lemmatization:** Performed using **SpaCy**.  
- **Feature Extraction:** TF-IDF vectorization and word frequency analysis.  
- **Network Analysis:** Constructed sender–receiver communication graphs using **NetworkX**.  

---

## 🧩 Machine Learning Pipeline

1. **Data Parsing & Cleaning** — Extracts and structures raw email data.  
2. **Exploratory Data Analysis (EDA)** — Analyzes communication frequency, sentiment, and relationships.  
3. **Feature Engineering** — Uses TF-IDF and embeddings for semantic representation.  
4. **Classification / Clustering** — Identifies thematic or behavioral patterns in email content.  
5. **Visualization** — Builds network graphs using **NetworkX** and **Matplotlib**.  

---

## 🤖 Gemini API Integration

The project integrates **Google’s Gemini API** for:
- Text summarization  
- Semantic similarity comparison  
- Context-aware keyword extraction  
- Insight generation on communication trends  

---

## 🧰 Tech Stack

- **Languages & Libraries:** Python, Pandas, NumPy  
- **NLP Tools:** SpaCy, TextBlob  
- **ML Framework:** scikit-learn  
- **Visualization:** Matplotlib, NetworkX  
- **API:** Gemini API  
- **Frontend:** React (Vite + Tailwind + shadcn/ui)  
- **Backend:** FastAPI (deployed on Render)  

---

## 🚀 Running the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/nikitagrover19/ProInsight-ML.git
cd ProInsight-ML
cd scripts
```

## Design Decisions & Tradeoffs

- **TF-IDF over dense embeddings**: Chose TF-IDF for interpretability and 
speed on 72k emails. Tradeoff: loses semantic similarity. Would use 
sentence-transformers if rebuilding.

- **Gemini API**: Free tier sufficient for prototype. Tradeoff: latency 
spikes on summarization. Would cache results with Redis in v2.

- **FastAPI + Redis**: Redis for inference caching, cutting repeat query 
latency by ~60%. Tradeoff: adds infra complexity for a prototype.

## What I'd Do Differently
- Replace TF-IDF with embeddings for semantic search
- Add proper eval pipeline with golden Q&A set
- Use async job queue for large batch processing
