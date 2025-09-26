
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import pandas as pd
import spacy
import os
import requests
from itertools import combinations
import redis
import hashlib
from textblob import TextBlob
import networkx as nx
from collections import Counter
import pickle
import numpy as np
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
app = FastAPI(title="Enhanced ML Service with Project Success Prediction", version="2.0.0")

# --- Global variables ---
nlp = None
df = None
trained_model = None
trained_vectorizer = None
model_metadata = None

# Gemini API config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# --- Redis Setup ---
try:
    redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    redis_client.ping()
    USE_REDIS = True
    print("✅ Redis connected")
except Exception as e:
    print(f"⚠️ Redis not available: {e}")
    redis_client = None
    USE_REDIS = False

# --- Pydantic Models ---
class ProjectPredictionRequest(BaseModel):
    project_name: str

class ProjectPredictionResponse(BaseModel):
    project: str
    success_probability: float
    predicted_label: int
    confidence: float
    reasoning: str

class BatchPredictionRequest(BaseModel):
    project_names: List[str]

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    global nlp, df, trained_model, trained_vectorizer, model_metadata
    
    try:
        # Load spaCy model
        print("Loading spaCy model...")
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model loaded")
        
        # Load email dataset
        print("Loading email dataset...")
        df = pd.read_csv("/Users/nikitagrover/ml+proj/data/processed/emails_clean.csv")
        print(f"✅ Loaded {len(df)} emails")
        
        # Load your trained maximum accuracy model
        print("Loading trained ML model...")
        model_path = "/Users/nikitagrover/ml+proj/models/max_accuracy_project_classifier.pkl"
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                trained_model = model_data['ensemble_model']
                trained_vectorizer = model_data['vectorizer']
                model_metadata = {
                    'training_accuracy': model_data['training_accuracy'],
                    'cv_accuracy': model_data['cv_accuracy'],
                    'training_samples': model_data['training_samples'],
                    'model_type': model_data['model_type'],
                    'feature_count': len(model_data['feature_names'])
                }
            print(f"✅ Loaded trained model: {model_metadata['model_type']}")
            print(f"   - CV Accuracy: {model_metadata['cv_accuracy']:.3f}")
            print(f"   - Training samples: {model_metadata['training_samples']}")
            print(f"   - Features: {model_metadata['feature_count']}")
        else:
            print(f"⚠️ Model not found at {model_path}")
            trained_model = None
            trained_vectorizer = None
            
    except Exception as e:
        print(f"❌ Startup error: {e}")

# --- Helper Functions ---
def get_cache_key(*args) -> str:
    raw_key = ":".join(str(arg) for arg in args)
    return hashlib.sha256(raw_key.encode()).hexdigest()

def rule_based_label(sentence: str, from_ent: str, to_ent: str) -> str:
    s = sentence.lower()
    if any(word in s for word in ["email", "sent", "cc", "forwarded", "replied", "contacted", "request"]):
        if "forwarded" in s or "request" in s:
            return "referred to"
        elif "replied" in s:
            return "replied to"
        return "emailed"
    elif any(word in s for word in ["works at", "joined", "employee", "staff", "manager", "supervisor"]):
        return "works at"
    elif any(word in s for word in ["project", "task", "assigned", "about", "report", "deadline"]):
        return "about"
    elif any(word in s for word in ["located", "near", "behind", "in", "at"]):
        return "located in"
    elif any(word in s for word in ["part of", "subsidiary", "division", "department", "unit"]):
        return "part of"
    return "related"

def get_semantic_label(sentence: str, from_ent: str, to_ent: str) -> str:
    cache_key = f"relation:{from_ent}:{to_ent}:{sentence}"
    if USE_REDIS:
        cached_value = redis_client.get(cache_key)
        if cached_value:
            return cached_value

    prompt = f"""
    Sentence: "{sentence}"
    Entities: "{from_ent}" and "{to_ent}"

    Task: Describe the relationship between these entities in 1–3 words only.
    If nothing specific applies, answer "related".
    Respond ONLY with the relationship phrase.
    """
    headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    label = "related"

    try:
        response = requests.post(GEMINI_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        label = result["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
    except Exception as e:
        print(f"Gemini API Error: {e}")

    if label == "related":
        label = rule_based_label(sentence, from_ent, to_ent)

    if USE_REDIS:
        redis_client.set(cache_key, label, ex=86400)
    return label

def get_project_emails(project_name: str) -> pd.DataFrame:
    """Get emails related to a specific project."""
    if df is None:
        return pd.DataFrame()
    
    project_name_lower = project_name.lower()
    return df[df['Body'].str.lower().str.contains(project_name_lower, na=False)]

def extract_project_features(project_emails: pd.DataFrame):
    """Extract network and sentiment features from project emails."""
    if nlp is None or project_emails.empty:
        return {}, nx.DiGraph()
    
    G = nx.DiGraph()
    email_counts = Counter()
    sentiment_scores = []

    for _, row in project_emails.iterrows():
        try:
            text = str(row['Body'])
            doc = nlp(text)
            entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]]
            
            for from_ent, to_ent in combinations(entities, 2):
                G.add_edge(from_ent, to_ent)
            
            for ent in entities:
                email_counts[ent] += 1
                
            sentiment_scores.append(TextBlob(text).sentiment.polarity)
        except Exception as e:
            print(f"Error processing email: {e}")
            continue

    centrality = nx.degree_centrality(G)
    features = {
        "avg_sentiment": sum(sentiment_scores)/len(sentiment_scores) if sentiment_scores else 0,
        "avg_email_count": sum(email_counts.values()) / len(email_counts) if email_counts else 0,
        "avg_centrality": sum(centrality.values()) / len(centrality) if centrality else 0,
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges()
    }
    return features, G

# --- API Endpoints ---
@app.get("/")
def home():
    return {
        "message": "Enhanced ML Service with Project Success Prediction",
        "status": "running",
        "model_loaded": trained_model is not None,
        "model_info": model_metadata if model_metadata else "No model loaded"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "spacy_loaded": nlp is not None,
        "data_loaded": df is not None,
        "model_loaded": trained_model is not None,
        "redis_connected": USE_REDIS,
        "total_emails": len(df) if df is not None else 0
    }

@app.get("/model_info")
def get_model_info():
    """Get information about the loaded ML model."""
    if trained_model is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    return {
        "model_metadata": model_metadata,
        "model_type": "Ensemble Classifier (Random Forest + Gradient Boosting + Logistic Regression)",
        "accuracy": f"{model_metadata['cv_accuracy']:.1%}",
        "status": "Production Ready" if model_metadata['cv_accuracy'] > 0.7 else "Needs Improvement"
    }

@app.post("/predict_project_success", response_model=ProjectPredictionResponse)
def predict_project_success(request: ProjectPredictionRequest):
    """Predict project success using the trained ensemble model."""
    if trained_model is None or trained_vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check startup logs.")
    
    try:
        # Normalize project name (similar to training preprocessing)
        project_name = request.project_name.lower().strip()
        
        # Transform using the trained vectorizer
        X = trained_vectorizer.transform([project_name])
        
        # Make prediction
        prediction = trained_model.predict(X)[0]
        probabilities = trained_model.predict_proba(X)[0]
        
        success_probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
        confidence = float(max(probabilities))
        
        # Create reasoning
        reasoning = f"Model prediction based on project name analysis. "
        if prediction == 1:
            reasoning += f"SUCCESS predicted with {success_probability:.1%} probability."
        else:
            reasoning += f"FAILURE predicted with {1-success_probability:.1%} probability."
        
        return ProjectPredictionResponse(
            project=request.project_name,
            success_probability=success_probability,
            predicted_label=int(prediction),
            confidence=confidence,
            reasoning=reasoning
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
def predict_batch_projects(request: BatchPredictionRequest):
    """Predict success for multiple projects."""
    if trained_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    for project_name in request.project_names:
        try:
            pred_request = ProjectPredictionRequest(project_name=project_name)
            prediction = predict_project_success(pred_request)
            predictions.append(prediction.dict())
        except Exception as e:
            predictions.append({
                "project": project_name,
                "error": str(e)
            })
    
    return {"predictions": predictions}

@app.get("/emails")
def get_emails(limit: int = Query(5, ge=1, le=100)):
    """Get sample emails from the dataset."""
    if df is None:
        raise HTTPException(status_code=503, detail="Email data not loaded")
    
    sample = df.head(limit).fillna("").to_dict(orient="records")
    return {"emails": sample, "total_emails": len(df)}

@app.get("/graph")
def get_knowledge_graph(email_index: int = Query(0, ge=0)):
    """Extract knowledge graph from a specific email."""
    if df is None or nlp is None:
        raise HTTPException(status_code=503, detail="Services not loaded")
    
    if email_index >= len(df):
        raise HTTPException(status_code=404, detail="Email index out of range")

    text = df.iloc[email_index]["Body"].replace("\n", " ").strip()
    doc = nlp(text)

    # Extract unique entities
    unique_nodes = {}
    for ent in doc.ents:
        if len(ent.text) > 2 and not any(c.isdigit() for c in ent.text):
            unique_nodes[ent.text] = ent.label_
    nodes = [{"id": k, "type": v} for k, v in unique_nodes.items()]

    # Extract relationships
    edges = []
    seen_edges = set()
    for sent in doc.sents:
        ents_in_sent = [ent for ent in sent.ents if ent.text in unique_nodes]
        for from_ent, to_ent in combinations(ents_in_sent, 2):
            from_type = unique_nodes[from_ent.text]
            to_type = unique_nodes[to_ent.text]
            
            # Maintain consistent direction
            if from_type == "ORG" and to_type == "PERSON":
                from_ent, to_ent = to_ent, from_ent
            
            key = (from_ent.text, to_ent.text)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            
            label = get_semantic_label(sent.text, from_ent.text, to_ent.text)
            edges.append({
                "from": from_ent.text,
                "to": to_ent.text,
                "label": label,
                "context": sent.text
            })

    return {
        "email_index": email_index,
        "email_body": text,
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": list(set(node["type"] for node in nodes))
        }
    }

@app.get("/project_analysis")
def comprehensive_project_analysis(project: str = Query(..., description="Project name to analyze")):
    """Comprehensive project analysis combining ML prediction with email analysis."""
    cache_key = get_cache_key("project_analysis", project)
    
    if USE_REDIS:
        cached = redis_client.get(cache_key)
        if cached:
            return eval(cached)

    # Get project emails

    project_emails = get_project_emails(project).head(50)  # process only first 50

    if project_emails.empty:
        raise HTTPException(status_code=404, detail="No emails found for this project")

    # ML prediction using trained model
    ml_prediction = None
    if trained_model is not None:
        try:
            pred_request = ProjectPredictionRequest(project_name=project)
            ml_prediction = predict_project_success(pred_request).dict()
        except Exception as e:
            print(f"ML prediction failed: {e}")

    # Network analysis
    features, graph = extract_project_features(project_emails)
    
    # Generate summary using Gemini
    summary = "Summary generation unavailable"
    if GEMINI_API_KEY:
        try:
            text = " ".join(project_emails['Body'].head(5).astype(str).tolist())
            prompt = f"""
            Analyze these project emails and provide insights:
            {text[:2000]}...
            
            Focus on:
            1. Communication patterns
            2. Potential risks or issues
            3. Team dynamics
            4. Project timeline concerns
            
            Provide 3-5 concise bullet points.
            """
            headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            
            response = requests.post(GEMINI_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            summary = result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            print(f"Summary generation failed: {e}")

    result = {
        "project": project,
        "email_count": len(project_emails),
        "ml_prediction": ml_prediction,
        "network_analysis": {
            "features": features,
            "key_people": list(graph.nodes)[:10],
            "connections": len(graph.edges)
        },
        "ai_summary": summary,
        "analysis_timestamp": pd.Timestamp.now().isoformat()
    }

    if USE_REDIS:
        redis_client.set(cache_key, str(result), ex=3600)  # Cache for 1 hour

    return result

@app.get("/business_predictions")
def test_business_predictions():
    """Test the model on various business project scenarios."""
    if trained_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    test_projects = [
        "website redesign project",
        "mobile app development",
        "database migration task",
        "system integration project",
        "marketing campaign launch",
        "cost reduction initiative",
        "legacy system maintenance",
        "emergency response task"
    ]
    
    predictions = []
    for project in test_projects:
        try:
            pred_request = ProjectPredictionRequest(project_name=project)
            prediction = predict_project_success(pred_request)
            predictions.append(prediction.dict())
        except Exception as e:
            predictions.append({"project": project, "error": str(e)})
    
    return {
        "test_predictions": predictions,
        "model_info": model_metadata
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
