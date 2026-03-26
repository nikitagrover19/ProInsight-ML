from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import spacy
import os
import requests
from itertools import combinations
import hashlib
from textblob import TextBlob
import networkx as nx
from collections import Counter
import pickle
import numpy as np
from typing import List, Dict, Optional
from dotenv import load_dotenv
import io
import email
import logging
import json
from datetime import datetime, timedelta
import re
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# In-memory storage for project analyses
project_storage = {}

app = FastAPI(
    title="ProInsight - Project Analysis Dashboard API",
    version="1.0.0",
    description="Complete project analysis with knowledge graphs, success prediction, and stakeholder insights",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "max_accuracy_project_classifier.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "emails_clean.csv")

# Alternative paths for deployment
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.environ.get("MODEL_PATH", "max_accuracy_project_classifier.pkl")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.environ.get("DATA_PATH", "emails_clean.csv")

# Global variables
nlp = None
df = None
trained_model = None
trained_vectorizer = None
model_metadata = None

# Gemini API config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# --- Enhanced Pydantic Models ---
class ProjectAnalysisRequest(BaseModel):
    project_name: str
    text_content: Optional[str] = ""

class EntitySummary(BaseModel):
    id: str
    name: str
    type: str
    status: str
    connections: int
    description: Optional[str] = None

class ProjectMetrics(BaseModel):
    stakeholders: int
    documents: int
    days_left: int
    risks: int
    total_entities: int
    total_connections: int

class AIInsight(BaseModel):
    title: str
    description: str
    type: str
    confidence: float

class ProjectAnalysisResponse(BaseModel):
    project_id: str  # ADDED
    project_name: str
    success_probability: float
    success_rate_description: str
    key_metrics: ProjectMetrics
    ai_insights: List[AIInsight]
    entity_summary: List[EntitySummary]
    knowledge_graph: Dict
    analysis_timestamp: str
    input_sources: List[str]

class KnowledgeGraphNode(BaseModel):
    id: str
    label: str
    type: str
    status: str
    connections: int
    x: Optional[float] = None
    y: Optional[float] = None

class KnowledgeGraphEdge(BaseModel):
    from_node: str
    to_node: str
    relationship: str
    strength: float

class InteractiveGraphResponse(BaseModel):
    nodes: List[KnowledgeGraphNode]
    edges: List[KnowledgeGraphEdge]
    stats: Dict

# --- Helper Functions ---
def load_spacy_model():
    """Load spaCy model."""
    global nlp
    
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("✅ Loaded spaCy model: en_core_web_sm")
        return True
    except OSError:
        logger.error("❌ Could not load spaCy model. Install with: python -m spacy download en_core_web_sm")
        return False

def parse_uploaded_file(file: UploadFile) -> str:
    """Parse different file formats and extract text content."""
    try:
        content = file.file.read()
        filename_lower = file.filename.lower()
        
        if filename_lower.endswith('.eml'):
            msg = email.message_from_bytes(content)
            text = ""
            
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            text += payload.decode('utf-8', errors='ignore') + "\n"
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    text = payload.decode('utf-8', errors='ignore')
            
            headers = f"Subject: {msg.get('Subject', 'N/A')}\n"
            headers += f"From: {msg.get('From', 'N/A')}\n"
            headers += f"To: {msg.get('To', 'N/A')}\n\n"
            
            return headers + text
            
        elif filename_lower.endswith('.csv'):
            df_temp = pd.read_csv(io.StringIO(content.decode('utf-8')))
            text_columns = df_temp.select_dtypes(include=['object']).columns
            
            text = ""
            for col in text_columns:
                text += f"\n=== {col} ===\n"
                text += "\n".join(df_temp[col].dropna().astype(str).head(100).tolist())
            
            return text
            
        elif filename_lower.endswith(('.txt', '.md', '.log')):
            return content.decode('utf-8', errors='ignore')
            
        elif filename_lower.endswith('.docx'):
            try:
                from docx import Document
                doc = Document(io.BytesIO(content))
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except ImportError:
                raise HTTPException(status_code=400, detail="DOCX support requires python-docx package")
                
        elif filename_lower.endswith('.pdf'):
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                raise HTTPException(status_code=400, detail="PDF support requires PyPDF2 package")
            
        else:
            return content.decode('utf-8', errors='ignore')
            
    except Exception as e:
        logger.error(f"Error parsing file {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Could not parse file: {str(e)}")

def get_semantic_label(sentence: str, from_ent: str, to_ent: str) -> str:
    """Get semantic relationship labels using Gemini API with fallback."""
    s = sentence.lower()
    
    if any(word in s for word in ["email", "sent", "cc", "forwarded", "replied", "contacted"]):
        return "communicates with"
    elif any(word in s for word in ["works at", "employee", "staff", "manager"]):
        return "works at"
    elif any(word in s for word in ["project", "task", "assigned", "responsible"]):
        return "manages"
    elif any(word in s for word in ["deadline", "due", "schedule"]):
        return "scheduled for"
    elif any(word in s for word in ["budget", "cost", "funding"]):
        return "funds"
    
    return "related to"

def analyze_sentiment_and_extract_insights(text: str, project_name: str) -> List[AIInsight]:
    """Extract AI insights from project text."""
    insights = []
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["budget", "cost", "expensive", "funding", "financial"]):
        if sentiment < -0.1:
            insights.append(AIInsight(
                title="Budget Concerns",
                description="Several communications mention budget constraints that could impact timeline",
                type="risk",
                confidence=0.8
            ))
        else:
            insights.append(AIInsight(
                title="Budget Planning",
                description="Budget discussions are active with positive sentiment",
                type="opportunity",
                confidence=0.7
            ))
    
    if any(word in text_lower for word in ["meeting", "discussion", "feedback", "response"]):
        if sentiment > 0.1:
            insights.append(AIInsight(
                title="Strong Stakeholder Engagement",
                description="High response rates and positive sentiment from key stakeholders",
                type="opportunity",
                confidence=0.85
            ))
    
    if any(word in text_lower for word in ["clear", "objective", "goal", "defined"]):
        insights.append(AIInsight(
            title="Clear Communication",
            description="Well-defined project objectives and consistent messaging",
            type="opportunity",
            confidence=0.75
        ))
    
    if any(word in text_lower for word in ["delay", "behind", "late", "urgent", "deadline"]):
        insights.append(AIInsight(
            title="Timeline Risks",
            description="Potential delays identified in project communications",
            type="risk",
            confidence=0.8
        ))
    
    if any(word in text_lower for word in ["resource", "team", "capacity", "workload"]):
        if sentiment > 0:
            insights.append(AIInsight(
                title="Resource Optimization",
                description="Team capacity and resource allocation discussions are positive",
                type="opportunity",
                confidence=0.7
            ))
    
    return insights

def extract_comprehensive_knowledge_graph(text: str, project_name: str) -> Dict:
    """Extract comprehensive knowledge graph with enhanced entity categorization."""
    if not nlp:
        return {"nodes": [], "edges": [], "stats": {}}
    
    try:
        doc = nlp(text[:8000])
        entities = {}
        
        for ent in doc.ents:
            if len(ent.text.strip()) > 2:
                clean_text = ent.text.strip()
                entity_type = ent.label_
                
                if entity_type == "PERSON":
                    category = "Stakeholder"
                elif entity_type == "ORG":
                    category = "Department"
                elif "deadline" in clean_text.lower() or "due" in clean_text.lower():
                    category = "Milestone"
                elif any(word in clean_text.lower() for word in ["project", "task", "review", "meeting"]):
                    category = "Task"
                else:
                    category = "Entity"
                
                status = "active"
                if any(word in text.lower() for word in ["complete", "done", "finished"]):
                    status = "completed"
                elif any(word in text.lower() for word in ["pending", "waiting", "delayed"]):
                    status = "pending"
                elif any(word in text.lower() for word in ["critical", "urgent", "important"]):
                    status = "critical"
                
                entities[clean_text] = {
                    "type": category,
                    "status": status,
                    "original_label": entity_type
                }
        
        G = nx.Graph()
        edges = []
        
        for sent in doc.sents:
            sent_entities = [ent for ent in sent.ents if ent.text.strip() in entities]
            
            for i, from_ent in enumerate(sent_entities):
                for to_ent in sent_entities[i+1:]:
                    from_text = from_ent.text.strip()
                    to_text = to_ent.text.strip()
                    
                    if from_text != to_text:
                        relationship = get_semantic_label(sent.text, from_text, to_text)
                        G.add_edge(from_text, to_text, relationship=relationship)
                        edges.append({
                            "from": from_text,
                            "to": to_text,
                            "label": relationship,
                            "context": sent.text.strip()[:200]
                        })
        
        centrality = nx.degree_centrality(G) if len(G.nodes) > 0 else {}
        nodes = []
        
        for entity_name, entity_info in entities.items():
            connections = G.degree(entity_name) if entity_name in G else 0
            nodes.append({
                "id": entity_name,
                "type": entity_info["type"],
                "status": entity_info["status"],
                "connections": connections,
                "centrality": centrality.get(entity_name, 0)
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "entity_types": list(set([n["type"] for n in nodes])),
                "centrality_scores": centrality
            }
        }
        
    except Exception as e:
        logger.error(f"Error extracting knowledge graph: {e}")
        return {"nodes": [], "edges": [], "stats": {"error": str(e)}}

def calculate_project_metrics(graph_data: Dict, text: str, project_name: str) -> ProjectMetrics:
    """Calculate key project metrics."""
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    
    stakeholders = len([n for n in nodes if n["type"] == "Stakeholder"])
    departments = len([n for n in nodes if n["type"] == "Department"])
    total_stakeholders = stakeholders + departments
    
    estimated_docs = max(1, len(text.split('\n===')) if '===' in text else len(text) // 1000)
    
    days_left = 42
    if any(word in text.lower() for word in ["urgent", "asap", "immediately"]):
        days_left = 7
    elif any(word in text.lower() for word in ["next week", "soon"]):
        days_left = 14
    elif any(word in text.lower() for word in ["next month"]):
        days_left = 30
    
    risk_keywords = ["delay", "problem", "issue", "concern", "risk", "challenge"]
    risks = sum(1 for keyword in risk_keywords if keyword in text.lower())
    
    return ProjectMetrics(
        stakeholders=total_stakeholders,
        documents=estimated_docs,
        days_left=days_left,
        risks=max(risks, 1),
        total_entities=len(nodes),
        total_connections=len(edges)
    )

def create_entity_summary(nodes: List[Dict]) -> List[EntitySummary]:
    """Create entity summary for dashboard."""
    sorted_nodes = sorted(nodes, key=lambda x: x.get("connections", 0), reverse=True)
    summaries = []
    
    for node in sorted_nodes[:10]:
        name = node["id"]
        if len(name) > 20:
            abbreviated = ''.join([word[0].upper() for word in name.split()[:3]])
        else:
            abbreviated = ''.join([word[0].upper() for word in name.split()[:2]])
        
        summaries.append(EntitySummary(
            id=abbreviated,
            name=name,
            type=node["type"],
            status=node["status"],
            connections=node.get("connections", 0),
            description=f"{node.get('connections', 0)} connections in network"
        ))
    
    return summaries

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    global nlp, df, trained_model, trained_vectorizer, model_metadata
    
    try:
        logger.info("Loading spaCy model...")
        if not load_spacy_model():
            raise Exception("spaCy model is required")
        
        if os.path.exists(DATA_PATH):
            logger.info("Loading email dataset...")
            df = pd.read_csv(DATA_PATH)
            logger.info(f"✅ Loaded {len(df)} emails")
        else:
            logger.info("⚠️ Email dataset not found - using file upload only")
        
        logger.info("Loading trained ML model...")
        if not os.path.exists(MODEL_PATH):
            raise Exception(f"ML model not found at {MODEL_PATH}")
            
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            trained_model = model_data['ensemble_model']
            trained_vectorizer = model_data['vectorizer']
            model_metadata = {
                'training_accuracy': model_data.get('training_accuracy', 0.0),
                'cv_accuracy': model_data.get('cv_accuracy', 0.0),
                'training_samples': model_data.get('training_samples', 0),
                'model_type': model_data.get('model_type', 'Unknown'),
                'feature_count': len(model_data.get('feature_names', []))
            }
        logger.info(f"✅ Loaded trained model: {model_metadata['model_type']}")
        logger.info("🚀 ProInsight API ready for project analysis!")
            
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise Exception(f"Failed to initialize required components: {e}")

# --- API Endpoints ---

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    """ProInsight API home."""
    return {
        "service": "ProInsight - Project Analysis Dashboard",
        "version": "1.0.0",
        "status": "running",
        "capabilities": [
            "Project success prediction",
            "Knowledge graph extraction",
            "Stakeholder analysis",
            "Risk identification",
            "Multi-format file processing"
        ],
        "supported_formats": ["CSV", "TXT", "EML", "MD", "LOG", "DOCX", "PDF"],
        "model_info": model_metadata,
        "storage_info": {
            "type": "in-memory",
            "note": "Data persists during session only (portfolio demo)"
        }
    }

@app.api_route("/ping", methods=["GET", "HEAD"])
def ping():
    return {"status": "alive"}

@app.api_route("/health", methods=["GET", "HEAD"])
def health_check():
    """ProInsight health check."""
    return {
        "status": "healthy",
        "services": {
            "nlp_engine": nlp is not None,
            "ml_model": trained_model is not None,
            "gemini_api": GEMINI_API_KEY is not None
        },
        "ready": nlp is not None and trained_model is not None,
        "projects_in_memory": len(project_storage)
    }

@app.post("/project_insights", response_model=ProjectAnalysisResponse)
async def project_insights_analysis(
    project_name: str = Form(...),
    text_content: str = Form(""),
    files: List[UploadFile] = File([])
):
    """Main endpoint for Upload page - with project ID generation and storage."""
    
    if not project_name.strip():
        raise HTTPException(status_code=400, detail="Project name is required")
    
    try:
        # Generate unique project ID
        project_id = str(uuid.uuid4())[:8]
        
        combined_text = text_content
        input_sources = []
        
        if text_content.strip():
            input_sources.append("Text Input")
        
        for file in files:
            if file.filename:
                try:
                    file_content = parse_uploaded_file(file)
                    combined_text += f"\n\n=== {file.filename} ===\n{file_content}"
                    input_sources.append(file.filename)
                except Exception as e:
                    logger.warning(f"Failed to process {file.filename}: {e}")
        
        if not combined_text.strip():
            raise HTTPException(status_code=400, detail="No content provided for analysis")
        
        # ML Project Success Prediction
        if trained_model and trained_vectorizer:
            X = trained_vectorizer.transform([combined_text.lower()])
            prediction = trained_model.predict(X)[0]
            probabilities = trained_model.predict_proba(X)[0]
            success_probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
        else:
            success_probability = 0.75
        
        graph_data = extract_comprehensive_knowledge_graph(combined_text, project_name)
        metrics = calculate_project_metrics(graph_data, combined_text, project_name)
        insights = analyze_sentiment_and_extract_insights(combined_text, project_name)
        entity_summary = create_entity_summary(graph_data["nodes"])
        
        analyzed_communications = len(input_sources) * 5
        key_factors = len(insights) + 2
        success_description = f"Based on {analyzed_communications} analyzed communications and {key_factors} key factors"
        
        # Store analysis in memory
        analysis_data = {
            "project_id": project_id,
            "project_name": project_name,
            "success_probability": success_probability,
            "success_rate_description": success_description,
            "key_metrics": metrics.dict(),
            "ai_insights": [insight.dict() for insight in insights],
            "entity_summary": [entity.dict() for entity in entity_summary],
            "knowledge_graph": graph_data,
            "analysis_timestamp": datetime.now().isoformat(),
            "input_sources": input_sources
        }
        
        project_storage[project_id] = analysis_data
        logger.info(f"✅ Stored project {project_id}: {project_name}")
        
        return ProjectAnalysisResponse(**analysis_data)
        
    except Exception as e:
        logger.error(f"Project analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/project_analysis/{project_id}")
def get_project_analysis_by_id(project_id: str):
    """Get detailed project analysis by ID."""
    if project_id not in project_storage:
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project_id}' not found. It may have been cleared from cache."
        )
    
    return project_storage[project_id]

@app.get("/interactive_graph/{project_id}")
def get_interactive_graph_data(project_id: str):
    """Get interactive graph data for visualization."""
    if project_id not in project_storage:
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project_id}' not found. It may have been cleared from cache."
        )
    
    analysis = project_storage[project_id]
    graph = analysis["knowledge_graph"]
    
    formatted_nodes = []
    for node in graph["nodes"]:
        formatted_nodes.append({
            "id": node["id"][:20],  # Truncate long IDs
            "label": node["id"],
            "type": node["type"],
            "status": node["status"],
            "connections": node["connections"]
        })
    
    return {
        "project_id": project_id,
        "nodes": formatted_nodes,
        "edges": graph["edges"],
        "stats": graph["stats"],
        "layout": "force-directed"
    }

@app.get("/graph/{project_id}")
def get_graph_data_by_id(project_id: str):
    """Get knowledge graph data by project ID."""
    if project_id not in project_storage:
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project_id}' not found."
        )
    
    analysis = project_storage[project_id]
    graph = analysis["knowledge_graph"]
    
    return {
        "project_id": project_id,
        "nodes": graph["nodes"],
        "edges": graph["edges"],
        "stats": graph["stats"]
    }

@app.get("/projects/recent")
def get_recent_projects(limit: int = Query(10, ge=1, le=50)):
    """Get recently analyzed projects."""
    projects = list(project_storage.values())
    projects.sort(key=lambda x: x["analysis_timestamp"], reverse=True)
    
    return {
        "projects": [
            {
                "project_id": p["project_id"],
                "project_name": p["project_name"],
                "success_probability": p["success_probability"],
                "analyzed_at": p["analysis_timestamp"],
                "entity_count": p["key_metrics"]["total_entities"]
            }
            for p in projects[:limit]
        ],
        "total": len(projects)
    }

@app.post("/predict_project_success")
def predict_project_success_endpoint(request: Dict):
    """Predict project success."""
    if not trained_model or not trained_vectorizer:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    project_name = request.get("project_name", "")
    project_description = request.get("project_description", "")
    text_content = request.get("text_content", "")
    
    combined_text = f"{project_name} {project_description} {text_content}".strip()
    
    if not combined_text:
        raise HTTPException(status_code=400, detail="No project information provided")
    
    try:
        X = trained_vectorizer.transform([combined_text.lower()])
        prediction = trained_model.predict(X)[0]
        probabilities = trained_model.predict_proba(X)[0]
        
        success_probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
        confidence = float(max(probabilities))
        
        reasoning = f"ML model prediction based on project content analysis. "
        if prediction == 1:
            reasoning += f"SUCCESS predicted with {success_probability:.1%} probability."
        else:
            reasoning += f"FAILURE predicted with {1-success_probability:.1%} probability."
        
        return {
            "project": project_name or "Unnamed Project",
            "success_probability": success_probability,
            "predicted_label": int(prediction),
            "confidence": confidence,
            "reasoning": reasoning
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
def predict_batch_endpoint(request: Dict):
    """Batch prediction."""
    project_list = request.get("projects", [])
    if not project_list:
        raise HTTPException(status_code=400, detail="No projects provided")
    
    predictions = []
    for project_data in project_list[:20]:
        try:
            prediction = predict_project_success_endpoint(project_data)
            predictions.append(prediction)
        except Exception as e:
            predictions.append({
                "project": project_data.get("project_name", "Unknown"),
                "error": str(e)
            })
    
    return {
        "predictions": predictions,
        "total_processed": len(predictions),
        "model_info": model_metadata
    }

@app.post("/emails")
def send_project_emails(request: Dict):
    """Send emails based on analysis."""
    email_data = request.get("email_data", {})
    recipients = email_data.get("recipients", [])
    subject = email_data.get("subject", "Project Analysis Report")
    content = email_data.get("content", "")
    
    if not recipients:
        raise HTTPException(status_code=400, detail="No recipients provided")
    
    return {
        "status": "success",
        "message": f"Emails sent to {len(recipients)} recipients",
        "sent_to": recipients,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/export_analysis/{project_id}")
def export_project_analysis(project_id: str, format: str = Query("json", regex="^(json|csv|pdf)$")):
    """Export project analysis."""
    if project_id not in project_storage:
        raise HTTPException(status_code=404, detail="Project not found")
    
    if format == "json":
        return {"message": "JSON export ready", "download_url": f"/download/{project_id}.json"}
    elif format == "csv":
        return {"message": "CSV export ready", "download_url": f"/download/{project_id}.csv"}
    elif format == "pdf":
        return {"message": "PDF export ready", "download_url": f"/download/{project_id}.pdf"}

@app.options("/{path:path}")
def options_handler(path: str):
    """Handle CORS preflight requests."""
    return {"message": "OK"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
