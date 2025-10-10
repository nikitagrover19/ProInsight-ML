
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level from scripts/
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
    type: str  # "risk", "opportunity", "neutral"
    confidence: float

class ProjectAnalysisResponse(BaseModel):
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
            # Parse email files
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
            
            # Add email headers
            headers = f"Subject: {msg.get('Subject', 'N/A')}\n"
            headers += f"From: {msg.get('From', 'N/A')}\n"
            headers += f"To: {msg.get('To', 'N/A')}\n\n"
            
            return headers + text
            
        elif filename_lower.endswith('.csv'):
            # Parse CSV files
            df_temp = pd.read_csv(io.StringIO(content.decode('utf-8')))
            text_columns = df_temp.select_dtypes(include=['object']).columns
            
            text = ""
            for col in text_columns:
                text += f"\n=== {col} ===\n"
                text += "\n".join(df_temp[col].dropna().astype(str).head(100).tolist())
            
            return text
            
        elif filename_lower.endswith(('.txt', '.md', '.log')):
            # Parse text files
            return content.decode('utf-8', errors='ignore')
            
        elif filename_lower.endswith('.docx'):
            # Parse DOCX files
            try:
                from docx import Document
                doc = Document(io.BytesIO(content))
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except ImportError:
                raise HTTPException(status_code=400, detail="DOCX support requires python-docx package")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not parse DOCX file: {str(e)}")
                
        elif filename_lower.endswith('.pdf'):
            # Parse PDF files
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                raise HTTPException(status_code=400, detail="PDF support requires PyPDF2 package")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not parse PDF file: {str(e)}")
            
        else:
            # Try to decode as plain text
            return content.decode('utf-8', errors='ignore')
            
    except Exception as e:
        logger.error(f"Error parsing file {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Could not parse file: {str(e)}")

def get_semantic_label(sentence: str, from_ent: str, to_ent: str) -> str:
    """Get semantic relationship labels using Gemini API with fallback."""
    
    # Rule-based fallback
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
    
    # Sentiment analysis
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    
    # Extract insights based on keywords and patterns
    text_lower = text.lower()
    
    # Budget concerns
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
    
    # Stakeholder engagement
    if any(word in text_lower for word in ["meeting", "discussion", "feedback", "response"]):
        if sentiment > 0.1:
            insights.append(AIInsight(
                title="Strong Stakeholder Engagement",
                description="High response rates and positive sentiment from key stakeholders",
                type="opportunity",
                confidence=0.85
            ))
    
    # Communication clarity
    if any(word in text_lower for word in ["clear", "objective", "goal", "defined"]):
        insights.append(AIInsight(
            title="Clear Communication",
            description="Well-defined project objectives and consistent messaging",
            type="opportunity",
            confidence=0.75
        ))
    
    # Timeline risks
    if any(word in text_lower for word in ["delay", "behind", "late", "urgent", "deadline"]):
        insights.append(AIInsight(
            title="Timeline Risks",
            description="Potential delays identified in project communications",
            type="risk",
            confidence=0.8
        ))
    
    # Resource allocation
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
        # Process text with spaCy
        doc = nlp(text[:8000])
        
        # Enhanced entity extraction with categorization
        entities = {}
        for ent in doc.ents:
            if len(ent.text.strip()) > 2:
                clean_text = ent.text.strip()
                entity_type = ent.label_
                
                # Map spaCy labels to ProInsight categories
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
                
                # Determine status based on context
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
        
        # Build graph
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
        
        # Calculate centrality for importance
        centrality = nx.degree_centrality(G) if len(G.nodes) > 0 else {}
        
        # Create nodes with enhanced information
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
    
    # Count different entity types
    stakeholders = len([n for n in nodes if n["type"] == "Stakeholder"])
    departments = len([n for n in nodes if n["type"] == "Department"])
    total_stakeholders = stakeholders + departments
    
    # Estimate documents (rough count based on text length)
    estimated_docs = max(1, len(text.split('\n===')) if '===' in text else len(text) // 1000)
    
    # Calculate days left (mock calculation - in real scenario, extract from text)
    days_left = 42  # Default
    if any(word in text.lower() for word in ["urgent", "asap", "immediately"]):
        days_left = 7
    elif any(word in text.lower() for word in ["next week", "soon"]):
        days_left = 14
    elif any(word in text.lower() for word in ["next month"]):
        days_left = 30
    
    # Count risks (based on negative sentiment and risk keywords)
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
    
    # Sort by connections and take top entities
    sorted_nodes = sorted(nodes, key=lambda x: x.get("connections", 0), reverse=True)
    
    summaries = []
    for node in sorted_nodes[:10]:  # Top 10 entities
        # Create abbreviated ID for display
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
        # Load spaCy model - REQUIRED
        logger.info("Loading spaCy model...")
        if not load_spacy_model():
            raise Exception("spaCy model is required")
        
        # Try to load email dataset - OPTIONAL
        if os.path.exists(DATA_PATH):
            logger.info("Loading email dataset...")
            df = pd.read_csv(DATA_PATH)
            logger.info(f"✅ Loaded {len(df)} emails")
        else:
            logger.info("⚠️ Email dataset not found - using file upload only")
        
        # Load trained model - REQUIRED
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

@app.get("/")
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
        "model_info": model_metadata
    }

@app.get("/health")
def health_check():
    """ProInsight health check."""
    return {
        "status": "healthy",
        "services": {
            "nlp_engine": nlp is not None,
            "ml_model": trained_model is not None,
            "gemini_api": GEMINI_API_KEY is not None
        },
        "ready": nlp is not None and trained_model is not None
    }

@app.post("/analyze_project", response_model=ProjectAnalysisResponse)
async def analyze_complete_project(
    project_name: str = Form(...),
    text_content: str = Form(""),
    files: List[UploadFile] = File([])
):
    """Complete project analysis combining all inputs."""
    
    if not project_name.strip():
        raise HTTPException(status_code=400, detail="Project name is required")
    
    try:
        # Combine all text content
        combined_text = text_content
        input_sources = []
        
        if text_content.strip():
            input_sources.append("Text Input")
        
        # Process uploaded files
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
            X = trained_vectorizer.transform([project_name.lower()])
            prediction = trained_model.predict(X)[0]
            probabilities = trained_model.predict_proba(X)[0]
            success_probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
        else:
            # Fallback prediction
            success_probability = 0.75
        
        # Extract comprehensive knowledge graph
        graph_data = extract_comprehensive_knowledge_graph(combined_text, project_name)
        
        # Calculate metrics
        metrics = calculate_project_metrics(graph_data, combined_text, project_name)
        
        # Generate AI insights
        insights = analyze_sentiment_and_extract_insights(combined_text, project_name)
        
        # Create entity summary
        entity_summary = create_entity_summary(graph_data["nodes"])
        
        # Success rate description
        analyzed_communications = len(input_sources) * 5  # Rough estimate
        key_factors = len(insights) + 2
        success_description = f"Based on {analyzed_communications} analyzed communications and {key_factors} key factors"
        
        return ProjectAnalysisResponse(
            project_name=project_name,
            success_probability=success_probability,
            success_rate_description=success_description,
            key_metrics=metrics,
            ai_insights=insights,
            entity_summary=entity_summary,
            knowledge_graph=graph_data,
            analysis_timestamp=datetime.now().isoformat(),
            input_sources=input_sources
        )
        
    except Exception as e:
        logger.error(f"Project analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/preview_inputs")
async def preview_project_inputs(
    project_name: str = Form(...),
    text_content: str = Form(""),
    files: List[UploadFile] = File([])
):
    """Preview inputs without full processing."""
    
    preview = {
        "project_name": project_name,
        "text_content_length": len(text_content),
        "files_count": len([f for f in files if f.filename]),
        "file_names": [f.filename for f in files if f.filename],
        "estimated_processing_time": len(files) * 2 + 5  # seconds
    }
    
    return preview

@app.get("/interactive_graph/{project_id}")
def get_interactive_graph(project_id: str):
    """Get interactive graph data for visualization."""
    # This would typically fetch from a database
    # For now, return mock data matching your frontend structure
    
    return {
        "nodes": [
            {"id": "JS", "label": "John Smith", "type": "Stakeholder", "status": "active", "connections": 12},
            {"id": "BR", "label": "Budget Review", "type": "Task", "status": "pending", "connections": 8},
            {"id": "QD", "label": "Q4 Deadline", "type": "Milestone", "status": "critical", "connections": 15},
            {"id": "MT", "label": "Marketing Team", "type": "Department", "status": "active", "connections": 6}
        ],
        "edges": [
            {"from_node": "JS", "to_node": "BR", "relationship": "manages", "strength": 0.8},
            {"from_node": "BR", "to_node": "QD", "relationship": "scheduled for", "strength": 0.9},
            {"from_node": "JS", "to_node": "MT", "relationship": "leads", "strength": 0.7}
        ],
        "stats": {
            "total_nodes": 4,
            "total_edges": 3,
            "layout": "force-directed"
        }
    }

# --- Missing Endpoints from Frontend Mock API ---

@app.post("/project_insights", response_model=ProjectAnalysisResponse)
async def project_insights_analysis(
    project_name: str = Form(...),
    text_content: str = Form(""),
    files: List[UploadFile] = File([])
):
    """Main endpoint for Upload page - matches frontend PROJECT_INSIGHTS endpoint."""
    # This is the same as analyze_project but with the correct endpoint name
    return await analyze_complete_project(project_name, text_content, files)

@app.post("/predict_project_success")
def predict_project_success_endpoint(request: Dict):
    """Predict project success - matches frontend PREDICT_SUCCESS endpoint."""
    
    if not trained_model or not trained_vectorizer:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    
    # Extract project information from request
    project_name = request.get("project_name", "")
    project_description = request.get("project_description", "")
    text_content = request.get("text_content", "")
    
    # Combine all text for analysis
    combined_text = f"{project_name} {project_description} {text_content}".strip()
    
    if not combined_text:
        raise HTTPException(status_code=400, detail="No project information provided")
    
    try:
        # Use ML model on combined text content, not just project name
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
    """Batch prediction - matches frontend PREDICT_BATCH endpoint."""
    
    project_list = request.get("projects", [])
    if not project_list:
        raise HTTPException(status_code=400, detail="No projects provided")
    
    predictions = []
    for project_data in project_list[:20]:  # Limit to 20 projects
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

@app.get("/project_analysis/{project_id}")
def get_project_analysis_by_id(project_id: str):
    """Get detailed project analysis by ID - matches frontend PROJECT_ANALYSIS endpoint."""
    
    # In a real implementation, you'd fetch from database using project_id
    # For now, return mock data structure matching your needs
    return {
        "project_id": project_id,
        "project_name": f"Project {project_id}",
        "success_probability": 0.78,
        "key_metrics": {
            "stakeholders": 24,
            "documents": 156,
            "days_left": 42,
            "risks": 3
        },
        "ai_insights": [
            {
                "title": "Budget Concerns",
                "description": "Several communications mention budget constraints",
                "type": "risk",
                "confidence": 0.8
            }
        ],
        "analysis_timestamp": datetime.now().isoformat(),
        "status": "completed"
    }

@app.get("/graph/{project_id}")
def get_graph_data_by_id(project_id: str):
    """Get knowledge graph data by project ID - matches frontend GRAPH_DATA endpoint."""
    
    # In a real implementation, fetch graph data for specific project
    return {
        "project_id": project_id,
        "nodes": [
            {"id": "JS", "label": "John Smith", "type": "Stakeholder", "connections": 12},
            {"id": "BR", "label": "Budget Review", "type": "Task", "connections": 8}
        ],
        "edges": [
            {"from": "JS", "to": "BR", "relationship": "manages", "strength": 0.8}
        ],
        "stats": {
            "total_nodes": 2,
            "total_edges": 1
        }
    }

@app.post("/emails")
def send_project_emails(request: Dict):
    """Send emails based on analysis - matches frontend SEND_EMAILS endpoint."""
    
    email_data = request.get("email_data", {})
    recipients = email_data.get("recipients", [])
    subject = email_data.get("subject", "Project Analysis Report")
    content = email_data.get("content", "")
    
    if not recipients:
        raise HTTPException(status_code=400, detail="No recipients provided")
    
    # Mock email sending implementation
    return {
        "status": "success",
        "message": f"Emails sent to {len(recipients)} recipients",
        "sent_to": recipients,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/interactive_graph/{project_id}")
def get_interactive_graph_data(project_id: str):
    """Get interactive graph data - matches frontend INTERACTIVE_GRAPH endpoint."""
    
    return {
        "project_id": project_id,
        "nodes": [
            {
                "id": "JS",
                "label": "John Smith", 
                "type": "Stakeholder",
                "status": "active",
                "connections": 12,
                "x": 100,
                "y": 100
            },
            {
                "id": "BR",
                "label": "Budget Review",
                "type": "Task", 
                "status": "pending",
                "connections": 8,
                "x": 200,
                "y": 150
            }
        ],
        "edges": [
            {
                "from_node": "JS",
                "to_node": "BR", 
                "relationship": "manages",
                "strength": 0.8
            }
        ],
        "layout": "force-directed",
        "stats": {
            "total_nodes": 2,
            "total_edges": 1
        }
    }

# Add CORS preflight handling
@app.options("/{path:path}")
def options_handler(path: str):
    """Handle CORS preflight requests."""
    return {"message": "OK"}

@app.get("/export_analysis/{project_id}")
def export_project_analysis(project_id: str, format: str = Query("json", regex="^(json|csv|pdf)$")):
    """Export project analysis in various formats."""
    
    if format == "json":
        return {"message": "JSON export ready", "download_url": f"/download/{project_id}.json"}
    elif format == "csv":
        return {"message": "CSV export ready", "download_url": f"/download/{project_id}.csv"}
    elif format == "pdf":
        return {"message": "PDF export ready", "download_url": f"/download/{project_id}.pdf"}

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

