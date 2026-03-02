# SHL Assessment Recommendation System

This project is an intelligent recommendation system designed to help recruiters find relevant SHL assessments based on job requirements (Natural Language queries, Job Descriptions, or URLs).

## 🚀 Key Features

- **Intelligent Search**: Semantically understands job requirements beyond simple keyword matching.
- **Web Crawler**: Automatically extracts over 500+ assessments from the SHL official catalog.
- **Hybrid Ranking**: Combines semantic similarity (60%), industry-aligned skill matching (25%), and assessment type diversity (15%).
- **LLM Integration**: Uses GPT-4o for expert-level skill extraction and providing natural language explanations for matches.
- **Production-Ready API**: Built with FastAPI for high performance and standard compliance.
- **Interactive UI**: User-friendly Streamlit interface for exploring recommendations.

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **NLP/Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`), Scikit-learn
- **Data Acquisition**: BeautifulSoup4, Requests
- **LLM**: OpenAI GPT-4o

## 📋 Prerequisites

- Python 3.9+
- OpenAI API Key (optional, for enhanced skill extraction and explanations)

## 🔧 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**:
   ```bash
   set OPENAI_API_KEY=your_api_key_here
   ```

4. **Prepare the Data**:
   Ensure `shl_catalogue.csv` is present. If not, run the crawler:
   ```bash
   python crawler.py
   ```

## 🏃 Running the Application

1. **Start the FastAPI Backend**:
   ```bash
   python main.py
   ```

2. **Launch the Streamlit Frontend**:
   ```bash
   streamlit run app.py
   ```

3. **Access the UI**:
   Open `http://localhost:8501` in your browser.

## 📊 Evaluation & Predictions

- To run performance metrics (Recall@10):
  ```bash
  python evaluate.py
  ```
- To generate final predictions for the test set:
  ```bash
  python predict.py
  ```

## 🏗️ Architecture

1. **DATA LOADING**: Parses the SHL catalogue and local training datasets.
2. **EMBEDDINGS**: Uses Sentence Transformers to create vector representations of assessments and queries.
3. **RETRIEVAL**: Performs cosine similarity search across the vector space.
4. **RANKING**: Applies a hybrid scoring formula and diversity logic to balance Knowledge (K) and Personality (P) tests.
5. **RECOMMENDATION**: Returns the top 10 ranked assessments with optional LLM-generated explanations.
