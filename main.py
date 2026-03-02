from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
from recommender import SHLRecommender
from llm_utils import LLMUtils
import uvicorn
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="SHL Assessment Recommendation API")

# Initialize components
llm_utils = LLMUtils()
recommender = SHLRecommender(llm_utils=llm_utils)

# Load data on startup
@app.on_event("startup")
async def startup_event():
    if recommender.load_data():
        recommender.generate_embeddings()
    else:
        logging.error("Failed to load catalogue data. API will not be functional.")

class RecommendRequest(BaseModel):
    query: str

class RecommendationResponse(BaseModel):
    assessment_name: str
    assessment_url: str
    score: float
    test_type: str
    explanation: Optional[str] = None

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommend", response_model=List[RecommendationResponse])
async def recommend(request: RecommendRequest):
    if recommender.catalogue_df is None:
        raise HTTPException(status_code=503, detail="System not initialized. Catalogue data missing.")
    
    try:
        # Get recommendations
        results = recommender.recommend(request.query)
        
        # Add LLM explanations for top 3 results if LLM is enabled
        if llm_utils.client:
            for i in range(min(3, len(results))):
                item = results[i]
                # Find original description
                desc = recommender.catalogue_df[recommender.catalogue_df['assessment_url'] == item['assessment_url']]['description'].values[0]
                item['explanation'] = llm_utils.generate_explanation(request.query, item['assessment_name'], desc)
        
        return results
    except Exception as e:
        logging.error(f"Error processing recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
