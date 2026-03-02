import pandas as pd
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SHLRecommender:
    def __init__(self, catalogue_path='shl_catalogue.csv', model_name='all-MiniLM-L6-v2', llm_utils=None):
        self.catalogue_path = catalogue_path
        self.model = SentenceTransformer(model_name)
        self.catalogue_df = None
        self.embeddings = None
        self.llm_utils = llm_utils
        
    def load_data(self):
        if not os.path.exists(self.catalogue_path):
            if os.path.exists('shl_catalogue_partial.csv'):
                self.catalogue_path = 'shl_catalogue_partial.csv'
                logging.info("Main catalogue missing. Using partial catalogue.")
            else:
                logging.error(f"Catalogue file {self.catalogue_path} not found.")
                return False
        
        self.catalogue_df = pd.read_csv(self.catalogue_path)
        # Handle missing descriptions
        self.catalogue_df['description'] = self.catalogue_df['description'].fillna('')
        
        # Combine fields for embedding
        self.catalogue_df['combined_text'] = (
            self.catalogue_df['assessment_name'] + " " + 
            self.catalogue_df['description'] + " " + 
            self.catalogue_df['test_type']
        )
        
        logging.info(f"Loaded {len(self.catalogue_df)} assessments from catalogue.")
        return True

    def generate_embeddings(self):
        if self.catalogue_df is None:
            logging.error("Catalogue data not loaded.")
            return
        
        logging.info("Generating embeddings for catalogue...")
        self.embeddings = self.model.encode(self.catalogue_df['combined_text'].tolist(), show_progress_bar=True)
        logging.info("Embeddings generated successfully.")

    def preprocess_query(self, query):
        # Basic cleanup
        query = query.strip()
        # You could add URL fetching logic here if the query is a URL
        return query

    def extract_skills(self, text):
        # Simplified skill extraction for now
        # In a full implementation, this would use an LLM
        skills = re.findall(r'\b[A-Za-z0-9#+.]+\b', text.lower())
        return set(skills)

    def calculate_skill_overlap(self, query_skills, doc_skills):
        if not query_skills:
            return 0
        intersection = query_skills.intersection(doc_skills)
        return len(intersection) / len(query_skills)

    def recommend(self, query, top_n=10):
        if self.embeddings is None:
            self.generate_embeddings()
            
        processed_query = self.preprocess_query(query)
        query_embedding = self.model.encode([processed_query])
        
        # Semantic Similarity
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        
        # Use LLM for query skills if available, else regex
        if self.llm_utils:
            res = self.llm_utils.extract_skills_and_intent(processed_query)
            query_skills = set([s.lower() for s in res.get('skills', [])])
        else:
            query_skills = self.extract_skills(processed_query)
        
        results = []
        for idx, score in enumerate(similarities):
            # Skill overlap
            doc_text = self.catalogue_df.iloc[idx]['combined_text']
            doc_skills = self.extract_skills(doc_text)
            skill_overlap = self.calculate_skill_overlap(query_skills, doc_skills)
            
            # Hybrid Ranking
            # Base score: 0.6 * similarity + 0.25 * skill_overlap
            # We'll handle diversity separately after getting base scores
            base_score = (0.6 * score) + (0.25 * skill_overlap)
            
            results.append({
                'assessment_name': self.catalogue_df.iloc[idx]['assessment_name'],
                'assessment_url': self.catalogue_df.iloc[idx]['assessment_url'],
                'base_score': float(base_score),
                'test_type': self.catalogue_df.iloc[idx]['test_type']
            })
            
        # Sort by base score
        results = sorted(results, key=lambda x: x['base_score'], reverse=True)
        
        # Diversity Logic: Ensure at least 2 of each type in top 10 if available
        final_results = results[:top_n]
        types_in_top = [r['test_type'] for r in final_results]
        
        if len(set(types_in_top)) < 2:
            # We are missing one type. Try to find the best of the missing type.
            missing_type = 'P' if types_in_top[0] == 'K' else 'K'
            remaining_pool = results[top_n:]
            missing_type_candidates = [r for r in remaining_pool if r['test_type'] == missing_type]
            
            if missing_type_candidates:
                # Replace the last few items with the best of the missing type for diversity
                # Only if they are reasonably similar (e.g. within 80% of the last item's score)
                last_item_score = final_results[-1]['base_score']
                for i in range(min(2, len(missing_type_candidates))):
                    candidate = missing_type_candidates[i]
                    if candidate['base_score'] > last_item_score * 0.5: # Relaxed threshold for diversity
                        final_results[-(i+1)] = candidate
        
        # Final scores (adding diversity weight for display)
        for r in final_results:
            r['score'] = r['base_score'] + 0.15 # Diversity weight constant for top results
            
        return sorted(final_results, key=lambda x: x['score'], reverse=True)

if __name__ == "__main__":
    # Test initialization
    recommender = SHLRecommender()
    if recommender.load_data():
        recommender.generate_embeddings()
        test_query = "Looking for Java developers with SQL skills"
        recommendations = recommender.recommend(test_query)
        for rec in recommendations:
            print(rec)
