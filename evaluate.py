import pandas as pd
import logging
from recommender import SHLRecommender
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_recall_at_n(recommender, train_df, n=10):
    total_recall = 0
    queries = train_df['Query'].unique()
    
    logging.info(f"Starting evaluation on {len(queries)} unique queries...")
    
    for query in queries:
        # Get ground truth URLs for this query
        ground_truth_urls = {u.split('/view/')[-1].strip('/') for u in train_df[train_df['Query'] == query]['Assessment_url'].tolist()}
        
        # Get recommendations
        recommendations = recommender.recommend(query, top_n=n)
        rec_urls = {r['assessment_url'].split('/view/')[-1].strip('/') for r in recommendations}
        
        # Calculate recall for this query
        hits = ground_truth_urls.intersection(rec_urls)
        recall = len(hits) / len(ground_truth_urls) if ground_truth_urls else 0
        total_recall += recall
        
        logging.info(f"Query: {query[:50]}... | Recall@{n}: {recall:.4f}")
        
    mean_recall = total_recall / len(queries) if queries.any() else 0
    return mean_recall

if __name__ == "__main__":
    # Load training data
    dataset_path = 'Gen_AI Dataset.xlsx'
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset {dataset_path} not found.")
        exit(1)
        
    df_train = pd.read_excel(dataset_path, sheet_name='Train-Set')
    
    # Initialize recommender
    recommender = SHLRecommender()
    if recommender.load_data():
        recommender.generate_embeddings()
        
        # Run evaluation
        mean_recall_10 = evaluate_recall_at_n(recommender, df_train, n=10)
        
        print("\n" + "="*30)
        print(f"EVALUATION RESULTS")
        print(f"Mean Recall@10: {mean_recall_10:.4f}")
        print("="*30)
    else:
        logging.error("Could not run evaluation. Catalogue data missing.")
