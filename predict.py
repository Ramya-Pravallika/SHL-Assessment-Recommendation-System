import pandas as pd
import logging
from recommender import SHLRecommender
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_predictions(recommender, test_df, output_file='predictions.csv'):
    all_predictions = []
    
    logging.info(f"Generating predictions for {len(test_df)} queries...")
    
    for idx, row in test_df.iterrows():
        query = row['Query']
        # Get top 10 recommendations
        recommendations = recommender.recommend(query, top_n=10)
        
        for rec in recommendations:
            all_predictions.append({
                'Query': query,
                'Assessment_url': rec['assessment_url']
            })
            
        if (idx + 1) % 5 == 0:
            logging.info(f"Processed {idx + 1}/{len(test_df)} queries")
            
    df_preds = pd.DataFrame(all_predictions)
    df_preds.to_csv(output_file, index=False)
    logging.info(f"Saved predictions to {output_file}")

if __name__ == "__main__":
    # Load test data
    dataset_path = 'Gen_AI Dataset.xlsx'
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset {dataset_path} not found.")
        exit(1)
        
    df_test = pd.read_excel(dataset_path, sheet_name='Test-Set')
    
    # Initialize recommender
    recommender = SHLRecommender()
    if recommender.load_data():
        recommender.generate_embeddings()
        
        # Run prediction
        generate_predictions(recommender, df_test)
    else:
        logging.error("Could not run predictions. Catalogue data missing.")
