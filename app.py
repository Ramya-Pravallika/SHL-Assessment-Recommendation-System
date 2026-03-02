import streamlit as st
import requests
import pandas as pd
import time

# Page Config
st.set_page_config(page_title="SHL Assessment Recommender", page_icon="📈", layout="wide")

# Title and Description
st.title("SHL Assessment Recommendation System")
st.markdown("""
Enter a natural language query, job description text, or a URL to get the most relevant SHL assessments.
""")

# API Configuration
API_URL = "http://localhost:8000/recommend"

# Sidebar
with st.sidebar:
    st.header("Settings")
    top_n = st.slider("Number of recommendations", 5, 20, 10)
    st.info("The system uses a hybrid ranking algorithm combining semantic similarity and skill matching.")

# Input Section
query = st.text_area("Hiring Requirements / Job Description / URL", height=200, placeholder="e.g., Hiring for a Java developer with AWS experience...")

if st.button("Run Recommendation", type="primary"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing your request..."):
            try:
                # Call FastAPI backend
                response = requests.post(API_URL, json={"query": query})
                
                if response.status_code == 200:
                    results = response.json()
                    
                    if not results:
                        st.info("No assessments found matching your criteria.")
                    else:
                        st.success(f"Found {len(results)} relevant assessments!")
                        
                        # Convert to DataFrame for display
                        df = pd.DataFrame(results)
                        
                        # Clean up DataFrame for display
                        display_df = df[['assessment_name', 'assessment_url', 'score', 'test_type']]
                        display_df.columns = ['Assessment Name', 'URL', 'Score', 'Test Type']
                        
                        # Display Results Table
                        st.dataframe(
                            display_df,
                            column_config={
                                "URL": st.column_config.LinkColumn("Assessment Link")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Detailed View with Explanations
                        st.divider()
                        st.subheader("Top Recommendations")
                        for i, item in enumerate(results[:3]):
                            with st.expander(f"{i+1}. {item['assessment_name']} (Score: {item['score']:.4f})"):
                                st.write(f"**Test Type:** {'Knowledge & Skills (K)' if item['test_type'] == 'K' else 'Personality & Behavior (P)'}")
                                if item.get('explanation'):
                                    st.write(f"**Why this match?** {item['explanation']}")
                                st.write(f"[Open Assessment Catalog Page]({item['assessment_url']})")
                else:
                    st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")

# Footer
st.divider()
st.caption("SHL GenAI Assessment - Intelligent Recommendation System")
