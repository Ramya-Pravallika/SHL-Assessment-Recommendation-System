from recommender import SHLRecommender
from llm_utils import LLMUtils
import os

# Page Config
st.set_page_config(page_title="SHL Assessment Recommender", page_icon="📈", layout="wide")

# Initialize Recommender and LLM
@st.cache_resource
def get_recommender():
    llm_utils = LLMUtils()
    recommender = SHLRecommender(llm_utils=llm_utils)
    if recommender.load_data():
        recommender.generate_embeddings()
    return recommender, llm_utils

recommender, llm_utils = get_recommender()

# Title and Description
st.title("SHL Assessment Recommendation System")
st.markdown("""
Enter a natural language query, job description text, or a URL to get the most relevant SHL assessments.
""")

# Sidebar
with st.sidebar:
    st.header("Settings")
    top_n = st.slider("Number of recommendations", 5, 20, 10)
    st.info("The system uses a hybrid ranking algorithm combining semantic similarity and skill matching.")
    
    # OpenAI Key Warning
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY environment variable not set. Explanations and advanced skill extraction are disabled.")

# Input Section
query = st.text_area("Hiring Requirements / Job Description / URL", height=200, placeholder="e.g., Hiring for a Java developer with AWS experience...")

if st.button("Run Recommendation", type="primary"):
    if not query:
        st.warning("Please enter a query.")
    elif recommender.catalogue_df is None:
        st.error("Recommender system failed to initialize. Please check if shl_catalogue.csv exists.")
    else:
        with st.spinner("Processing your request..."):
            try:
                # Get recommendations directly
                results = recommender.recommend(query, top_n=top_n)
                
                if not results:
                    st.info("No assessments found matching your criteria.")
                else:
                    # Add LLM explanations for top 3 results if LLM is enabled
                    if llm_utils.client:
                        for i in range(min(3, len(results))):
                            item = results[i]
                            # Find original description
                            desc_matches = recommender.catalogue_df[recommender.catalogue_df['assessment_url'] == item['assessment_url']]['description'].values
                            desc = desc_matches[0] if len(desc_matches) > 0 else ""
                            item['explanation'] = llm_utils.generate_explanation(query, item['assessment_name'], desc)
                    
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
            except Exception as e:
                st.error(f"Error processing recommendation: {e}")

# Footer
st.divider()
st.caption("SHL GenAI Assessment - Intelligent Recommendation System")
