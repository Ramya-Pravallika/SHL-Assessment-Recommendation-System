import openai
import logging
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMUtils:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logging.warning("OpenAI API Key not found. LLM features will be disabled or fall back to regex.")
            self.client = None
        else:
            self.client = openai.OpenAI(api_key=self.api_key)

    def extract_skills_and_intent(self, text):
        if not self.client:
            # Fallback to simple logic if LLM is unavailable
            return {"skills": [], "intent": "general"}
        
        try:
            prompt = f"""
            Extract a list of technical and soft skills, and the primary hiring intent from the following text (Job Description or Query).
            Return the result in JSON format with keys 'skills' and 'intent'.
            
            Text: {text}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logging.info(f"LLM Extracted: {result}")
            return result
        except Exception as e:
            logging.error(f"Error in LLM extraction: {e}")
            return {"skills": [], "intent": "error"}

    def generate_explanation(self, query, assessment_name, description):
        if not self.client:
            return "Recommendation based on semantic similarity and skill matching."
            
        try:
            prompt = f"""
            Briefly explain why the assessment '{assessment_name}' is a good match for the query: '{query}'.
            Use the assessment description for context: '{description}'
            Keep the explanation to 2 sentences max.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error in LLM explanation: {e}")
            return "Recommended based on job requirements."

if __name__ == "__main__":
    # Test
    utils = LLMUtils()
    res = utils.extract_skills_and_intent("Looking for a Senior Python Developer with experience in Django, React, and AWS. Must have good communication skills.")
    print(res)
