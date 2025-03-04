"""
LLM API utility module for enhanced insights.
"""
import os
import logging
from dotenv import load_dotenv
import openai
import groq

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class LLMProvider:
    """
    Provider for LLM API interactions.
    """
    
    def __init__(self, provider="openai"):
        """
        Initialize the LLM provider.
        
        Args:
            provider (str): The LLM provider to use ('openai' or 'groq')
        """
        self.provider = provider.lower()
        
        # Set up API keys
        if self.provider == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        elif self.provider == "groq":
            groq.api_key = os.getenv("GROQ_API_KEY")
            if not groq.api_key:
                logger.warning("Groq API key not found. Set GROQ_API_KEY environment variable.")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate_insights(self, data_description, context=None, max_tokens=1000):
        """
        Generate insights from data description.
        
        Args:
            data_description (str): Description of the data
            context (str, optional): Additional context
            max_tokens (int, optional): Maximum tokens in response
            
        Returns:
            str: Generated insights
        """
        prompt = self._create_insight_prompt(data_description, context)
        
        try:
            if self.provider == "openai":
                return self._call_openai(prompt, max_tokens)
            elif self.provider == "groq":
                return self._call_groq(prompt, max_tokens)
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return "Error generating insights. Please check API configuration."
    
    def explain_model_results(self, model_info, data_info, max_tokens=1000):
        """
        Generate natural language explanation of model results.
        
        Args:
            model_info (dict): Information about the model and its performance
            data_info (dict): Information about the dataset
            max_tokens (int, optional): Maximum tokens in response
            
        Returns:
            str: Natural language explanation
        """
        prompt = self._create_model_explanation_prompt(model_info, data_info)
        
        try:
            if self.provider == "openai":
                return self._call_openai(prompt, max_tokens)
            elif self.provider == "groq":
                return self._call_groq(prompt, max_tokens)
        except Exception as e:
            logger.error(f"Error explaining model results: {str(e)}")
            return "Error generating model explanation. Please check API configuration."
    
    def suggest_next_steps(self, workflow_results, max_tokens=1000):
        """
        Suggest next steps based on workflow results.
        
        Args:
            workflow_results (dict): Results from the workflow
            max_tokens (int, optional): Maximum tokens in response
            
        Returns:
            str: Suggested next steps
        """
        prompt = self._create_next_steps_prompt(workflow_results)
        
        try:
            if self.provider == "openai":
                return self._call_openai(prompt, max_tokens)
            elif self.provider == "groq":
                return self._call_groq(prompt, max_tokens)
        except Exception as e:
            logger.error(f"Error suggesting next steps: {str(e)}")
            return "Error generating next steps. Please check API configuration."
    
    def _call_openai(self, prompt, max_tokens):
        """Call OpenAI API."""
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a data science expert providing insights and explanations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def _call_groq(self, prompt, max_tokens=1000):
        """Call Groq API."""
        try:
            # The correct way to use the Groq client
            client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a data science expert providing insights and explanations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            return f"Error generating insights with Groq: {str(e)}"
    
    def _create_insight_prompt(self, data_description, context):
        """Create prompt for generating insights."""
        prompt = f"""
        I need you to analyze this dataset and provide valuable insights:
        
        Dataset Description:
        {data_description}
        """
        
        if context:
            prompt += f"""
            Additional Context:
            {context}
            """
        
        prompt += """
        Please provide:
        1. Key patterns or trends you notice in the data
        2. Potential business implications of these patterns
        3. Suggestions for further analysis
        4. Any concerns about data quality or potential biases
        
        Format your response in clear, concise language that would be understandable to non-technical stakeholders.
        """
        
        return prompt
    
    def _create_model_explanation_prompt(self, model_info, data_info):
        """Create prompt for explaining model results."""
        prompt = f"""
        I need you to explain these machine learning model results in plain language:
        
        Model Information:
        {model_info}
        
        Dataset Information:
        {data_info}
        
        Please provide:
        1. A clear explanation of what the model is predicting and how well it's performing
        2. The key factors influencing the model's predictions
        3. The practical implications of these results
        4. Potential limitations or caveats
        
        Format your response in clear, non-technical language that explains the value and limitations of this model.
        """
        
        return prompt
    
    def _create_next_steps_prompt(self, workflow_results):
        """Create prompt for suggesting next steps."""
        prompt = f"""
        Based on these data science workflow results, I need recommendations for next steps:
        
        Workflow Results:
        {workflow_results}
        
        Please suggest:
        1. Specific actions to improve model performance
        2. Additional analyses that could yield valuable insights
        3. How to operationalize these findings
        4. Potential business applications of the results
        
        Format your response as actionable recommendations with clear rationales.
        """
        
        return prompt 