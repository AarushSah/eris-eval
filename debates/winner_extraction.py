import json
import os
from openai import OpenAI
import re
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up the OpenAI client with OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="OPENROUTER_API_KEY"
)

def extract_winner(evaluation):
    prompt = f"""Extract the winner from this debate evaluation. 
    Respond only with <winner>Affirmative</winner> or <winner>Negative</winner>:

    {evaluation}"""
    
    try:
        response = client.chat.completions.create(
            model="anthropic/claude-3-haiku-20240307",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts information from text."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract text between <winner> tags
        match = re.search(r'<winner>(.*?)</winner>', content)
        if match:
            return match.group(1)
        else:
            logging.warning(f"No winner tag found in response: {content}")
            return "Unknown"
    except Exception as e:
        logging.error(f"Error in API call: {str(e)}")
        return "Error"

# Get all JSON files in the current directory
json_files = [f for f in os.listdir('.') if f.endswith('.json')]

# Process all JSON files with a progress bar
for filename in tqdm(json_files, desc="Processing files"):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        if 'evaluation' in data:
            winner = extract_winner(data['evaluation'])
            data['winner'] = winner
            logging.info(f"Extracted winner for {filename}: {winner}")
        else:
            logging.warning(f"No 'evaluation' field found in {filename}")
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON in file: {filename}")
    except Exception as e:
        logging.error(f"Error processing file {filename}: {str(e)}")

print("Processing complete.")