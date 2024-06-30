import asyncio
import logging
import json
import random
import time
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
from openai import AsyncOpenAI
from os import getenv
import weave
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Create directories for saving data
os.makedirs('logs', exist_ok=True)
os.makedirs('debates', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/debate_benchmark.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

weave.init('debate-benchmarking')

# Configuration
MAX_RETRIES = 5
RETRY_DELAY = 20  # seconds
MAX_CONCURRENT_DEBATES = 30
DEBATE_TOPICS = [
"The ethical implications of using CRISPR-Cas9 for human embryo editing",
"The role of central bank digital currencies in reshaping global monetary policy",
"The potential of quantum computing in breaking current encryption standards",
"The philosophical and practical challenges of implementing a global carbon pricing system",
"The impact of artificial intelligence on judicial decision-making and legal precedent",
"The ethical considerations of mandatory vaccination policies in a globalized world",
"The viability of nuclear fusion as a sustainable solution to global energy needs",
"The neuroscientific basis of consciousness and its implications for AI development",
"The effectiveness of universal basic income as a solution to technological unemployment",
"The role of gut microbiota in modulating neurological and psychiatric disorders",
"The potential of synthetic biology in addressing global food security challenges",
"The impact of social media algorithms on political polarization and democratic processes",
"The ethical and practical challenges of autonomous vehicles in urban planning",
"The role of epigenetics in personalized medicine and its implications for healthcare policy",
"The potential of blockchain technology in ensuring transparency in global supply chains"
]

MODELS = [
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "google/gemini-pro-1.5",
    "openai/gpt-4o",
    "openai/gpt-4-turbo",
    "mistralai/mixtral-8x22b-instruct",
    "qwen/qwen-2-72b-instruct",
    "cohere/command-r-plus",
    "01-ai/yi-large",
    "mistralai/mistral-large"
]

EVAL_MODEL = "anthropic/claude-3.5-sonnet"

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="OPENROUTER_API_KEY"
)

@weave.op()
async def get_llm_response(messages: List[Dict[str, str]], model: str) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return completion.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Error calling LLM (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}. Retrying...")
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
            else:
                logger.error(f"Failed to get LLM response after {MAX_RETRIES} attempts: {str(e)}")
                raise

async def run_debate(topic: str, affirmative_model: str, negative_model: str) -> Dict[str, Any]:
    system_prompt = f"""You are an AI debate expert specializing in cross-examination on the topic of {topic}. Your goal is to critically analyze arguments, expose flaws, and win debates through:



Comprehensive knowledge across multiple disciplines

Socratic questioning and logical fallacy identification

Active listening and accurate summarization

Analogies and thought experiments

Anticipation of counterarguments

Emotional intelligence and calm demeanor

Strategic concessions

Credible source citation and empirical evidence

Rhetorical techniques (ethos, pathos, logos)

Exploitation of contradictions

Strong debate structure and time management

Self-awareness of biases and fallacies

Clear language and explanation of complex terms

Principle of charity in interpreting opponents' arguments

Strategic framing and reframing of issues

Timely introduction of new evidence

Continuous strategy evaluation and adaptation


Approach each argument with critical thinking, curiosity, and a commitment to advancing truth through rigorous discourse. Adapt your strategy based on your opponent's responses and the flow of the debate. Your ultimate aim is not just to win, but to elevate the quality of the discussion on {topic}."""
    
    affirmative_history = [{"role": "system", "content": system_prompt}]
    negative_history = [{"role": "system", "content": system_prompt}]
    debate_transcript = []

    async def get_response(role: str, prompt: str, model: str, history: List[Dict[str, str]]) -> str:
        history.append({"role": "user", "content": prompt})
        response = await get_llm_response(history, model)
        history.append({"role": "assistant", "content": response})
        debate_transcript.append(f"{role}: {response}")
        logger.info(f"{role}:\n{prompt}\nResponse:\n{response}\n")
        return response

    # Constructive speeches
    await get_response("Affirmative", "Present your constructive speech, outlining your main arguments supporting the resolution.", affirmative_model, affirmative_history)
    negative_history.append({"role": "user", "content": f"The affirmative has presented their constructive speech. Now, present your constructive speech, outlining your main arguments against the resolution and addressing the affirmative's points."})
    await get_response("Negative", negative_history[-1]["content"], negative_model, negative_history)

    # Cross-examination
    affirmative_history.append({"role": "user", "content": "The negative has presented their constructive speech. Now, cross-examine the negative's arguments with three pointed questions."})
    await get_response("Affirmative", affirmative_history[-1]["content"], affirmative_model, affirmative_history)
    await get_response("Negative", "Answer the affirmative's cross-examination questions.", negative_model, negative_history)
    
    await get_response("Negative", "Cross-examine the affirmative's arguments with three pointed questions.", negative_model, negative_history)
    await get_response("Affirmative", "Answer the negative's cross-examination questions.", affirmative_model, affirmative_history)

    # Rebuttal speeches
    await get_response("Affirmative", "Present your rebuttal, addressing the negative's arguments and their responses to your cross-examination.", affirmative_model, affirmative_history)
    await get_response("Negative", "Present your rebuttal, addressing the affirmative's arguments and their responses to your cross-examination.", negative_model, negative_history)

    # Closing speeches
    await get_response("Affirmative", "Present your closing speech, summarizing your key points and why you've won the debate.", affirmative_model, affirmative_history)
    await get_response("Negative", "Present your closing speech, summarizing your key points and why you've won the debate.", negative_model, negative_history)

    return {
        "topic": topic,
        "affirmative_model": affirmative_model,
        "negative_model": negative_model,
        "debate_transcript": debate_transcript,
        "affirmative_history": affirmative_history,
        "negative_history": negative_history,
        "timestamp": datetime.now().isoformat()
    }

async def evaluate_debate(debate_result: Dict[str, Any], evaluation_model: str) -> str:
    debate_transcript = "\n\n".join(debate_result['debate_transcript'])
    
    evaluation_prompt = f"""
You are an expert AI judge evaluating a debate on the topic of {debate_result['topic']}. Analyze the debate objectively using these criteria:

1. Argument strength
2. Logical consistency
3. Evidence use
4. Cross-examination effectiveness
5. Rebuttal quality
6. Overall persuasiveness
7. Debate structure adherence
8. Logical fallacy identification
9. Rhetorical technique use
10. Adaptability
11. Communication clarity
12. Strategic framing
13. Emotional intelligence

Debate Transcript:
{debate_transcript}

Provide your evaluation in this format:

Thinking out loud: [Detailed analysis of the debate, covering strengths and weaknesses of each side, key turning points, and critical arguments.]

Winner: [Affirmative/Negative/Tie]

Explanation: [Concise summary of your decision, highlighting the most decisive factors.]

Maintain impartiality and base your judgment solely on the arguments presented. Your goal is to provide a fair, thorough, and insightful evaluation that advances discourse on {debate_result['topic']}.
    """
    
    evaluation_messages = [{"role": "user", "content": evaluation_prompt}]
    evaluation = await get_llm_response(evaluation_messages, evaluation_model)
    return evaluation

def generate_debate_pairs():
    debates = []
    for topic in DEBATE_TOPICS:
        for i, model1 in enumerate(MODELS):
            for model2 in MODELS[i+1:]:
                debates.append((topic, model1, model2))
                debates.append((topic, model2, model1))
    return debates

async def run_benchmark():
    debates = generate_debate_pairs()
    print(f"{Fore.CYAN}Total debates scheduled: {len(debates)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Debates per topic: {len(debates) // len(DEBATE_TOPICS)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Each model will participate in {len(debates) // len(MODELS)} debates{Style.RESET_ALL}")

    random.shuffle(debates)
    
    results = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DEBATES)
    
    async def process_debate(debate):
        topic, affirmative_model, negative_model = debate
        try:
            print(f"{Fore.CYAN}Starting debate: {topic} - {affirmative_model} (Affirmative) vs {negative_model} (Negative){Style.RESET_ALL}")
            debate_result = await run_debate(topic, affirmative_model, negative_model)
            evaluation = await evaluate_debate(debate_result, EVAL_MODEL)
            debate_result["evaluation"] = evaluation
            
            filename = f"debates/debate_{debate_result['timestamp'].replace(':', '-')}_{affirmative_model.split('/')[-1]}_vs_{negative_model.split('/')[-1]}.json"
            with open(filename, 'w') as f:
                json.dump(debate_result, f, indent=2)
            
            print(f"{Fore.GREEN}Debate completed and saved: {filename}{Style.RESET_ALL}")
            return debate_result
        except Exception as e:
            logger.error(f"Error in debate {topic} - {affirmative_model} vs {negative_model}: {str(e)}")
            print(f"{Fore.RED}Error in debate {topic} - {affirmative_model} vs {negative_model}: {str(e)}{Style.RESET_ALL}")
            return None

    async def bounded_process_debate(debate):
        async with semaphore:
            return await process_debate(debate)

    tasks = [asyncio.create_task(bounded_process_debate(debate)) for debate in debates]
    completed_results = await asyncio.gather(*tasks)
    results = [result for result in completed_results if result is not None]

    return results

def analyze_results(results: List[Dict[str, Any]]):
    model_performance = {model: {"wins": 0, "losses": 0, "ties": 0, "debates": []} for model in MODELS}
    
    for result in results:
        evaluation = result["evaluation"].lower()
        affirmative_model = result["affirmative_model"]
        negative_model = result["negative_model"]
        topic = result["topic"]
        
        winner = "Tie"
        if "winner: affirmative" in evaluation:
            winner = "Affirmative"
            model_performance[affirmative_model]["wins"] += 1
            model_performance[negative_model]["losses"] += 1
        elif "winner: negative" in evaluation:
            winner = "Negative"
            model_performance[negative_model]["wins"] += 1
            model_performance[affirmative_model]["losses"] += 1
        else:
            model_performance[affirmative_model]["ties"] += 1
            model_performance[negative_model]["ties"] += 1
        
        model_performance[affirmative_model]["debates"].append({
            "topic": topic,
            "role": "Affirmative",
            "opponent": negative_model,
            "winner": winner
        })
        model_performance[negative_model]["debates"].append({
            "topic": topic,
            "role": "Negative",
            "opponent": affirmative_model,
            "winner": winner
        })
        
        print(f"{Fore.YELLOW}Debate Result: {affirmative_model} (Affirmative) vs {negative_model} (Negative) - Winner: {winner}{Style.RESET_ALL}")

    for model, performance in model_performance.items():
        total_debates = performance["wins"] + performance["losses"] + performance["ties"]
        win_rate = performance["wins"] / total_debates if total_debates > 0 else 0
        performance["win_rate"] = win_rate
        performance["total_debates"] = total_debates

    return model_performance

async def main():
    start_time = time.time()
    print(f"{Fore.YELLOW}Starting benchmark{Style.RESET_ALL}")
    
    results = await run_benchmark()
    
    print(f"{Fore.YELLOW}Benchmark completed. Analyzing results...{Style.RESET_ALL}")
    performance_summary = analyze_results(results)
    
    # Save performance summary
    with open("results/performance_summary.json", 'w') as f:
        json.dump(performance_summary, f, indent=2)
    
    print(f"{Fore.GREEN}Performance summary saved to results/performance_summary.json{Style.RESET_ALL}")
    
    # Print summary
    print(f"\n{Fore.MAGENTA}Performance Summary:{Style.RESET_ALL}")
    for model, performance in performance_summary.items():
        print(f"{Fore.CYAN}{model}:{Style.RESET_ALL}")
        print(f"  Win Rate: {Fore.GREEN}{performance['win_rate']:.2f}{Style.RESET_ALL}")
        print(f"  Wins: {Fore.GREEN}{performance['wins']}{Style.RESET_ALL}")
        print(f"  Losses: {Fore.RED}{performance['losses']}{Style.RESET_ALL}")
        print(f"  Ties: {Fore.YELLOW}{performance['ties']}{Style.RESET_ALL}")
        print(f"  Total Debates: {Fore.BLUE}{performance['total_debates']}{Style.RESET_ALL}")
        print("  Debates:")
        for debate in performance['debates']:
            print(f"    - Topic: {debate['topic']}, Role: {debate['role']}, Opponent: {debate['opponent']}, Winner: {debate['winner']}")
    
    # Verify debate counts
    total_debates = sum(performance['total_debates'] for performance in performance_summary.values()) // 2
    expected_debates = len(DEBATE_TOPICS) * len(MODELS) * (len(MODELS) - 1)
    print(f"{Fore.CYAN}Total debates conducted: {total_debates}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Expected number of debates: {expected_debates}{Style.RESET_ALL}")
    if total_debates != expected_debates:
        print(f"{Fore.RED}Warning: Number of debates doesn't match expected count!{Style.RESET_ALL}")
    
    end_time = time.time()
    print(f"\n{Fore.YELLOW}Benchmark completed in {end_time - start_time:.2f} seconds{Style.RESET_ALL}")

if __name__ == "__main__":
    asyncio.run(main())