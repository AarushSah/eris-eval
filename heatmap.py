import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore, Style

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

def load_data(directory):
    data = {model: {'wins': 0, 'losses': 0, 'debates': []} for model in MODELS}
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                debate = json.load(f)
                affirmative = debate['affirmative_model']
                negative = debate['negative_model']
                winner = debate['winner']

                if winner == 'Affirmative':
                    data[affirmative]['wins'] += 1
                    data[negative]['losses'] += 1
                elif winner == 'Negative':
                    data[negative]['wins'] += 1
                    data[affirmative]['losses'] += 1

                data[affirmative]['debates'].append({'opponent': negative, 'role': 'Affirmative', 'winner': winner})
                data[negative]['debates'].append({'opponent': affirmative, 'role': 'Negative', 'winner': winner})

    for model in MODELS:
        data[model]['total_debates'] = data[model]['wins'] + data[model]['losses']
        data[model]['win_rate'] = data[model]['wins'] / data[model]['total_debates'] if data[model]['total_debates'] > 0 else 0

    return data

def calculate_win_rates(data):
    df = pd.DataFrame(index=MODELS, columns=MODELS, dtype=float)
    model_performance = []

    for model in MODELS:
        wins = data[model]['wins']
        losses = data[model]['losses']
        total_debates = data[model]['total_debates']
        win_rate = data[model]['win_rate']
        model_performance.append((model, win_rate, wins, losses, total_debates))
        
        for opponent in MODELS:
            if model != opponent:
                model_wins = sum(1 for debate in data[model]['debates'] 
                                 if debate['opponent'] == opponent and debate['winner'] == debate['role'])
                total_debates = sum(1 for debate in data[model]['debates'] 
                                    if debate['opponent'] == opponent)
                win_rate = model_wins / total_debates if total_debates > 0 else 0
                df.loc[model, opponent] = win_rate

    np.fill_diagonal(df.values, 0.5)
    df = df.fillna(0)
    return df, model_performance

def print_model_performance(model_performance):
    for model, win_rate, wins, losses, total_debates in sorted(model_performance, key=lambda x: x[1], reverse=True):
        print(f"{model}: {Fore.GREEN}{win_rate:.2f}{Style.RESET_ALL} ({wins} wins, {losses} losses)")

def create_heatmap(df, title, output_file):
    overall_win_rate = df.mean(axis=1)
    sorted_index = overall_win_rate.sort_values(ascending=False).index
    df_sorted = df.loc[sorted_index, sorted_index]

    plt.figure(figsize=(14, 12))
    sns.heatmap(df_sorted, annot=True, cmap='RdYlGn', vmin=0, vmax=1, center=0.5, fmt='.2f', 
                cbar_kws={'label': 'Win Rate'})

    plt.title(title, fontsize=16)
    plt.xlabel('Opponent Model', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sorted win rate heatmap has been saved as '{output_file}'")

def main():
    directory = './debates/'  # Change this to the directory containing your JSON files
    data = load_data(directory)
    df, model_performance = calculate_win_rates(data)
    print_model_performance(model_performance)
    create_heatmap(df, 'Debate Win Rates', 'debate_win_rates_heatmap.png')

if __name__ == "__main__":
    main()