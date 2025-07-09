import json
from groq import Groq

def evaluate_performance(metrics_file):
    # Load metrics
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    # Initialize LLM client
    client = Groq()

    # Construct LLM prompt
    prompt = f"""
    Analyze the following backtest metrics and decide if the strategy is viable. 
    Provide suggestions for improvement if necessary:
    {json.dumps(metrics, indent=4)}
    """

    # LLM analysis
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1024,
    )
    
    analysis = response["choices"][0]["message"]["content"]
    print("LLM Analysis:")
    print(analysis)

    return analysis

if __name__ == "__main__":
    evaluate_performance("backtest_metrics.json")
