import json
import sys
import os
from groq import Groq

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../core'))

try:
    from multi_llm_manager import MultiLLMManager
    USE_MULTI_LLM = True
except ImportError:
    USE_MULTI_LLM = False
    print("⚠️  Multi-LLM manager not available, using single model")

def evaluate_performance(metrics_file):
    # Load metrics
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    # Construct LLM prompt
    prompt = f"""
    Analyze the following backtest metrics and decide if the strategy is viable. 
    Provide suggestions for improvement if necessary:
    {json.dumps(metrics, indent=4)}
    """

    if USE_MULTI_LLM:
        # Use multi-LLM manager
        llm_manager = MultiLLMManager()
        selected_model = llm_manager.select_model("evaluation")
        print(f"🎯 Evaluating with model: {selected_model}")
        
        analysis, metadata = llm_manager.generate_with_model(prompt, selected_model)
        print("🤖 Multi-LLM Analysis:")
        print(analysis)
        
        return analysis
    else:
        # Fallback to single model
        client = Groq()
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1024,
        )
        
        analysis = response.choices[0].message.content
        print("LLM Analysis:")
        print(analysis)

        return analysis

if __name__ == "__main__":
    evaluate_performance("backtest_metrics.json")
