import subprocess

def run_backtest():
    """
    Run the backtest strategy script.
    """
    print("Running backtest...")
    result = subprocess.run(["python", "backtest_strategy.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error during backtest:")
        print(result.stderr)
        return False
    print("Backtest completed.")
    return True

def evaluate_strategy():
    """
    Evaluate the strategy using the LLM decision script.
    """
    print("Evaluating strategy...")
    result = subprocess.run(
        ["python", "evaluate_strategy.py"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Error during evaluation:")
        print(result.stderr)
        return None

    print("LLM Analysis:")
    print(result.stdout)
    return result.stdout

def save_to_database():
    """
    Save the results of a successful strategy to the database.
    """
    print("Saving results to the database...")
    result = subprocess.run(["python", "save_results.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error while saving to the database:")
        print(result.stderr)
    else:
        print("Results successfully saved to the database.")

def generate_new_strategy():
    """
    Generate a new strategy using the LLM.
    """
    print("Generating a new strategy...")
    result = subprocess.run(["python", "llm_strategy_generator.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error during strategy generation:")
        print(result.stderr)
    else:
        print("New strategy generated.")

def feedback_loop():
    """
    Automates the feedback loop:
    1. Runs the backtest.
    2. Evaluates the strategy using the LLM.
    3. Either saves the strategy or generates a new one based on LLM feedback.
    """
    if not run_backtest():
        return

    analysis = evaluate_strategy()
    if analysis is None:
        return

    # Simple check for approval. Adjust logic based on LLM response patterns.
    if "viable" in analysis.lower():
        save_to_database()
    else:
        print("Strategy needs improvement. Generating a new strategy...")
        generate_new_strategy()

if __name__ == "__main__":
    feedback_loop()
