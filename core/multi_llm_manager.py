import json
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from groq import Groq
import os
from dotenv import load_dotenv
try:
    from .enhanced_logger import logger
    from .q_learning_agent import QLearningModelSelector
except ImportError:
    from enhanced_logger import logger
    from q_learning_agent import QLearningModelSelector

load_dotenv()

class ModelPerformanceTracker:
    """Track performance metrics for each model"""
    
    def __init__(self):
        self.performance_file = "outputs/model_performance.json"
        self.model_stats = self._load_performance_data()
    
    def _load_performance_data(self) -> Dict:
        """Load existing performance data or initialize new"""
        try:
            if os.path.exists(self.performance_file):
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading performance data: {e}")
        
        # Initialize default performance tracking for all models
        default_stats = {
            "strategies_generated": 0,
            "successful_strategies": 0,
            "total_return": 0.0,
            "total_sharpe": 0.0,
            "best_strategy_return": 0.0,
            "avg_return": 0.0,
            "avg_sharpe": 0.0,
            "success_rate": 0.0
        }
        
        return {
            # Groq models only - simplified system
            "versatile": default_stats.copy(),
            "analytical": default_stats.copy(),
            "maverick": default_stats.copy(),
            "scout": default_stats.copy(),
            "diverse": default_stats.copy()
        }
    
    def update_performance(self, model_key: str, strategy_metrics: Dict, is_successful: bool = False):
        """Update performance metrics for a model"""
        if model_key not in self.model_stats:
            return
        
        stats = self.model_stats[model_key]
        stats["strategies_generated"] += 1
        
        if is_successful:
            stats["successful_strategies"] += 1
            
        # Update metrics if available
        if "Total Return [%]" in strategy_metrics:
            return_val = strategy_metrics["Total Return [%]"]
            stats["total_return"] += return_val
            stats["best_strategy_return"] = max(stats["best_strategy_return"], return_val)
            
        if "Sharpe Ratio" in strategy_metrics:
            sharpe_val = strategy_metrics["Sharpe Ratio"]
            stats["total_sharpe"] += sharpe_val
            
        # Calculate averages
        if stats["strategies_generated"] > 0:
            stats["avg_return"] = stats["total_return"] / stats["strategies_generated"]
            stats["avg_sharpe"] = stats["total_sharpe"] / stats["strategies_generated"]
            stats["success_rate"] = stats["successful_strategies"] / stats["strategies_generated"]
        
        # Log performance update
        logger.model_performance(model_key, stats)
        
        self._save_performance_data()
    
    def _save_performance_data(self):
        """Save performance data to file"""
        try:
            os.makedirs(os.path.dirname(self.performance_file), exist_ok=True)
            with open(self.performance_file, 'w') as f:
                json.dump(self.model_stats, f, indent=2)
        except Exception as e:
            print(f"Error saving performance data: {e}")
    
    def get_best_model_for_task(self, task_type: str = "strategy_generation") -> str:
        """Get the best-performing model for a specific task"""
        if task_type == "strategy_generation":
            # Sort by success rate, then by average return
            sorted_models = sorted(
                self.model_stats.items(),
                key=lambda x: (x[1]["success_rate"], x[1]["avg_return"]),
                reverse=True
            )
            return sorted_models[0][0] if sorted_models else "versatile"
        
        return "analytical"  # Default for other tasks
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get dynamic weights based on performance"""
        base_weight = 0.2
        weights = {}
        
        for model_key, stats in self.model_stats.items():
            # Adjust weight based on success rate
            success_bonus = stats["success_rate"] * 0.3
            performance_bonus = min(stats["avg_return"] / 100, 0.2)  # Cap at 20% bonus
            
            weights[model_key] = base_weight + success_bonus + performance_bonus
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = {k: 0.2 for k in weights.keys()}
        
        return weights
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_rankings": [],
            "overall_stats": {
                "total_strategies": 0,
                "total_successful": 0,
                "overall_success_rate": 0.0
            }
        }
        
        # Sort models by performance
        sorted_models = sorted(
            self.model_stats.items(),
            key=lambda x: (x[1]["success_rate"], x[1]["avg_return"]),
            reverse=True
        )
        
        total_strategies = sum(stats["strategies_generated"] for stats in self.model_stats.values())
        total_successful = sum(stats["successful_strategies"] for stats in self.model_stats.values())
        
        report["overall_stats"]["total_strategies"] = total_strategies
        report["overall_stats"]["total_successful"] = total_successful
        report["overall_stats"]["overall_success_rate"] = (
            total_successful / total_strategies if total_strategies > 0 else 0.0
        )
        
        for model_key, stats in sorted_models:
            report["model_rankings"].append({
                "model": model_key,
                "success_rate": stats["success_rate"],
                "avg_return": stats["avg_return"],
                "avg_sharpe": stats["avg_sharpe"],
                "strategies_generated": stats["strategies_generated"],
                "best_strategy_return": stats["best_strategy_return"]
            })
        
        return report

class MultiLLMManager:
    """Manage multiple LLM models for diverse strategy generation"""
    
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.performance_tracker = ModelPerformanceTracker()
        
        # Simplified system - Groq only
        self.cloudflare_client = None
        self.openrouter_client = None
        logger.info("🚀 Simplified Groq-only system initialized")
        
        self.available_models = self._test_model_availability()
        
        # Initialize Q-learning model selector
        self.q_learning_selector = QLearningModelSelector(self)
        
        # Model definitions - Groq + Cloudflare models
        self.models = {
            # Groq Models
            "versatile": {
                "name": "llama-3.3-70b-versatile",
                "provider": "groq",
                "description": "Balanced & reliable, proven performance",
                "strengths": ["general_purpose", "balanced_creativity", "reliable"]
            },
            "analytical": {
                "name": "deepseek-r1-distill-llama-70b",
                "provider": "groq",
                "description": "Analytical & reasoning focused",
                "strengths": ["mathematical_analysis", "pattern_recognition", "logical_reasoning"]
            },
            "maverick": {
                "name": "meta-llama/llama-4-maverick-17b-128e-instruct",
                "provider": "groq",
                "description": "Creative & unconventional approaches",
                "strengths": ["creative_strategies", "novel_approaches", "fast_inference"]
            },
            "scout": {
                "name": "openai/gpt-oss-20b",
                "provider": "groq",
                "description": "Exploration & discovery focused",
                "strengths": ["pattern_discovery", "exploration", "new_insights"]
            },
            "diverse": {
                "name": "qwen/qwen3-32b",
                "provider": "groq",
                "description": "Different architectural approach",
                "strengths": ["diverse_perspectives", "alternative_thinking", "unique_insights"]
            }
        }
        
        # Task-specific model assignments (Groq-only - 5 models)
        self.task_models = {
            "strategy_generation": [
                "versatile", "analytical", "maverick", "scout", "diverse"
            ],
            "evaluation": [
                "analytical", "versatile", "scout"
            ],
            "learning_analysis": [
                "analytical", "scout", "diverse"
            ],
            "risk_assessment": [
                "analytical", "versatile"
            ],
            "pattern_recognition": [
                "analytical", "scout", "diverse"
            ],
            "code_generation": [
                "analytical", "versatile"
            ],
            "problem_solving": [
                "analytical", "versatile", "scout"
            ],
            "creative_analysis": [
                "maverick", "scout", "diverse"
            ],
            "fast_processing": [
                "scout", "maverick", "diverse"
            ]
        }
        
        # Model parameters (Groq-only)
        self.model_params = {
            "versatile": {"temperature": 0.7, "max_tokens": 2048},
            "analytical": {"temperature": 0.5, "max_tokens": 2048},
            "maverick": {"temperature": 0.8, "max_tokens": 1536},
            "scout": {"temperature": 0.6, "max_tokens": 1536},
            "diverse": {"temperature": 0.7, "max_tokens": 1536}
        }
        
        # Selection mode
        self.selection_mode = os.getenv("GROQ_MODEL_ROTATION", "round_robin")
        self.strategy_counter = 0
        
        # Q-learning integration
        self.use_q_learning = os.getenv("USE_Q_LEARNING", "false").lower() == "true"
        
        logger.info(f"Multi-LLM Manager initialized with Q-learning: {'enabled' if self.use_q_learning else 'disabled'}")
    
    def _test_model_availability(self):
        """Test which models are actually available"""
        available = {}
        
        # Test Groq models
        groq_models = {
            "versatile": "llama-3.3-70b-versatile",
            "analytical": "deepseek-r1-distill-llama-70b",
            "maverick": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "scout": "openai/gpt-oss-20b",
            "diverse": "qwen/qwen3-32b"
        }
        
        for model_key, model_name in groq_models.items():
            try:
                # Test with a minimal request
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hello"}],
                    model=model_name,
                    max_tokens=1,
                    temperature=0.1
                )
                available[model_key] = True
                print(f"✅ Groq model {model_key} ({model_name}) is available")
            except Exception as e:
                available[model_key] = False
                print(f"❌ Groq model {model_key} ({model_name}) is not available: {str(e)[:100]}...")
        
        # Simplified system - Groq only
        print(f"🚀 Simplified system: Using {len(available)} Groq models only")
        
        return available
    
    def select_model(self, task_type: str = "strategy_generation", 
                    strategy_index: Optional[int] = None,
                    market_conditions: Optional[str] = None,
                    market_data: Optional[Dict] = None) -> str:
        """Select the best model for a given task using Q-learning or traditional methods"""
        
        # Filter to only available models
        all_models = self.task_models.get(task_type, ["versatile"])
        available_models = [model for model in all_models if self.available_models.get(model, False)]
        
        # Fallback to versatile if no models available
        if not available_models:
            available_models = ["versatile"] if self.available_models.get("versatile", False) else []
        
        if not available_models:
            raise Exception("No models are available for use")
        
        # Use Q-learning if enabled and market data is available
        if self.use_q_learning and market_data:
            try:
                selected_model = self.q_learning_selector.select_model(market_data, available_models)
                logger.info(f"Q-learning selected model: {selected_model}")
                return selected_model
            except Exception as e:
                logger.error(f"Q-learning model selection failed: {e}")
                # Fall back to traditional selection
        
        if self.selection_mode == "round_robin":
            index = (strategy_index or self.strategy_counter) % len(available_models)
            return available_models[index]
            
        elif self.selection_mode == "performance_based":
            return self.performance_tracker.get_best_model_for_task(task_type)
            
        elif self.selection_mode == "weighted_random":
            weights = self.performance_tracker.get_model_weights()
            available_weights = {k: weights[k] for k in available_models if k in weights}
            
            if available_weights:
                models = list(available_weights.keys())
                weights_list = list(available_weights.values())
                
                # Normalize weights to ensure they sum to 1
                total_weight = sum(weights_list)
                if total_weight > 0:
                    weights_list = [w / total_weight for w in weights_list]
                    return np.random.choice(models, p=weights_list)
                else:
                    return np.random.choice(models)
            
        elif self.selection_mode == "market_adaptive":
            # Adjust selection based on market conditions
            if market_conditions == "high_volatility":
                return "analytical"
            elif market_conditions == "trending":
                return "maverick"
            elif market_conditions == "sideways":
                return "scout"
            elif market_conditions == "uncertain":
                return "diverse"
        
        # Default fallback
        return available_models[0]
    
    def generate_with_model(self, prompt: str, model_key: str, **kwargs) -> Tuple[str, Dict]:
        """Generate content using specified model with robust error recovery"""
        
        # Check if model is available
        if not self.available_models.get(model_key, False):
            logger.warning(f"⚠️  Model {model_key} is not available, finding fallback...")
            model_key = self._find_fallback_model(model_key)
        
        if model_key not in self.models:
            model_key = self._find_fallback_model("versatile")
        
        model_info = self.models[model_key]
        model_name = model_info["name"]
        provider = model_info["provider"]
        params = self.model_params[model_key].copy()
        params.update(kwargs)
        
        # Simplified routing - Groq only
        if provider == "groq":
            return self._generate_with_groq(prompt, model_key, model_name, params)
        else:
            raise Exception(f"Unsupported provider: {provider}. Only Groq models are supported in simplified mode.")
    
    def _generate_with_groq(self, prompt: str, model_key: str, model_name: str, params: Dict) -> Tuple[str, Dict]:
        """Generate content using Groq API"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,
                    **params
                )
                
                content = response.choices[0].message.content
                metadata = {
                    "model_key": model_key,
                    "model_name": model_name,
                    "provider": "groq",
                    "timestamp": datetime.now().isoformat(),
                    "token_usage": getattr(response, 'usage', None),
                    "attempts": attempt + 1
                }
                
                return content, metadata
                
            except Exception as e:
                logger.error(f"Error with Groq model {model_key} (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    import time
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise e
    
    # Removed Cloudflare and OpenRouter methods - Groq only system
    
    def _find_fallback_model(self, preferred_model: str = None) -> str:
        """Find the best available fallback model (Groq only)"""
        # Priority order for fallbacks (Groq models only)
        fallback_priority = [
            "versatile", "analytical", "maverick", "scout", "diverse"
        ]
        
        # If preferred model is specified, try it first
        if preferred_model and self.available_models.get(preferred_model, False):
            return preferred_model
        
        # Find first available model in priority order
        for model_key in fallback_priority:
            if self.available_models.get(model_key, False):
                return model_key
        
        # If no models available, raise exception
        raise Exception("No fallback models available")
    
    def generate_ensemble_strategies(self, prompt: str, num_models: int = 3) -> List[Dict]:
        """Generate multiple strategies using different models"""
        
        available_models = self.task_models["strategy_generation"]
        selected_models = random.sample(
            available_models,
            min(num_models, len(available_models))
        )
        
        strategies = []
        for model_key in selected_models:
            try:
                content, metadata = self.generate_with_model(prompt, model_key)
                strategies.append({
                    "model": model_key,
                    "content": content,
                    "metadata": metadata
                })
            except Exception as e:
                print(f"Failed to generate with model {model_key}: {e}")
                continue
        
        return strategies
    
    def generate_ensemble_consensus(self, prompt: str, num_models: int = 5, task_type: str = "strategy_generation") -> Dict:
        """Generate collaborative consensus from multiple models working together"""
        
        # Select diverse Groq models for ensemble
        available_models = self.task_models.get(task_type, self.task_models["strategy_generation"])
        
        # All models are Groq in simplified system
        groq_models = [m for m in available_models if self.models.get(m, {}).get("provider") == "groq"]
        
        # Select up to num_models from available Groq models
        selected_models = random.sample(groq_models, min(num_models, len(groq_models)))
        
        logger.info(f"🤝 Ensemble consensus with {len(selected_models)} models: {', '.join(selected_models)}")
        
        # Generate strategies from all models
        ensemble_results = []
        for model_key in selected_models:
            try:
                if not self.available_models.get(model_key, False):
                    continue
                    
                content, metadata = self.generate_with_model(prompt, model_key)
                provider = self.models[model_key]["provider"]
                
                ensemble_results.append({
                    "model": model_key,
                    "provider": provider,
                    "content": content,
                    "metadata": metadata,
                    "strengths": self.models[model_key]["strengths"]
                })
                
                logger.info(f"✅ {model_key} ({provider}): Generated {len(content)} chars")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate with model {model_key}: {e}")
                continue
        
        # Create consensus analysis
        consensus = {
            "ensemble_size": len(ensemble_results),
            "providers_used": list(set([r["provider"] for r in ensemble_results])),
            "models_used": [r["model"] for r in ensemble_results],
            "individual_results": ensemble_results,
            "consensus_metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "task_type": task_type,
                "prompt_length": len(prompt),
                "total_content_generated": sum(len(r["content"]) for r in ensemble_results)
            }
        }
        
        logger.info(f"🎯 Ensemble consensus complete: {len(ensemble_results)} models, {len(consensus['providers_used'])} providers")
        
        return consensus
    
    def update_model_performance(self, model_key: str, strategy_metrics: Dict, is_successful: bool = False, next_market_data: Dict = None):
        """Update performance tracking for a model and Q-learning agent"""
        self.performance_tracker.update_performance(model_key, strategy_metrics, is_successful)
        
        # Update Q-learning agent if enabled
        if self.use_q_learning:
            try:
                self.q_learning_selector.update_performance(strategy_metrics, next_market_data)
                logger.info(f"Updated Q-learning performance for model: {model_key}")
            except Exception as e:
                logger.error(f"Failed to update Q-learning performance: {e}")
    
    def get_model_info(self, model_key: str) -> Dict:
        """Get information about a specific model"""
        if model_key in self.models:
            return self.models[model_key]
        return {}
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        return self.performance_tracker.get_performance_report()
    
    def switch_selection_mode(self, mode: str):
        """Switch model selection mode"""
        valid_modes = ["round_robin", "performance_based", "weighted_random", "market_adaptive"]
        if mode in valid_modes:
            self.selection_mode = mode
            print(f"✅ Model selection mode switched to: {mode}")
        else:
            print(f"❌ Invalid mode. Valid modes: {valid_modes}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model keys"""
        return list(self.models.keys())
    
    def increment_strategy_counter(self):
        """Increment strategy counter for round-robin selection"""
        self.strategy_counter += 1
    
    def enable_q_learning(self, enable: bool = True):
        """Enable or disable Q-learning model selection"""
        self.use_q_learning = enable
        logger.info(f"Q-learning {'enabled' if enable else 'disabled'}")
    
    def end_q_learning_cycle(self):
        """End current Q-learning cycle"""
        if self.use_q_learning:
            try:
                self.q_learning_selector.end_learning_cycle()
                logger.info("Q-learning cycle ended")
            except Exception as e:
                logger.error(f"Failed to end Q-learning cycle: {e}")
    
    def get_q_learning_stats(self) -> Dict:
        """Get Q-learning statistics"""
        if self.use_q_learning:
            try:
                return self.q_learning_selector.get_learning_stats()
            except Exception as e:
                logger.error(f"Failed to get Q-learning stats: {e}")
                return {}
        return {"q_learning": "disabled"}

if __name__ == "__main__":
    # Test the multi-LLM manager
    manager = MultiLLMManager()
    
    # Test model selection
    print("🧠 Testing Multi-LLM Manager")
    print(f"Available models: {manager.get_available_models()}")
    
    # Test different selection modes
    for mode in ["round_robin", "performance_based", "weighted_random"]:
        manager.switch_selection_mode(mode)
        selected = manager.select_model("strategy_generation", 0)
        print(f"Mode: {mode}, Selected: {selected}")
    
    # Test generation
    test_prompt = "Generate a simple momentum trading strategy for BTCUSDT"
    try:
        content, metadata = manager.generate_with_model(test_prompt, "versatile")
        print(f"✅ Test generation successful with model: {metadata['model_key']}")
    except Exception as e:
        print(f"❌ Test generation failed: {e}")
    
    # Test ensemble
    try:
        ensemble = manager.generate_ensemble_strategies(test_prompt, 3)
        print(f"✅ Ensemble generation successful with {len(ensemble)} strategies")
    except Exception as e:
        print(f"❌ Ensemble generation failed: {e}")
    
    print("🎯 Multi-LLM Manager ready for integration!")