#!/usr/bin/env python3
"""
OpenRouter AI Client for Bit-Trade Multi-LLM System
Provides access to free models on OpenRouter platform
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

try:
    from .enhanced_logger import logger
except ImportError:
    from enhanced_logger import logger

load_dotenv()

class OpenRouterAIClient:
    """Client for interacting with OpenRouter AI models"""
    
    def __init__(self):
        """Initialize OpenRouter client"""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://bit-trade.local",  # Optional for tracking
            "X-Title": "Bit-Trade Autonomous Trading System"  # Optional for tracking
        }
        
        # Rate limiting (free tier: 50-1000 requests/day)
        self.rate_limit_delay = 1.0  # Minimum delay between requests
        self.last_request_time = 0
        
        # Model configurations for free models
        self.model_configs = {
            # Tier 1: High-performance models
            "horizon": {
                "name": "openrouter/horizon-beta",
                "description": "Improved version of Horizon Alpha, general purpose",
                "context_length": 256000,
                "strengths": ["general_purpose", "large_context", "versatile"],
                "specialties": ["strategy_generation", "analysis"]
            },
            "glm_air": {
                "name": "z-ai/glm-4.5-air",
                "description": "Lightweight variant optimized for agent-centric applications",
                "context_length": 131072,
                "strengths": ["agent_tasks", "efficiency", "reasoning"],
                "specialties": ["strategy_optimization", "decision_making"]
            },
            "kimi_k2": {
                "name": "moonshot/kimi-k2",
                "description": "Large-scale MoE model with 1T total params, 32B active",
                "context_length": 33000,
                "strengths": ["complex_reasoning", "performance", "efficiency"],
                "specialties": ["market_analysis", "pattern_recognition"]
            },
            "kimi_dev": {
                "name": "moonshot/kimi-dev-72b:free",
                "description": "Open-source large language model fine-tuned for software engineering",
                "context_length": 131072,
                "strengths": ["software_engineering", "problem_solving", "code_optimization"],
                "specialties": ["algorithm_design", "strategy_implementation"]
            },
            # Tier 2: Specialized models
            "qwen_coder": {
                "name": "qwen/qwen3-coder",
                "description": "Code generation specialist optimized for agentic coding tasks",
                "context_length": 262144,
                "strengths": ["code_generation", "algorithm_design", "technical_implementation"],
                "specialties": ["strategy_implementation", "backtesting_code"]
            },
            "deepseek_r1": {
                "name": "deepseek/deepseek-r1:free",
                "description": "Advanced reasoning model from DeepSeek",
                "context_length": 65536,
                "strengths": ["reasoning", "analysis", "problem_solving"],
                "specialties": ["risk_analysis", "strategy_evaluation"]
            },
            "deepseek_0528": {
                "name": "deepseek/deepseek-r1-0528:free",
                "description": "Updated DeepSeek R1 with improved performance",
                "context_length": 164000,
                "strengths": ["advanced_reasoning", "mathematical_analysis", "pattern_recognition"],
                "specialties": ["quantitative_analysis", "market_modeling"]
            },
            "deepseek_qwen": {
                "name": "deepseek/deepseek-r1-0528-qwen3-8b:free",
                "description": "DeepSeek R1 optimized with Qwen3 architecture",
                "context_length": 131072,
                "strengths": ["hybrid_reasoning", "efficiency", "multilingual"],
                "specialties": ["cross_market_analysis", "global_strategies"]
            },
            "chimera": {
                "name": "tng/deepseek-r1t2-chimera:free",
                "description": "Second-generation Chimera model from TNG Tech",
                "context_length": 164000,
                "strengths": ["text_generation", "creative_strategies", "hybrid_approaches"],
                "specialties": ["novel_strategy_generation", "creative_analysis"]
            },
            # Tier 3: Efficient models
            "qwen_qwq": {
                "name": "qwen/qwq-32b:free",
                "description": "Question-answering specialist from Qwen family",
                "context_length": 32768,
                "strengths": ["question_answering", "problem_solving", "analysis"],
                "specialties": ["market_research", "strategy_validation"]
            },
            "gemma3_4b": {
                "name": "google/gemma-3n-4b:free",
                "description": "Google's efficient Gemma 3 model optimized for mobile devices",
                "context_length": 8192,
                "strengths": ["efficiency", "multimodal", "mobile_optimization"],
                "specialties": ["quick_analysis", "real_time_processing"]
            },
            "gemma3_2b": {
                "name": "google/gemma-3n-2b:free",
                "description": "Ultra-efficient Gemma 3 model for low-resource devices",
                "context_length": 8192,
                "strengths": ["ultra_efficiency", "speed", "low_resource"],
                "specialties": ["fast_decisions", "lightweight_analysis"]
            },
            "mistral_small": {
                "name": "mistral/mistral-small-3.2-24b:free",
                "description": "Updated 24B parameter model optimized for instruction following",
                "context_length": 96000,
                "strengths": ["instruction_following", "function_calling", "optimization"],
                "specialties": ["strategy_execution", "parameter_optimization"]
            },
            "mistral_devstral": {
                "name": "mistral/devstral-small-2505:free",
                "description": "Agentic LLM fine-tuned for advanced software engineering tasks",
                "context_length": 33000,
                "strengths": ["software_engineering", "agent_tasks", "tool_usage"],
                "specialties": ["system_architecture", "automated_trading"]
            },
            "sarvam_m": {
                "name": "sarvam/sarvam-m:free",
                "description": "24B-parameter instruction-tuned model for multiple languages",
                "context_length": 33000,
                "strengths": ["multilingual", "instruction_tuned", "diverse_perspectives"],
                "specialties": ["global_market_analysis", "cross_cultural_strategies"]
            },
            "venice_uncensored": {
                "name": "venice/uncensored:free",
                "description": "Fine-tuned variant of Mistral-Small designed for unrestricted analysis",
                "context_length": 93000,
                "strengths": ["uncensored_analysis", "creative_freedom", "diverse_thinking"],
                "specialties": ["unconventional_strategies", "creative_analysis"]
            },
            "hunyuan": {
                "name": "tencent/hunyuan-a13b-instruct:free",
                "description": "13B active parameter MoE model with Chain-of-Thought reasoning",
                "context_length": 33000,
                "strengths": ["chain_of_thought", "reasoning", "efficiency"],
                "specialties": ["step_by_step_analysis", "logical_reasoning"]
            }
        }
        
        # Test model availability on initialization
        self.available_models = self._test_model_availability()
        
        logger.info(f"OpenRouter client initialized with {len(self.available_models)} available models")
    
    def _test_model_availability(self) -> List[str]:
        """Test which models are actually available"""
        available = []
        
        for model_key, config in self.model_configs.items():
            try:
                # Test with a minimal request
                test_response = self._make_request(
                    config["name"],
                    "Test",
                    max_tokens=10,
                    temperature=0.1
                )
                
                if test_response and "error" not in test_response:
                    available.append(model_key)
                    logger.info(f"✅ {model_key} ({config['name']}) - Available")
                else:
                    logger.warning(f"❌ {model_key} ({config['name']}) - Unavailable: {test_response}")
                    
            except Exception as e:
                logger.warning(f"❌ {model_key} ({config['name']}) - Error: {str(e)}")
                
            # Rate limiting delay
            time.sleep(self.rate_limit_delay)
        
        return available
    
    def _make_request(self, model_name: str, prompt: str, 
                     max_tokens: int = 2048, temperature: float = 0.7) -> Optional[Dict]:
        """Make a request to OpenRouter API"""
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
        
        # Prepare request payload (OpenAI-compatible format)
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"OpenRouter API error {response.status_code}: {response.text}")
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error(f"OpenRouter request failed: {str(e)}")
            return {"error": str(e)}
    
    def generate_with_model(self, model_key: str, prompt: str, 
                           max_tokens: int = 2048, temperature: float = 0.7) -> Tuple[str, Dict]:
        """Generate content using a specific OpenRouter model"""
        
        if model_key not in self.available_models:
            raise ValueError(f"Model {model_key} not available. Available: {self.available_models}")
        
        config = self.model_configs[model_key]
        model_name = config["name"]
        
        logger.info(f"🔮 Generating with OpenRouter model: {model_key} ({model_name})")
        
        # Make the request
        response = self._make_request(model_name, prompt, max_tokens, temperature)
        
        if response and "error" not in response:
            # Extract content from OpenAI-compatible response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Prepare metadata
            metadata = {
                "model_key": model_key,
                "model_name": model_name,
                "provider": "openrouter",
                "timestamp": datetime.now().isoformat(),
                "tokens_used": response.get("usage", {}).get("total_tokens", 0),
                "context_length": config["context_length"],
                "strengths": config["strengths"],
                "specialties": config["specialties"]
            }
            
            logger.info(f"✅ Generated {len(content)} characters using {model_key}")
            return content, metadata
            
        else:
            error_msg = response.get("error", "Unknown error") if response else "No response"
            logger.error(f"❌ Failed to generate with {model_key}: {error_msg}")
            raise Exception(f"OpenRouter generation failed: {error_msg}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model keys"""
        return self.available_models.copy()
    
    def get_model_info(self, model_key: str) -> Dict:
        """Get detailed information about a specific model"""
        if model_key in self.model_configs:
            return self.model_configs[model_key].copy()
        else:
            raise ValueError(f"Unknown model key: {model_key}")
    
    def get_best_model_for_task(self, task_type: str) -> Optional[str]:
        """Get the best available model for a specific task"""
        
        task_preferences = {
            "strategy_generation": ["horizon", "glm_air", "gemma3"],
            "code_generation": ["qwen_coder", "glm_air", "horizon"],
            "analysis": ["deepseek_r1", "kimi_k2", "qwen_qwq"],
            "reasoning": ["deepseek_r1", "glm_air", "kimi_k2"],
            "evaluation": ["qwen_qwq", "deepseek_r1", "gemma3"],
            "market_research": ["kimi_k2", "qwen_qwq", "horizon"]
        }
        
        preferred_models = task_preferences.get(task_type, list(self.available_models))
        
        # Return first available preferred model
        for model in preferred_models:
            if model in self.available_models:
                return model
        
        # Fallback to any available model
        return self.available_models[0] if self.available_models else None
    
    def test_all_models(self) -> Dict[str, bool]:
        """Test all configured models and return availability status"""
        results = {}
        
        for model_key in self.model_configs.keys():
            try:
                content, metadata = self.generate_with_model(
                    model_key, 
                    "Generate a brief trading strategy idea", 
                    max_tokens=100
                )
                results[model_key] = True
                print(f"✅ {model_key}: {content[:100]}...")
                
            except Exception as e:
                results[model_key] = False
                print(f"❌ {model_key}: {str(e)}")
                
            # Rate limiting delay
            time.sleep(self.rate_limit_delay)
        
        return results

def main():
    """Test OpenRouter client functionality"""
    print("🔮 Testing OpenRouter AI Client")
    print("=" * 50)
    
    try:
        client = OpenRouterAIClient()
        print(f"Available models: {client.get_available_models()}")
        print()
        
        # Test each available model
        results = client.test_all_models()
        
        print(f"\n📊 Summary:")
        print(f"Total models configured: {len(client.model_configs)}")
        print(f"Available models: {len(client.available_models)}")
        print(f"Success rate: {sum(results.values())}/{len(results)} models working")
        
    except Exception as e:
        print(f"❌ Error testing OpenRouter client: {e}")

if __name__ == "__main__":
    main()