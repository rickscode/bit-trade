#!/usr/bin/env python3
"""
Cloudflare Workers AI Client for Multi-LLM Trading System
"""

import os
import requests
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

try:
    from .enhanced_logger import logger
except ImportError:
    from enhanced_logger import logger

load_dotenv()

class CloudflareAIClient:
    """Client for Cloudflare Workers AI API"""
    
    def __init__(self):
        self.account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.auth_token = os.getenv("CLOUDFLARE_AUTH_TOKEN")
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run"
        
        if not self.account_id or not self.auth_token:
            raise ValueError("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN must be set in environment")
        
        # Cloudflare Workers AI models for trading strategies
        self.models = {
            "reasoning": {
                "name": "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
                "description": "Advanced reasoning and analysis model",
                "strengths": ["logical_reasoning", "complex_analysis", "pattern_recognition"],
                "params": {"temperature": 0.5, "max_tokens": 2048}
            },
            "coder": {
                "name": "@cf/qwen/qwen2.5-coder-32b-instruct", 
                "description": "Code-specialized model for strategy implementation",
                "strengths": ["code_generation", "algorithm_design", "technical_precision"],
                "params": {"temperature": 0.3, "max_tokens": 2048}
            },
            "questioner": {
                "name": "@cf/qwen/qwq-32b",
                "description": "Question-answering and problem-solving model", 
                "strengths": ["problem_solving", "market_analysis", "strategic_thinking"],
                "params": {"temperature": 0.6, "max_tokens": 1536}
            }
        }
        
        # Test availability on initialization
        self.available_models = self._test_model_availability()
        
    def _test_model_availability(self) -> Dict[str, bool]:
        """Test which Cloudflare AI models are available"""
        available = {}
        
        for model_key, model_info in self.models.items():
            try:
                # Test with a simple request
                response = self._make_request(
                    model_info["name"],
                    [{"role": "user", "content": "Hello"}],
                    max_tokens=1,
                    temperature=0.1
                )
                
                if response and 'result' in response:
                    available[model_key] = True
                    logger.info(f"✅ Cloudflare model {model_key} ({model_info['name']}) is available")
                else:
                    available[model_key] = False
                    logger.error(f"❌ Cloudflare model {model_key} test failed: {response}")
                    
            except Exception as e:
                available[model_key] = False
                logger.error(f"❌ Cloudflare model {model_key} ({model_info['name']}) is not available: {str(e)[:100]}...")
        
        return available
    
    def _make_request(self, model_name: str, messages: List[Dict], **kwargs) -> Dict:
        """Make a request to Cloudflare Workers AI API"""
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": messages,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/{model_name}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            raise Exception("Request timed out")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def generate_with_model(self, prompt: str, model_key: str, **kwargs) -> Tuple[str, Dict]:
        """Generate content using specified Cloudflare AI model with error recovery"""
        
        # Check if model is available
        if not self.available_models.get(model_key, False):
            logger.warning(f"⚠️  Cloudflare model {model_key} is not available, finding fallback...")
            model_key = self._find_fallback_model(model_key)
        
        if model_key not in self.models:
            model_key = self._find_fallback_model("reasoning")
        
        model_info = self.models[model_key]
        model_name = model_info["name"]
        params = model_info["params"].copy()
        params.update(kwargs)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": "You are an expert cryptocurrency trading strategy developer. Generate precise, actionable trading strategies with clear entry/exit rules."},
            {"role": "user", "content": prompt}
        ]
        
        # Multiple retry attempts with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"🔮 Calling Cloudflare AI model: {model_key} (attempt {attempt + 1})")
                
                response = self._make_request(model_name, messages, **params)
                
                if response and 'result' in response:
                    # Extract content from Cloudflare response format
                    result = response['result']
                    if 'response' in result:
                        content = result['response']
                    elif 'text' in result:
                        content = result['text']  
                    elif isinstance(result, str):
                        content = result
                    else:
                        content = str(result)
                    
                    metadata = {
                        "model_key": model_key,
                        "model_name": model_name,
                        "provider": "cloudflare",
                        "timestamp": datetime.now().isoformat(),
                        "attempts": attempt + 1,
                        "response_metadata": response.get('messages', [])
                    }
                    
                    logger.info(f"✅ Cloudflare AI generation successful with {model_key}")
                    return content, metadata
                else:
                    raise Exception(f"Invalid response format: {response}")
                
            except Exception as e:
                logger.error(f"Error with Cloudflare model {model_key} (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Final fallback to any available model
                    if model_key != "reasoning" and self.available_models.get("reasoning", False):
                        logger.info(f"🔄 Final fallback to reasoning model")
                        return self.generate_with_model(prompt, "reasoning", **kwargs)
                    else:
                        # Find any available model
                        fallback = self._find_fallback_model()
                        if fallback and fallback != model_key:
                            logger.info(f"🔄 Emergency fallback to {fallback}")
                            return self.generate_with_model(prompt, fallback, **kwargs)
                        else:
                            raise Exception(f"All Cloudflare models failed after {max_retries} attempts: {e}")
    
    def _find_fallback_model(self, preferred_model: str = None) -> str:
        """Find the best available fallback model"""
        # Priority order for fallbacks
        fallback_priority = ["reasoning", "coder", "questioner"]
        
        # If preferred model is specified, try it first
        if preferred_model and self.available_models.get(preferred_model, False):
            return preferred_model
        
        # Find first available model in priority order
        for model_key in fallback_priority:
            if self.available_models.get(model_key, False):
                return model_key
        
        # If no models available, raise exception
        raise Exception("No Cloudflare fallback models available")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model keys"""
        return [key for key, available in self.available_models.items() if available]
    
    def get_model_info(self, model_key: str) -> Dict:
        """Get information about a specific model"""
        return self.models.get(model_key, {})
    
    def test_connection(self) -> bool:
        """Test connection to Cloudflare Workers AI"""
        try:
            available_models = self.get_available_models()
            if available_models:
                test_model = available_models[0]
                content, metadata = self.generate_with_model("Test", test_model, max_tokens=5)
                logger.info(f"✅ Cloudflare AI connection test successful")
                return True
            else:
                logger.error("❌ No available Cloudflare models for connection test")
                return False
        except Exception as e:
            logger.error(f"❌ Cloudflare AI connection test failed: {e}")
            return False

if __name__ == "__main__":
    # Test Cloudflare AI client
    try:
        client = CloudflareAIClient()
        
        print("🔮 Testing Cloudflare Workers AI Client")
        print(f"Available models: {client.get_available_models()}")
        
        # Test each available model
        for model_key in client.get_available_models():
            print(f"\n🧪 Testing model: {model_key}")
            try:
                content, metadata = client.generate_with_model(
                    "Generate a simple momentum trading strategy for BTCUSDT", 
                    model_key,
                    max_tokens=100
                )
                print(f"✅ {model_key}: Generated {len(content)} characters")
                print(f"   Provider: {metadata['provider']}")
                print(f"   Model: {metadata['model_name']}")
            except Exception as e:
                print(f"❌ {model_key}: {e}")
        
        # Test connection
        connection_ok = client.test_connection()
        print(f"\n🌐 Connection test: {'✅ PASSED' if connection_ok else '❌ FAILED'}")
        
    except Exception as e:
        print(f"❌ Failed to initialize Cloudflare AI client: {e}")