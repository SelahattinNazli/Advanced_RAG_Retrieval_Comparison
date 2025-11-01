"""
Ollama LLM integration for Advanced RAG Comparison
"""
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from typing import List, Optional
from src.config import Config


class OllamaLLM:
    """Wrapper for Ollama LLM with configuration"""
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        """
        Initialize Ollama LLM
        
        Args:
            model: Model name (default from config)
            base_url: Ollama base URL (default from config)
            temperature: Sampling temperature (default from config)
            request_timeout: Request timeout in seconds (default from config)
        """
        self.model = model or Config.OLLAMA_MODEL
        self.base_url = base_url or Config.OLLAMA_BASE_URL
        self.temperature = temperature if temperature is not None else Config.OLLAMA_TEMPERATURE
        self.request_timeout = request_timeout or Config.OLLAMA_REQUEST_TIMEOUT
        
        # Initialize Ollama
        self.llm = Ollama(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            request_timeout=self.request_timeout,
        )
        
        print(f"✅ Ollama LLM initialized: {self.model}")
    
    def complete(self, prompt: str) -> str:
        """
        Generate completion for a prompt
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        response = self.llm.complete(prompt)
        return response.text
    
    def chat(self, messages: List[ChatMessage]) -> str:
        """
        Chat completion with message history
        
        Args:
            messages: List of chat messages
            
        Returns:
            Generated response
        """
        response = self.llm.chat(messages)
        return response.message.content
    
    def stream_complete(self, prompt: str):
        """
        Stream completion for a prompt
        
        Args:
            prompt: Input prompt
            
        Yields:
            Text chunks
        """
        response = self.llm.stream_complete(prompt)
        for chunk in response:
            yield chunk.delta
    
    @classmethod
    def test_connection(cls) -> bool:
        """
        Test if Ollama is accessible and model is available
        
        Returns:
            True if connection successful
        """
        try:
            import requests
            
            # Check if Ollama is running
            response = requests.get(
                f"{Config.OLLAMA_BASE_URL}/api/tags",
                timeout=5
            )
            
            if response.status_code != 200:
                print(f"❌ Ollama not accessible at {Config.OLLAMA_BASE_URL}")
                return False
            
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            if Config.OLLAMA_MODEL not in model_names:
                print(f"❌ Model {Config.OLLAMA_MODEL} not found")
                print(f"Available models: {model_names}")
                print(f"\nTo pull the model, run:")
                print(f"  ollama pull {Config.OLLAMA_MODEL}")
                return False
            
            print(f"✅ Ollama connection successful")
            print(f"✅ Model {Config.OLLAMA_MODEL} is available")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Ollama: {e}")
            print(f"\nMake sure Ollama is running:")
            print(f"  ollama serve")
            return False
    
    def get_llm(self):
        """Get the underlying LlamaIndex LLM object"""
        return self.llm


def get_llm() -> Ollama:
    """
    Convenience function to get configured Ollama LLM
    
    Returns:
        Configured Ollama LLM instance
    """
    llm_wrapper = OllamaLLM()
    return llm_wrapper.get_llm()


# Test on import (optional - comment out if not needed)
if __name__ == "__main__":
    print("\n🧪 Testing Ollama connection...")
    OllamaLLM.test_connection()
    
    # Test completion
    print("\n🧪 Testing completion...")
    llm = OllamaLLM()
    response = llm.complete("What is RAG in 10 words?")
    print(f"Response: {response}")