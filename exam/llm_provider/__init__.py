import getpass
import os
from langchain_groq import ChatGroq


KEY_GROQ_API_KEY = "GROQ_API_KEY"


def ensure_groq_api_key():
    """Ensures Groq API key is available in environment."""
    if not os.environ.get(KEY_GROQ_API_KEY):
        os.environ[KEY_GROQ_API_KEY] = getpass.getpass("Enter API key for Groq: ")
    return os.environ[KEY_GROQ_API_KEY]


def llm_client(model_name: str = "llama-3.3-70b-versatile", model_provider: str = "groq", structured_output: type = None):
    """
    Creates an LLM client configured for Groq.
    
    Args:
        model_name: llama-3.3-70b
        model_provider: Groq
    
    Returns:
        Tuple of (llm_instance, model_name, model_provider)
    """
    # Model configurations
    model_configs = {
        "llama-3.3": "llama-3.3-70b-versatile",
        "llama-8b": "llama-3.1-8b-instant",
        "llama-4":"meta-llama/llama-4-maverick-17b-128e-instruct",
        "openAI":"openai/gpt-oss-120b",
        "gwen":"qwen/qwen3-32b"
    }
    
    # Use model_name directly if it's a full Groq model name
    if model_name and model_name in model_configs:
        model_name = model_configs[model_name]
    elif not model_name:
        model_name = "llama-3.3-70b-versatile"  # Default
    
    # For compatibility
    if not model_provider:
        model_provider = "groq"
    
    ensure_groq_api_key()
    
    # Create ChatGroq instance
    llm = ChatGroq(
        model=model_name,
        groq_api_key=os.environ[KEY_GROQ_API_KEY],
        temperature=0.1,  # Lower temperature for more consistent grading
        max_tokens=8000,
    )
    
    if structured_output is not None:
        llm = llm.with_structured_output(structured_output)
    
    return llm, model_name, model_provider


class AIOracle:
    """Base class for AI-powered operations using Groq."""
    
    def __init__(self, model_name: str = None, model_provider: str = None, structured_output: type = None):
        self.__llm, self.__model_name, self.__model_provider = llm_client(
            model_name, model_provider, structured_output
        )

    @property
    def llm(self):
        return self.__llm

    @property
    def model_name(self):
        return self.__model_name

    @property
    def model_provider(self):
        return self.__model_provider