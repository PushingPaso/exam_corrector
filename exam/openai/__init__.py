import getpass
import os
from langchain_openai import ChatOpenAI


KEY_OPENROUTER_API_KEY = "OPENROUTER_API_KEY"


def ensure_openrouter_api_key():
    """Ensures OpenRouter API key is available in environment."""
    if not os.environ.get(KEY_OPENROUTER_API_KEY):
        os.environ[KEY_OPENROUTER_API_KEY] = getpass.getpass("Enter API key for OpenRouter: ")
    return os.environ[KEY_OPENROUTER_API_KEY]


def llm_client(model_name: str = None, model_provider: str = None, structured_output: type = None):
    """
    Creates an LLM client configured for OpenRouter.
    
    Args:
        model_name: Model identifier (default: mistralai/mistral-7b-instruct)
        model_provider: Provider name (kept for compatibility, but uses OpenRouter)
        structured_output: Optional Pydantic model for structured output
    
    Returns:
        Tuple of (llm_instance, model_name, model_provider)
    """
    if not model_name:
        model_name = "qwen/qwen3-coder:free"
    
    # For compatibility, default provider to "openrouter"
    if not model_provider:
        model_provider = "openrouter"
    
    ensure_openrouter_api_key()
    
    # Create ChatOpenAI instance configured for OpenRouter
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=os.environ[KEY_OPENROUTER_API_KEY],
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.1,  # Lower temperature for more consistent grading
    )
    
    if structured_output is not None:
        llm = llm.with_structured_output(structured_output)
    
    return llm, model_name, model_provider


class AIOracle:
    """Base class for AI-powered operations using OpenRouter/Mistral."""
    
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