import getpass
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

load_dotenv()

GROQ_API_KEY = "GROQ_API_KEY"


def ensure_groq_api_key():
    """Ensures Groq API key is available in environment."""
    if not os.environ.get(GROQ_API_KEY):
        os.environ[GROQ_API_KEY] = getpass.getpass("Enter API key for Groq: ")
    return os.environ[GROQ_API_KEY]

KEY_OPENAI_API_KEY = "OPENAI_API_KEY"


def ensure_openai_api_key():
    if not os.environ.get(KEY_OPENAI_API_KEY):
        os.environ[KEY_OPENAI_API_KEY] = getpass.getpass("Enter API key for OpenAI: ")
    return os.environ[KEY_OPENAI_API_KEY]

def llm_client(model_name: str = None, model_provider: str = "groq", structured_output: type = None):
    """
    Creates an LLM client configured for Groq.
    
    Args:
        model_name: gpt-4.1-mini
        model_provider: OpenAI
    
    Returns:
        Tuple of (llm_instance, model_name, model_provider)
    """




    model_configs = {
        "chat" : "gpt-4.1",
        "gemini" : "google_genai:gemini-2.5-flash-lite",
        "llama-3.3": "groq/llama-3.3-70b-versatile",
        "llama-8b": "groq/llama-3.1-8b-instant",
        "llama-4":"meta-llama/llama-4-maverick-17b-128e-instruct",
        "openAI":"openai/gpt-oss-120b",
        "gwen":"qwen/qwen3-32b"
    }

    # Use model_name directly if it's a full Groq model name
    if model_name and model_name in model_configs:
        model_name = model_configs[model_name]
    elif not model_name:
        #model_name = "gpt-4o"
        model_name ="gpt-4.1-mini"

    # For compatibility
    if not model_provider:
        model_provider = "groq"

    ensure_groq_api_key()

    # rate_limiter = InMemoryRateLimiter(
    #     requests_per_second=1,
    #     check_every_n_seconds=0.1,
    #     max_bucket_size=10,
    # )

    # Create ChatGroq instance
    model = ChatOpenAI(
        model = model_name,
        temperature=0.1,
        #rate_limiter = rate_limiter
    )

    if structured_output is not None:
        model = model.with_structured_output(structured_output)
    
    return model, model_name, model_provider


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