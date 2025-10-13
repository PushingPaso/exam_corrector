from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from exam import DIR_ROOT, Question
from exam.llm_provider import AIOracle
from exam.rag import sqlite_vector_store
from yaml import safe_dump, safe_load


FILE_TEMPLATE = DIR_ROOT / "exam" / "solution" / "prompt-template.txt"
DIR_SOLUTIONS = DIR_ROOT / "solutions"
DIR_SOLUTIONS.mkdir(exist_ok=True)


class Answer(BaseModel):
    core: list[str] = Field(
        description="Elementi essenziali che devono essere presenti nella risposta perfetta per rispondere alla parte più importante della domanda. Ogni item è una stringa Markdown.",
    )
    details_important: list[str] = Field(
        description="Dettagli importanti che dovrebbero essere menzionati per arricchire la risposta. Ogni item è una stringa Markdown.",
    )
    details_additional: list[str] = Field(
        description="Dettagli aggiuntivi opzionali che possono ulteriormente migliorare la risposta. Ogni item è una stringa Markdown.",
    )

    def pretty(self, indent=0, prefix="\t") -> str:
        result = "Core (elementi essenziali):\n"
        if self.core:
            result += "\n".join(f"- {item}" for item in self.core) + "\n"
        else:
            result += "- <none>\n"
        
        result += "Details - Importanti:\n"
        if self.details_important:
            result += "\n".join(f"- {item}" for item in self.details_important) + "\n"
        else:
            result += "- <none>\n"
        
        result += "Details - Aggiuntivi:\n"
        if self.details_additional:
            result += "\n".join(f"- {item}" for item in self.details_additional) + "\n"
        else:
            result += "- <none>\n"
        
        result = result.strip()
        if indent > 0:
            result = (indent * prefix) + result.replace("\n", "\n" + indent * prefix)
        return result


TEMPLATE = FILE_TEMPLATE.read_text(encoding="utf-8")


def get_prompt(question: str, *helps: str):
    template = ChatPromptTemplate.from_template(TEMPLATE)
    return template.invoke({
        "class_name": Answer.__name__,
        "question": question,
        "help": "\n\n".join(helps) if helps else "",
    })


def cache_file(question: Question):
    return DIR_SOLUTIONS / f"{question.id}.yaml"


def save_cache(
        question: Question,
        answer: Answer,
        helps: list[str] = None,
        model_name: str = None,
        model_provider: str = None):
    cache_file_path = cache_file(question)
    with open(cache_file_path, "w", encoding="utf-8") as f:
        print(f"# saving answer to {cache_file_path}")
        yaml = answer.model_dump()
        yaml["question"] = question.text
        yaml["helps"] = helps
        yaml["id"] = question.id
        if model_name:
            yaml["model_name"] = model_name
        if model_provider:
            yaml["model_provider"] = model_provider
        yaml["prompt_template"] = TEMPLATE
        safe_dump(yaml, f, sort_keys=True, allow_unicode=True)
        return yaml


def load_cache(question: Question) -> Answer | None:
    cache_file_path = cache_file(question)
    if not cache_file_path.exists():
        return None
    with open(cache_file_path, "r", encoding="utf-8") as f:
        print(f"# loading cached answer from {cache_file_path}")
        try:
            cached_answer = safe_load(f)
            return Answer(
                core=cached_answer.get("core", []),
                details_important=cached_answer.get("details_important", []),
                details_additional=cached_answer.get("details_additional", []),
            )
        except Exception as e:
            print(f"# error loading cached answer from {cache_file_path}: {e}")
            cache_file_path.unlink()
            return None

class SolutionProvider(AIOracle):
    def __init__(self, model_name: str = None, model_provider: str = "bge-large"):
        super().__init__(model_name, model_provider, Answer)
        self.__vector_store = sqlite_vector_store()
        self.__use_helps = self.__vector_store.get_dimensionality() > 0

    def answer(self, question: Question, max_helps=5) -> Answer:
        if (cache := load_cache(question)):
            return cache
        text = question.text
        helps = []
        if self.__use_helps:
            helps = [doc.page_content for doc in self.__vector_store.similarity_search(text, k=max_helps)]
        prompt = get_prompt(text, *helps)
        result = self.llm.invoke(prompt)
        if isinstance(result, Answer):
            save_cache( question, result, helps, self.model_name, self.model_provider)
            return result
        else:
            raise ValueError(f"Expected {Answer.__name__}, got {type(result)}: {result}")
