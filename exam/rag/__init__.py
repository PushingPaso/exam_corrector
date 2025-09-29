from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import SQLiteVec
from exam import DIR_ROOT
from pydantic import BaseModel
import re


DIR_CONTENT = DIR_ROOT / "content"
FILE_DB = DIR_ROOT / "slides-rag.db"
MARKDOWN_FILES = list(DIR_CONTENT.glob("**/_index.md"))
REGEX_SLIDE_DELIMITER = re.compile(r"^\s*(---|\+\+\+)")


class Slide(BaseModel):
    content: str
    source: str
    lines: tuple[int, int]
    index: int

    @property
    def lines_count(self):
        return self.content.count("\n") + 1 if self.content else 0


def all_slides(files = None):
    if files is None:
        files = MARKDOWN_FILES
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            slide_beginning_line_num = 0
            line_number = 0
            slide_lines = []
            slide_index = 0
            last_was_blank = False
            for line in f.readlines():
                line_number += 1
                if REGEX_SLIDE_DELIMITER.match(line):
                    if slide_lines:
                        yield Slide(
                            content="\n".join(slide_lines),
                            source=str(file.relative_to(DIR_CONTENT)),
                            lines=(slide_beginning_line_num, line_number - 1),
                            index=slide_index,
                        )
                        slide_index += 1
                    slide_lines = []
                    slide_beginning_line_num = line_number + 1
                else:
                    if (stripped := line.strip()) or not last_was_blank:
                        slide_lines.append(line.rstrip())
                    last_was_blank = not stripped
            yield Slide(
                content="\n".join(slide_lines),
                source=str(file.relative_to(DIR_CONTENT)),
                lines=(slide_beginning_line_num, line_number - 1),
                index=slide_index,
            )


def huggingface_embeddings(model=None):
    """
    Creates HuggingFace embeddings model.
    
    Args:
        model: Model identifier or size hint (small/large/multilingual)
               Default: sentence-transformers/all-MiniLM-L6-v2
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    if not model:
        model = "small"
    
    model = model.lower()
    
    # Map model hints to actual HuggingFace model names
    if model == "small" or "mini" in model:
        # Fast, lightweight model - good for most use cases
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    elif model == "large" or "mpnet" in model:
        # More accurate but slower
        model_name = "sentence-transformers/all-mpnet-base-v2"
    elif model == "multilingual" or "multi" in model:
        # Multilingual support (English + Italian)
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    elif model.startswith("sentence-transformers/") or "/" in model:
        # Direct model name provided
        model_name = model
    else:
        raise ValueError(
            f"Unknown model hint: {model}. "
            "Use 'small', 'large', 'multilingual', or a full HuggingFace model name."
        )
    
    print(f"# Loading embeddings model: {model_name}")
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': True}  # For better similarity search
    )


def sqlite_vector_store(
        db_file: str = str(FILE_DB), 
        model: str = None, 
        table_name: str = "se_slides"):
    """
    Creates or loads a SQLite vector store with HuggingFace embeddings.
    
    Args:
        db_file: Path to SQLite database file
        model: Embedding model hint or name
        table_name: Name of the table in the database
    
    Returns:
        SQLiteVec instance
    """
    embeddings = huggingface_embeddings(model)
    
    return SQLiteVec(
        db_file=db_file,
        embedding=embeddings,
        table=table_name,
        connection=None,
    )