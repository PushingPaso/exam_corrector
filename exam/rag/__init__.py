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
        model: Model identifier or size hint
               Recommended: 'bge-large' for best accuracy
               Default: bge-base (balance accuracy/speed)

    Returns:
        HuggingFaceEmbeddings instance
    """
    if not model:
        model = "bge-large"  # Default più potente del vecchio

    model = model.lower()

    # STATO DELL'ARTE 2024-2025
    if model == "bge-large" or model == "best":
        # Massima accuratezza (MTEB: 64.2)
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {
            'device': 'cpu',  # o 'cuda' se GPU disponibile
        }
        encode_kwargs = {
            'normalize_embeddings': True,
            'batch_size': 32,  # Ottimizzato per large model
        }

    elif model == "bge-base" or model == "recommended":
        # Ottimo compromesso accuratezza/velocità (MTEB: 63.5)
        model_name = "BAAI/bge-base-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {
            'normalize_embeddings': True,
            'batch_size': 64,
        }

    elif model == "bge-small" or model == "fast":
        # Veloce ma comunque superiore a MiniLM (MTEB: 62.1)
        model_name = "BAAI/bge-small-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {
            'normalize_embeddings': True,
            'batch_size': 128,
        }

    # ALTERNATIVE (per confronto nella tesi)
    elif model == "nomic":
        # Open source + ottimo per long context
        model_name = "nomic-ai/nomic-embed-text-v1"
        model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
        encode_kwargs = {'normalize_embeddings': True}

    elif model == "gte-large":
        # Alibaba, molto buono per retrieval
        model_name = "thenlper/gte-large"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}

    # LEGACY (per baseline comparison)
    elif model == "legacy-small" or "mini" in model:
        # Il tuo attuale default (BASELINE)
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}

    elif model == "legacy-large" or "mpnet" in model:
        # Il tuo attuale "large" (BASELINE)
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}

    elif model.startswith("BAAI/") or model.startswith("sentence-transformers/") or "/" in model:
        # Direct model name provided
        model_name = model
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}

    else:
        raise ValueError(
            f"Unknown model hint: {model}. "
            "Use 'bge-large', 'bge-base', 'bge-small', 'nomic', 'gte-large', "
            "'legacy-small', 'legacy-large', or a full HuggingFace model name."
        )

    print(f"# Loading embeddings model: {model_name}")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
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