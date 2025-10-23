import os
import re
import sys
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from exam import DIR_ROOT
from exam import get_questions_store
from exam.solution import Answer

OUTPUT_FILE = os.getenv("OUTPUT_FILE", None)
if OUTPUT_FILE:
    OUTPUT_FILE = open(OUTPUT_FILE, "w", encoding="utf-8")
else:
    OUTPUT_FILE = sys.stdout

ALL_QUESTIONS = get_questions_store()
PATTERN_QUESTION_FOLDER = re.compile(r"^Q\d+\s+-\s+(\w+-\d+)$")
FILE_TEMPLATE = DIR_ROOT / "exam" / "assess" / "prompt-template.txt"
TEMPLATE = FILE_TEMPLATE.read_text(encoding="utf-8")


def calculate_score_from_assessments(assessments: dict, max_score: float) -> tuple[float, str, dict]:
    """
    Calcola il punteggio da un dizionario di assessment.
    Sistema: 70% Core + 30% Important

    Args:
        assessments: dict[Feature, FeatureAssessment]
        max_score: Punteggio massimo della domanda

    Returns:
        tuple(score, breakdown, stats): Punteggio, spiegazione, e statistiche dettagliate
    """
    if not assessments:
        return 0.0, "No features assessed", {}

    # Conta feature per tipo
    core_total = sum(1 for f in assessments if f.type == FeatureType.CORE)
    core_satisfied = sum(1 for f, a in assessments.items()
                         if f.type == FeatureType.CORE and a.satisfied)

    important_total = sum(1 for f in assessments if f.type == FeatureType.DETAILS_IMPORTANT)
    important_satisfied = sum(1 for f, a in assessments.items()
                              if f.type == FeatureType.DETAILS_IMPORTANT and a.satisfied)

    # Calcolo percentuali per categoria
    core_percentage = (core_satisfied / core_total * 0.70) if core_total > 0 else 0.0
    important_percentage = (important_satisfied / important_total * 0.30) if important_total > 0 else 0.30

    # Percentuale finale
    final_percentage = core_percentage + important_percentage

    # Score finale
    score = round(final_percentage * max_score, 2)

    # Breakdown dettagliato
    breakdown_parts = []

    if core_total > 0:
        breakdown_parts.append(
            f"Core: {core_satisfied}/{core_total} "
            f"({core_percentage / 0.70 * 100:.0f}% → {core_percentage * 100:.0f}%)"
        )

    if important_total > 0:
        breakdown_parts.append(
            f"Important: {important_satisfied}/{important_total} "
            f"({important_percentage / 0.20 * 100:.0f}% → {important_percentage * 100:.0f}%)"
        )

    breakdown = " + ".join(breakdown_parts)
    breakdown += f" = {final_percentage * 100:.0f}% of {max_score} = {score}"

    # Statistiche dettagliate
    stats = {
        "core": {
            "total": core_total,
            "satisfied": core_satisfied,
            "percentage": round((core_satisfied / core_total * 100) if core_total > 0 else 0, 1)
        },
        "details_important": {
            "total": important_total,
            "satisfied": important_satisfied,
            "percentage": round((important_satisfied / important_total * 100) if important_total > 0 else 0, 1)
        }
    }

    return score, breakdown, stats

class FeatureType(str, Enum):
    """Enumeration of feature types that can be assessed in a question's answer."""
    CORE = "core"
    DETAILS_IMPORTANT = "important detail"

@dataclass(frozen=True)
class Feature:
    # Type of the feature
    type: FeatureType

    # Description of the feature
    description: str

    @property
    def verb_ideal(self) -> str:
        return "should be present"

    @property
    def verb_actual(self) -> str:
        return "is actually present"

    @property
    def is_core(self) -> bool:
        """Determina se questa feature è core (essenziale)."""
        return self.type == FeatureType.CORE

    @property
    def weight_percentage(self) -> float:
        """Restituisce il peso percentuale di questa feature nel punteggio totale."""
        if self.type == FeatureType.CORE:
            return 0.70  # 70% del punteggio
        elif self.type == FeatureType.DETAILS_IMPORTANT:
            return 0.20  # 20% del punteggio
        else:  # DETAILS_ADDITIONAL
            return 0.10  # 10% del punteggio


def enumerate_features(answer: Answer):
    """Enumera le features da valutare."""
    if not answer:
        return
    i = 0

    # CORE - elementi essenziali
    for core_item in answer.core:
        yield i, Feature(type=FeatureType.CORE, description=core_item)
        i += 1

    # DETAILS_IMPORTANT - dettagli importanti
    for detail in answer.details_important:
        yield i, Feature(type=FeatureType.DETAILS_IMPORTANT, description=detail)
        i += 1


class FeatureAssessment(BaseModel):
    satisfied: bool = Field(description="Whether the feature is present in the answer")
    motivation: str = Field(description="Explanation of why the feature is present or not")



