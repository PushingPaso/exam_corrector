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


class Assessor:
    """
    Classe per la valutazione strutturata delle risposte degli studenti.
    Separa la logica di assessment dal layer MCP per una migliore modularità.
    """

    def __init__(self):
        """
        Inizializza l'assessor con il modello LLM specificato.

        Args:
            model_name: Nome del modello da usare (default: llama-3.3-70b-versatile)
        """
        from exam.llm_provider import llm_client
        self.llm_client_func = llm_client

    async def assess_single_answer(
            self,
            question,
            checklist,
            student_response: str,
            max_score: float
    ) -> dict:
        """
        Valuta una singola risposta dello studente.

        Args:
            question: Oggetto Question con id e text
            checklist: Oggetto Answer con core e details_important
            student_response: Testo della risposta dello studente
            max_score: Punteggio massimo per questa domanda

        Returns:
            dict con:
                - status: "assessed" | "error" | "no_response"
                - score: float
                - max_score: float
                - statistics: dict con statistiche per tipo di feature
                - breakdown: str con spiegazione del calcolo
                - feature_assessments: list di assessment per ogni feature
                - error: str (solo se status="error")
        """
        if not student_response or student_response.strip() == '-':
            return {
                "status": "no_response",
                "score": 0.0,
                "max_score": max_score
            }

        try:
            # Valuta ogni feature
            feature_assessments_list = []
            feature_assessments_dict = {}

            for index, feature in enumerate_features(checklist):
                # Prepara il prompt
                prompt = TEMPLATE.format(
                    class_name="FeatureAssessment",
                    question=question.text,
                    feature_type=feature.type.value,
                    feature_verb_ideal=feature.verb_ideal,
                    feature_verb_actual=feature.verb_actual,
                    feature=feature.description,
                    answer=student_response
                )

                # Chiama il modello LLM
                llm, _, _ = self.llm_client_func(structured_output=FeatureAssessment)
                result = llm.invoke(prompt)

                # Salva risultati
                feature_assessments_list.append({
                    "feature": feature.description,
                    "feature_type": feature.type.name,
                    "satisfied": result.satisfied,
                    "motivation": result.motivation
                })

                feature_assessments_dict[feature] = result

            # Calcola il punteggio
            score, breakdown, stats = self.calculate_score(
                feature_assessments_dict,
                max_score
            )

            return {
                "status": "assessed",
                "score": score,
                "max_score": max_score,
                "statistics": stats,
                "breakdown": breakdown,
                "feature_assessments": feature_assessments_list
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "score": 0.0,
                "max_score": max_score
            }

    async def assess_student_exam(
            self,
            student_email: str,
            exam_questions: list,
            student_responses: dict,
            questions_store,
            context
    ) -> dict:
        """
        Valuta tutte le risposte di uno studente.

        Args:
            student_email: Email dello studente
            exam_questions: Lista di dict con question info (id, number, text, score)
            student_responses: Dict {question_number: response_text}
            questions_store: QuestionsStore instance
            context: AssessmentContext per accedere alle checklist

        Returns:
            dict con:
                - student_email: str
                - calculated_score: float
                - max_score: float
                - percentage: float
                - scoring_system: str
                - assessments: list di assessment per ogni domanda
        """
        from exam.solution import load_cache as load_answer_cache

        assessments = []
        total_score = 0.0
        total_max_score = 0.0

        for question_info in exam_questions:
            question_num = int(question_info["number"].replace("Question ", ""))

            # Verifica se lo studente ha risposto
            if question_num not in student_responses:
                assessments.append({
                    "question_number": question_num,
                    "question_id": question_info["id"],
                    "question_text": question_info.get("text", ""),
                    "status": "no_response",
                    "score": 0.0,
                    "max_score": question_info["score"]
                })
                total_max_score += question_info["score"]
                continue

            try:
                # Ottieni la domanda e la checklist
                question = questions_store.question(question_info["id"])
                checklist = context.get_checklist(question_info["id"])

                if not checklist:
                    # Prova a caricare la checklist se non in context
                    checklist = load_answer_cache(question)
                    if checklist:
                        context.store_checklist(question_info["id"], checklist)

                if not checklist:
                    raise ValueError(f"No checklist found for question {question_info['id']}")

                response_text = student_responses[question_num]

                # Valuta la singola risposta
                assessment = await self.assess_single_answer(
                    question=question,
                    checklist=checklist,
                    student_response=response_text,
                    max_score=question_info["score"]
                )

                # Aggiungi metadati
                assessment.update({
                    "question_number": question_num,
                    "question_id": question_info["id"],
                    "question_text": question.text,
                    "student_response": response_text
                })

                assessments.append(assessment)

                total_score += assessment["score"]
                total_max_score += question_info["score"]

            except Exception as e:
                assessments.append({
                    "question_number": question_num,
                    "question_id": question_info["id"],
                    "question_text": question_info.get("text", ""),
                    "status": "error",
                    "error": str(e),
                    "score": 0.0,
                    "max_score": question_info["score"]
                })
                total_max_score += question_info["score"]

        return {
            "student_email": student_email,
            "calculated_score": total_score,
            "max_score": total_max_score,
            "percentage": round((total_score / total_max_score * 100) if total_max_score > 0 else 0, 1),
            "scoring_system": "70% Core + 30% Important_Details",
            "assessments": assessments
        }

    def calculate_score(self, assessments: dict, max_score: float) -> tuple[float, str, dict]:
        """
        Calcola il punteggio da un dizionario di assessment.
        Sistema:
        - 70% Core + 30% Important (se entrambi presenti)
        - 100% Core (se mancano Important)
        - 100% Important (se mancano Core - caso raro)

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

        # Determina i pesi in base a cosa è presente
        if core_total > 0 and important_total > 0:
            # Entrambi presenti: 70% core + 30% important
            core_weight = 0.70
            important_weight = 0.30
            scoring_system = "70% Core + 30% Important"
        elif core_total > 0:
            # Solo core: 100% core
            core_weight = 1.0
            important_weight = 0.0
            scoring_system = "100% Core (no Important details)"
        elif important_total > 0:
            # Solo important (caso raro): 100% important
            core_weight = 0.0
            important_weight = 1.0
            scoring_system = "100% Important (no Core - unusual)"
        else:
            # Nessuna feature
            return 0.0, "No features assessed", {}

        # Calcolo percentuali per categoria
        core_percentage = (core_satisfied / core_total * core_weight) if core_total > 0 else 0.0
        important_percentage = (important_satisfied / important_total * important_weight) if important_total > 0 else 0.0

        # Percentuale finale
        final_percentage = core_percentage + important_percentage

        # Score finale
        score = round(final_percentage * max_score, 2)

        # Breakdown dettagliato
        breakdown_parts = []

        if core_total > 0:
            core_raw_pct = (core_satisfied / core_total * 100)
            core_weighted_pct = (core_percentage * 100)
            breakdown_parts.append(
                f"Core: {core_satisfied}/{core_total} "
                f"({core_raw_pct:.0f}% → {core_weighted_pct:.0f}%)"
            )

        if important_total > 0:
            important_raw_pct = (important_satisfied / important_total * 100)
            important_weighted_pct = (important_percentage * 100)
            breakdown_parts.append(
                f"Important: {important_satisfied}/{important_total} "
                f"({important_raw_pct:.0f}% → {important_weighted_pct:.0f}%)"
            )

        breakdown = " + ".join(breakdown_parts)
        breakdown += f" = {final_percentage * 100:.0f}% of {max_score} = {score}"
        breakdown += f" [{scoring_system}]"

        # Statistiche dettagliate
        stats = {
            "core": {
                "total": core_total,
                "satisfied": core_satisfied,
                "percentage": round((core_satisfied / core_total * 100) if core_total > 0 else 0, 1),
                "weight": core_weight
            },
            "details_important": {
                "total": important_total,
                "satisfied": important_satisfied,
                "percentage": round((important_satisfied / important_total * 100) if important_total > 0 else 0, 1),
                "weight": important_weight
            },
            "scoring_system": scoring_system
        }

        return score, breakdown, stats