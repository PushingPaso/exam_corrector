import os
import re
import sys
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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
    Include logica di assessment E salvataggio dei risultati.
    """

    def __init__(self, evaluations_dir=None):
        """
        Inizializza l'assessor con il modello LLM specificato.

        Args:
            evaluations_dir: Directory per salvare le valutazioni (default: DIR_ROOT/evaluations)
        """
        from exam.llm_provider import llm_client
        self.llm_client_func = llm_client

        # Setup evaluations directory
        if evaluations_dir is None:
            self.evaluations_dir = DIR_ROOT / "evaluations"
        else:
            self.evaluations_dir = Path(evaluations_dir)

        self.evaluations_dir.mkdir(parents=True, exist_ok=True)

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
            context,
            save_results: bool = True,
            original_grades: dict = None  # ← AGGIUNTO!
    ) -> dict:
        """
        Valuta tutte le risposte di uno studente.

        REFACTORIZZATO: Ora include la logica di salvataggio dei risultati.

        Args:
            student_email: Email dello studente
            exam_questions: Lista di dict con question info (id, number, text, score)
            student_responses: Dict {question_number: response_text}
            questions_store: QuestionsStore instance
            context: AssessmentContext per accedere alle checklist
            save_results: Se True, salva i risultati su file (default: True)

        Returns:
            dict con:
                - student_email: str
                - calculated_score: float
                - max_score: float
                - percentage: float
                - scoring_system: str
                - assessments: list di assessment per ogni domanda
                - saved_files: dict con percorsi dei file salvati (se save_results=True)
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

        result = {
            "student_email": student_email,
            "calculated_score": total_score,
            "max_score": total_max_score,
            "percentage": round((total_score / total_max_score * 100) if total_max_score > 0 else 0, 1),
            "scoring_system": "70% Core + 30% Important_Details",
            "assessments": assessments,
            "original_grades": original_grades if original_grades else {}
        }

        # =========================================================
        # NUOVA LOGICA: Salvataggio risultati (spostata da MCP)
        # =========================================================
        if save_results:
            saved_files = self._save_assessment_results(student_email, result, exam_questions)
            result["saved_files"] = saved_files

        return result

    def _save_assessment_results(self, student_email: str, result: dict, exam_questions: list) -> dict:
        """
        Salva i risultati della valutazione su file.

        NUOVA FUNZIONE: Logica di salvataggio estratta da MCP server.

        Args:
            student_email: Email dello studente
            result: Dizionario con i risultati della valutazione
            exam_questions: Lista delle domande dell'esame (per original grades)

        Returns:
            dict con percorsi dei file salvati
        """
        # Crea directory studente
        student_dir = self.evaluations_dir / student_email
        student_dir.mkdir(parents=True, exist_ok=True)

        # Salva assessment completo in JSON
        assessment_file = student_dir / "assessment.json"
        with open(assessment_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Salva summary leggibile
        summary_file = student_dir / "summary.txt"
        summary_content = self._generate_summary_text(student_email, result, exam_questions)

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)

        return {
            "assessment": str(assessment_file),
            "summary": str(summary_file)
        }

    def _generate_summary_text(self, student_email: str, result: dict, exam_questions: list) -> str:
        """
        Genera il testo del summary leggibile.

        NUOVA FUNZIONE: Logica di generazione summary estratta da MCP server.
        """
        lines = []
        lines.append("STUDENT ASSESSMENT SUMMARY")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Student: {student_email}")
        lines.append(f"Calculated Score: {result['calculated_score']:.2f}/{result['max_score']}")
        lines.append(f"Calculated Percentage: {result['percentage']}%")

        # Get original grades if available from first assessment
        original_grades = result.get('original_grades', {})

        if original_grades:
            original_total = original_grades.get("total_grade", 0)
            lines.append(f"Original Moodle Grade: {original_total:.2f}/27.00")

            score_diff = result['calculated_score'] - original_total
            diff_text = f"Difference: {score_diff:+.2f} "
            if abs(score_diff) < 0.5:
                diff_text += "( Very close)"
            elif abs(score_diff) < 2.0:
                diff_text += "( Reasonable)"
            else:
                diff_text += "(Significant difference)"
            lines.append(diff_text)

        lines.append(f"Scoring System: {result['scoring_system']}")
        lines.append("")
        lines.append("=" * 70)
        lines.append("")

        for assessment in result["assessments"]:
            question_num = assessment['question_number']
            lines.append(f"Question {question_num}: {assessment['question_id']}")
            lines.append("-" * 70)

            if assessment['status'] == 'assessed':
                lines.append(f"Calculated Score: {assessment['score']:.2f}/{assessment['max_score']}")

                # Add comparison with original grade if available
                if original_grades and 'question_grades' in original_grades:
                    orig_q_grade = original_grades['question_grades'].get(question_num)
                    if orig_q_grade is not None:
                        diff = assessment['score'] - orig_q_grade
                        lines.append(f"Original Grade: {orig_q_grade:.2f}/{assessment['max_score']}")
                        lines.append(f"Difference: {diff:+.2f}")

                lines.append(f"Breakdown: {assessment['breakdown']}")
                lines.append("")

                # Raggruppa per tipo
                core_features = [fa for fa in assessment['feature_assessments']
                                 if fa['feature_type'] == 'CORE']
                important_features = [fa for fa in assessment['feature_assessments']
                                      if fa['feature_type'] == 'DETAILS_IMPORTANT']

                if core_features:
                    lines.append("CORE Elements:")
                    for fa in core_features:
                        status = "✓ OK" if fa['satisfied'] else "✗ MISSING"
                        lines.append(f"  [{status}] {fa['feature']}")
                        lines.append(f"       {fa['motivation']}")
                        lines.append("")

                if important_features:
                    lines.append("Important Details:")
                    for fa in important_features:
                        status = "✓ OK" if fa['satisfied'] else "✗ MISSING"
                        lines.append(f"  [{status}] {fa['feature']}")
                        lines.append(f"       {fa['motivation']}")
                        lines.append("")

            else:
                lines.append(f"Status: {assessment['status']}")
                if 'error' in assessment:
                    lines.append(f"Error: {assessment['error']}")

            lines.append("")
            lines.append("=" * 70)
            lines.append("")

        return "\n".join(lines)

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
        important_percentage = (
                    important_satisfied / important_total * important_weight) if important_total > 0 else 0.0

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