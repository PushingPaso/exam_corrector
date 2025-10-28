"""
MCP Server con Context Condiviso per collaborazione tra tool.
REFACTORIZZATO: Logica di business spostata nei moduli appropriati.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from exam import get_questions_store, load_exam_from_yaml
from exam.solution import Answer, load_cache as load_answer_cache


@dataclass
class AssessmentContext:
    """Context condiviso tra tool calls."""

    # Cache dei dati caricati
    loaded_answers: Dict[str, str] = field(default_factory=dict)
    loaded_checklists: Dict[str, Answer] = field(default_factory=dict)

    # Risultati delle valutazioni
    feature_assessments: Dict[str, list] = field(default_factory=dict)

    def get_session_id(self, question_id: str, student_code: str) -> str:
        """Genera un ID univoco per una sessione di valutazione."""
        return f"{question_id}_{student_code}"

    def store_answer(self, question_id: str, student_code: str, answer: str):
        """Salva una risposta nel context."""
        key = f"{question_id}_{student_code}"
        self.loaded_answers[key] = answer
        return key

    def get_answer(self, question_id: str, student_code: str) -> str | None:
        """Recupera una risposta dal context."""
        key = f"{question_id}_{student_code}"
        return self.loaded_answers.get(key)

    def store_checklist(self, question_id: str, checklist: Answer):
        """Salva una checklist nel context."""
        self.loaded_checklists[question_id] = checklist

    def get_checklist(self, question_id: str) -> Answer | None:
        """Recupera una checklist dal context."""
        return self.loaded_checklists.get(question_id)

    def store_assessments(self, question_id: str, student_code: str, assessments: list):
        """Salva le valutazioni nel context."""
        session_id = self.get_session_id(question_id, student_code)
        self.feature_assessments[session_id] = assessments

    def get_assessments(self, question_id: str, student_code: str) -> list | None:
        """Recupera le valutazioni dal context."""
        session_id = self.get_session_id(question_id, student_code)
        return self.feature_assessments.get(session_id)


class ExamMCPServer:
    """
    MCP Server con context condiviso per collaborazione tra tool.
    REFACTORIZZATO: Ora è solo un layer di orchestrazione.
    """

    def __init__(self):
        self.questions_store = get_questions_store()
        self.context = AssessmentContext()
        self.context.loaded_exams = {}  # For batch exam processing

        from exam import DIR_ROOT
        self.evaluations_dir = DIR_ROOT / "evaluations"
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)

        # Directory for YAML exam files
        self.exams_dir = DIR_ROOT / "static" / "se-exams"
        self.exams_dir.mkdir(parents=True, exist_ok=True)

        self.tools = self._create_tools()

    def _create_tools(self):
        """Create all available tools."""
        tools = {}

        # TOOL: Load Checklist (ATOMICO)
        async def load_checklist(question_id: str) -> str:
            """
            Load the assessment checklist for a question into memory.
            The checklist will be available for other tools to use.

            Args:
                question_id: The question ID (e.g., "CI-5")

            Returns:
                JSON with checklist summary
            """
            # Check if already loaded
            cached = self.context.get_checklist(question_id)
            if cached:
                return json.dumps({
                    "status": "already_loaded",
                    "question_id": question_id,
                    "core_count": len(cached.core),
                    "important_count": len(cached.details_important),
                })

            try:
                question = self.questions_store.question(question_id)
                checklist = load_answer_cache(question)

                if not checklist:
                    return json.dumps({"error": f"No checklist found for question {question_id}"})

                # Store in context
                self.context.store_checklist(question_id, checklist)

                return json.dumps({
                    "status": "loaded",
                    "question_id": question_id,
                    "question_text": question.text,
                    "features": {
                        "core": len(checklist.core),
                        "important": len(checklist.details_important),
                    },
                    "core_items": checklist.core,
                    "important_items": checklist.details_important,
                    "message": "Checklist loaded into memory. Use assess_all_features to evaluate."
                })
            except Exception as e:
                return json.dumps({"error": str(e)})

        tools["load_checklist"] = load_checklist

        # TOOL: Load Exam from YAML (REFACTORIZZATO)
        async def load_exam_from_yaml_tool(questions_file: str, responses_file: str, grades_file: str = None) -> str:
            """
            Load an entire exam from YAML files in static/se-exams directory.

            REFACTORIZZATO: Ora usa la funzione load_exam_from_yaml dal modulo exam.

            Args:
                questions_file: Filename of questions YAML (e.g., "se-2025-06-05-questions.yml")
                responses_file: Filename of responses YAML (e.g., "se-2025-06-05-responses.yml")
                grades_file: Optional filename of grades YAML (e.g., "se-2025-06-05-grades.yml")

            Files are loaded from static/se-exams/ directory automatically.

            Returns:
                JSON with exam structure
            """
            try:
                # Usa la funzione refactorizzata
                exam_data = load_exam_from_yaml(
                    questions_file=questions_file,
                    responses_file=responses_file,
                    grades_file=grades_file,
                    exams_dir=self.exams_dir
                )

                # Store in context
                exam_id = exam_data["exam_id"]
                self.context.loaded_exams[exam_id] = exam_data

                return json.dumps({
                    "exam_id": exam_id,
                    "loaded_from": str(self.exams_dir),
                    "questions_file": Path(exam_data["files"]["questions"]).name,
                    "responses_file": Path(exam_data["files"]["responses"]).name,
                    "grades_file": Path(exam_data["files"]["grades"]).name if exam_data["files"]["grades"] else None,
                    "num_questions": len(exam_data["questions"]),
                    "num_students": len(exam_data["students"]),
                    "questions": exam_data["questions"],
                    "students_preview": [
                        {
                            "email": s["email"],
                            "num_responses": s["num_responses"],
                            "time_taken": s["time_taken"]
                        }
                        for s in exam_data["students"][:5]
                    ],
                    "message": f"Loaded exam with {len(exam_data['questions'])} questions and {len(exam_data['students'])} students from {self.exams_dir}"
                }, indent=2)

            except FileNotFoundError as e:
                return json.dumps({
                    "error": str(e),
                    "hint": "Use list_available_exams to see available files"
                })
            except Exception as e:
                import traceback
                return json.dumps({"error": str(e), "traceback": traceback.format_exc()})

        tools["load_exam_from_yaml"] = load_exam_from_yaml_tool

        # TOOL: Assess Student Exam (REFACTORIZZATO)
        async def assess_student_exam(student_email: str) -> str:
            """
            Assess all responses for a single student from loaded exam.
            Results are automatically saved to evaluations/{email}/assessment.json

            REFACTORIZZATO: Ora la classe Assessor gestisce anche il salvataggio.

            Args:
                student_email: Student's email (can use first 20 chars)

            Returns:
                Complete assessment for all student's responses
            """
            try:
                from exam.assess import Assessor

                student_email_clean = student_email.rstrip('.').strip()

                # Find student with FLEXIBLE matching
                student_data = None
                questions = None
                matched_email = None

                for exam_data in self.context.loaded_exams.values():
                    for student in exam_data["students"]:
                        full_email = student["email"]

                        if (full_email.lower() == student_email_clean.lower() or
                                (len(student_email_clean) >= 10 and
                                 full_email.lower().startswith(student_email_clean.lower()))):
                            student_data = student
                            questions = exam_data["questions"]
                            matched_email = full_email
                            break

                    if student_data:
                        break

                if not student_data:
                    # DEBUG: mostra studenti disponibili
                    available = []
                    for exam_data in self.context.loaded_exams.values():
                        available.extend([s["email"] for s in exam_data["students"][:5]])

                    return json.dumps({
                        "error": f"Student not found: '{student_email_clean}'",
                        "searched_for": student_email_clean,
                        "available_students_sample": available,
                        "hint": "Use exact email or at least first 10 characters",
                        "num_loaded_students": sum(len(e["students"]) for e in self.context.loaded_exams.values())
                    })

                # Usa matched_email (email completa) per tutto il resto
                student_email_full = matched_email

                print(f"[ASSESS] Matched student: {student_email_full}")

                # =========================================================
                # REFACTORING: Assessor ora gestisce tutto (valutazione + salvataggio)
                # =========================================================

                assessor = Assessor(evaluations_dir=self.evaluations_dir)

                result = await assessor.assess_student_exam(
                    student_email=student_email_full,
                    exam_questions=questions,
                    student_responses=student_data["responses"],
                    questions_store=self.questions_store,
                    context=self.context,
                    save_results=True  # Assessor salverà i risultati
                )

                # Aggiungi metadati Moodle (ancora gestito qui per ora)
                result["moodle_grade"] = student_data.get("moodle_grade")

                return json.dumps(result, indent=2)

            except Exception as e:
                import traceback
                return json.dumps({"error": str(e), "traceback": traceback.format_exc()})

        tools["assess_student_exam"] = assess_student_exam

        return tools