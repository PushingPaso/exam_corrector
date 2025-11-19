
import json
from dataclasses import dataclass, field
from typing import Dict


from exam import DIR_ROOT
from exam import get_questions_store, load_exam_from_yaml
from exam.assess import Assessor
from exam.solution import Answer, load_cache as load_answer_cache
from langchain.tools import tool


@dataclass
class AssessmentContext:
    """Context condiviso tra tool calls."""

    # Cache dei dati caricati
    loaded_answers: Dict[str, str] = field(default_factory=dict)
    loaded_checklists: Dict[str, Answer] = field(default_factory=dict)
    loaded_exams = {}

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
    I tool sono definiti come @staticmethod per operare sullo
    stato a livello di classe (es. ExamMCPServer.context) senza
    richiedere un'istanza ('self').
    """

    # --- STATO CONDIVISO A LIVELLO DI CLASSE ---
    questions_store = get_questions_store()
    context = AssessmentContext()

    evaluations_dir = DIR_ROOT / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)

    # Directory for YAML exam files
    exams_dir = DIR_ROOT / "static" / "se-exams"
    exams_dir.mkdir(parents=True, exist_ok=True)
    # -------------------------------------------

    """Create all available tools."""
    @staticmethod
    @tool
    async def load_checklist(question_ids: list[str]) -> str:
        """
        Load assessment checklists for a *list* of question IDs into memory.
        This tool processes a batch of IDs at once.
        Checklists will be available for other tools to use.

        Args:
            question_ids: A *list* of question IDs (e.g., ["CI-5", "GHA-1", "Python-15"])

        Returns:
            JSON summary of the batch operation (loaded, skipped, failed).
        """

        # Dizionario per tracciare i risultati
        results = {
            "loaded": [],
            "skipped_cache": [],
            "failed": []
        }

        # Elabora OGNI question_id nella lista
        for question_id in question_ids:
            try:
                # 1. Controlla se è già in cache
                cached = ExamMCPServer.context.get_checklist(question_id)
                if cached:
                    results["skipped_cache"].append(question_id)
                    continue  # Passa al prossimo ID, non fermarti

                question = ExamMCPServer.questions_store.question(question_id)
                checklist = load_answer_cache(question)

                if not checklist:
                    results["failed"].append({
                        "question_id": question_id,
                        "error": f"No checklist found"
                    })
                    continue  # Passa al prossimo ID

                ExamMCPServer.context.store_checklist(question_id, checklist)

                results["loaded"].append({
                    "question_id": question_id,
                    "core_items": len(checklist.core),
                    "important_items": len(checklist.details_important)
                })

            except Exception as e:
                results["failed"].append({
                    "question_id": question_id,
                    "error": str(e)
                })
                continue  # Passa al prossimo ID


        total_loaded = len(results["loaded"])
        total_skipped = len(results["skipped_cache"])
        total_failed = len(results["failed"])

        summary_message = f"Batch load complete. Loaded: {total_loaded}, Skipped (cached): {total_skipped}, Failed: {total_failed}."

        # Restituisci il riepilogo JSON
        return json.dumps({
            "status": "batch_completed",
            "message": summary_message,
            "details": results
        }, indent=2)

    @staticmethod
    @tool
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
                # ACCEDE ALLO STATO DELLA CLASSE
                exams_dir=ExamMCPServer.exams_dir
            )

            # Store in context
            exam_id = exam_data["exam_id"]
            # ACCEDE ALLO STATO DELLA CLASSE
            ExamMCPServer.context.loaded_exams[exam_id] = exam_data

            question_ids = [q["id"] for q in exam_data["questions"]]
            student_email = [q["email"] for q in exam_data["students"]]

            # 2. Crea un riepilogo pulito per l'LLM
            summary_output = {
                "status": "loaded",
                "exam_id": exam_id,
                "num_questions": len(exam_data["questions"]),
                "num_students": len(exam_data["students"]),
                "question_ids": question_ids,  # Invia solo gli ID
                "student_email": student_email,
                "message": "Exam data has been loaded into the server context. You must now use 'loading checklist tool' for each of the provided question_ids."
            }

            # 3. Restituisci il riepilogo
            return json.dumps(summary_output, indent=2)

            # --- !!! FINE MODIFICA !!! ---

        except FileNotFoundError as e:
            return json.dumps({
                "error": str(e),
                "hint": "Use list_available_exams to see available files"
            })
        except Exception as e:
            import traceback
            return json.dumps({"error": str(e), "traceback": traceback.format_exc()})

    @staticmethod
    @tool
    async def assess_student_exam(student_email: str) -> str:
        """
        Assess all responses for a single student from loaded exam.
        Results are automatically saved to evaluations/{email}/assessment.json


        Args:
            student_email: Student's email (can use first 20 chars)

        Returns:
            Complete assessment for all student's responses
        """
        try:
            student_email_clean = student_email.rstrip('.').strip()

            # Find student with FLEXIBLE matching
            student_data = None
            questions = None
            matched_email = None

            # ACCEDE ALLO STATO DELLA CLASSE
            for exam_data in ExamMCPServer.context.loaded_exams.values():
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
                # ACCEDE ALLO STATO DELLA CLASSE
                for exam_data in ExamMCPServer.context.loaded_exams.values():
                    available.extend([s["email"] for s in exam_data["students"][:5]])

                return json.dumps({
                    "error": f"Student not found: '{student_email_clean}'",
                    "searched_for": student_email_clean,
                    "available_students_sample": available,
                    "hint": "Use exact email or at least first 10 characters",
                    # ACCEDE ALLO STATO DELLA CLASSE
                    "num_loaded_students": sum(len(e["students"]) for e in ExamMCPServer.context.loaded_exams.values())
                })

            # Usa matched_email (email completa) per tutto il resto
            student_email_full = matched_email

            print(f"[ASSESS] Matched student: {student_email_full}")

            # =========================================================
            # REFACTORING: Assessor ora gestisce tutto (valutazione + salvataggio)
            # =========================================================

            # ACCEDE ALLO STATO DELLA CLASSE
            assessor = Assessor(evaluations_dir=ExamMCPServer.evaluations_dir)

            result = await assessor.assess_student_exam(
                student_email=student_email_full,
                exam_questions=questions,
                student_responses=student_data["responses"],
                questions_store=ExamMCPServer.questions_store,
                context=ExamMCPServer.context,
                original_grades=student_data.get("original_grades", {})
            )

            return json.dumps(result, indent=2)

        except Exception as e:
            import traceback
            return json.dumps({"error": str(e), "traceback": traceback.format_exc()})