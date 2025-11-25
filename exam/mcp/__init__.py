"""
MCP Server with Shared Context - LEAN VERSION.
Optimized to reduce token usage by returning concise summaries.
"""

import json
from dataclasses import dataclass, field
from typing import Dict

from langchain.tools import tool

from exam import DIR_ROOT
from exam import get_questions_store, load_exam_from_yaml
from exam.assess import Assessor
from exam.solution import Answer, load_cache as load_answer_cache


@dataclass
class AssessmentContext:
    """Shared context between tool calls."""
    # Cache of loaded data
    loaded_answers: Dict[str, str] = field(default_factory=dict)
    loaded_checklists: Dict[str, Answer] = field(default_factory=dict)
    loaded_exams = {}

    # Assessment results
    feature_assessments: Dict[str, list] = field(default_factory=dict)

    def get_session_id(self, question_id: str, student_code: str) -> str:
        return f"{question_id}_{student_code}"

    def store_answer(self, question_id: str, student_code: str, answer: str):
        key = f"{question_id}_{student_code}"
        self.loaded_answers[key] = answer
        return key

    def get_answer(self, question_id: str, student_code: str) -> str | None:
        key = f"{question_id}_{student_code}"
        return self.loaded_answers.get(key)

    def store_checklist(self, question_id: str, checklist: Answer):
        self.loaded_checklists[question_id] = checklist

    def get_checklist(self, question_id: str) -> Answer | None:
        return self.loaded_checklists.get(question_id)

    def store_assessments(self, question_id: str, student_code: str, assessments: list):
        session_id = self.get_session_id(question_id, student_code)
        self.feature_assessments[session_id] = assessments

    def get_assessments(self, question_id: str, student_code: str) -> list | None:
        session_id = self.get_session_id(question_id, student_code)
        return self.feature_assessments.get(session_id)


class ExamMCPServer:
    """
    MCP Server with shared context.
    Tools are static methods to operate on class-level state.
    """

    # --- SHARED CLASS STATE ---
    questions_store = get_questions_store()
    context = AssessmentContext()

    evaluations_dir = DIR_ROOT / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)

    # Directory for YAML exam files
    exams_dir = DIR_ROOT / "static" / "se-exams"
    exams_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------

    @staticmethod
    @tool("list_loaded_students_tool")
    def list_students() -> str:
        """
        Retrieve the list of all student emails currently loaded in the exam context.
        """
        students = []
        for exam_data in ExamMCPServer.context.loaded_exams.values():
            for student in exam_data["students"]:
                students.append(student["email"])

        if not students:
            return json.dumps({"error": "No students loaded. Did you run load_exam_tool first?"})

        return json.dumps(students, indent=2)

    @staticmethod
    @tool("load_checklist_tool")
    async def load_checklist(question_ids: list[str]) -> str:
        """
        Load assessment checklists for a *list* of question IDs into memory.
        """
        results = {
            "loaded": 0,
            "skipped_cache": 0,
            "failed": []
        }

        for question_id in question_ids:
            try:
                cached = ExamMCPServer.context.get_checklist(question_id)
                if cached:
                    results["skipped_cache"] += 1
                    continue

                question = ExamMCPServer.questions_store.question(question_id)
                checklist = load_answer_cache(question)

                if not checklist:
                    results["failed"].append(question_id)
                    continue

                ExamMCPServer.context.store_checklist(question_id, checklist)
                results["loaded"] += 1

            except Exception as e:
                results["failed"].append(f"{question_id} ({str(e)})")
                continue

        status = "batch_completed" if not results["failed"] else "completed_with_errors"

        return json.dumps({
            "status": status,
            "summary": f"Loaded: {results['loaded']}, Cached: {results['skipped_cache']}, Failed: {len(results['failed'])}",
            "failed_ids": results["failed"]  # Restituiamo solo gli errori, che sono importanti
        })

    @staticmethod
    @tool("load_exam_from_yaml_tool")
    async def load_exam_from_yaml_tool(questions_file: str, responses_file: str, grades_file: str = None) -> str:
        """
        Load an entire exam from YAML files.
        """
        try:
            exam_data = load_exam_from_yaml(
                questions_file=questions_file,
                responses_file=responses_file,
                grades_file=grades_file,
                exams_dir=ExamMCPServer.exams_dir
            )

            exam_id = exam_data["exam_id"]
            ExamMCPServer.context.loaded_exams[exam_id] = exam_data
            question_ids = [q["id"] for q in exam_data["questions"]]

            summary_output = {
                "status": "success",
                "exam_id": exam_id,
                "counts": {
                    "questions": len(exam_data["questions"]),
                    "students": len(exam_data["students"])
                },
                "question_ids": question_ids,
                "message": "Exam loaded. Use 'list_loaded_students_tool' to get emails."
            }

            return json.dumps(summary_output, indent=2)

        except FileNotFoundError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @staticmethod
    @tool("assess_students_batch_tool")
    async def assess_students_batch(student_emails: list[str]) -> str:
        """
        Assess a BATCH of students in one go.
        Input: A list of student email strings (e.g. ["email1", "email2"]).
        """
        results_summary = []
        failed = []

        print(f"\n[BATCH] Starting assessment for {len(student_emails)} students...")

        # Initialize assessor once
        assessor = Assessor(evaluations_dir=ExamMCPServer.evaluations_dir)
        questions = None  # Will be fetched from first valid student match

        for student_email in student_emails:
            try:
                student_email_clean = student_email.rstrip('.').strip()
                student_data = None
                matched_email = None

                # Logic to find student data in memory
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
                    failed.append(f"{student_email} (Not Found)")
                    continue

                # Perform assessment
                result = await assessor.assess_student_exam(
                    student_email=matched_email,
                    exam_questions=questions,
                    student_responses=student_data["responses"],
                    questions_store=ExamMCPServer.questions_store,
                    context=ExamMCPServer.context,
                    original_grades=student_data.get("original_grades", {})
                )

                score = result.get("calculated_score", 0.0)
                max_score = result.get("max_score", 0.0)

                print(f"[BATCH] Processed {matched_email[:15]}... Score: {score}")
                results_summary.append(f"{matched_email}: {score}/{max_score}")

            except Exception as e:
                failed.append(f"{student_email} (Error: {str(e)})")

        # Return a single summary for the whole batch
        output = {
            "status": "batch_completed",
            "processed": len(results_summary),
            "failed_count": len(failed),
            "grades_summary": results_summary,
            "failures": failed
        }
        return json.dumps(output, indent=2)