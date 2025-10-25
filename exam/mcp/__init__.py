"""
MCP Server con Context Condiviso per collaborazione tra tool.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from exam import get_questions_store
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
    """MCP Server con context condiviso per collaborazione tra tool."""

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

        async def load_exam_from_yaml(questions_file: str, responses_file: str, grades_file: str = None) -> str:
            """
            Load an entire exam from YAML files in static/se-exams directory.

            Args:
                questions_file: Filename of questions YAML (e.g., "se-2025-06-05-questions.yml")
                responses_file: Filename of responses YAML (e.g., "se-2025-06-05-responses.yml")
                grades_file: Optional filename of grades YAML (e.g., "se-2025-06-05-grades.yml")

            Files are loaded from static/se-exams/ directory automatically.

            Returns:
                JSON with exam structure
            """
            try:
                import yaml

                # Check if full path or just filename
                questions_path = Path(questions_file)
                if not questions_path.is_absolute():
                    questions_path = self.exams_dir / questions_file

                responses_path = Path(responses_file)
                if not responses_path.is_absolute():
                    responses_path = self.exams_dir / responses_file

                if not questions_path.exists():
                    return json.dumps({
                        "error": f"Questions file not found: {questions_path}",
                        "searched_in": str(self.exams_dir),
                        "hint": "Use list_available_exams to see available files"
                    })

                if not responses_path.exists():
                    return json.dumps({
                        "error": f"Responses file not found: {responses_path}",
                        "searched_in": str(self.exams_dir),
                        "hint": "Use list_available_exams to see available files"
                    })

                # Load YAML files
                with open(questions_path, 'r', encoding='utf-8') as f:
                    questions_data = yaml.safe_load(f)

                with open(responses_path, 'r', encoding='utf-8') as f:
                    responses_data = yaml.safe_load(f)

                # Load grades file if provided
                grades_data = None
                if grades_file:
                    grades_path = Path(grades_file)
                    if not grades_path.is_absolute():
                        grades_path = self.exams_dir / grades_file

                    if grades_path.exists():
                        with open(grades_path, 'r', encoding='utf-8') as f:
                            grades_data = yaml.safe_load(f)
                        print(f"[LOAD_EXAM] Loaded grades from {grades_path.name}")
                    else:
                        print(f"[LOAD_EXAM] Warning: grades file not found: {grades_path}")

                # Create a mapping of email -> grades
                grades_by_email = {}
                if grades_data:
                    import re
                    # Pattern per trovare chiavi come q1300, q2400, q3500, etc.
                    question_grade_pattern = re.compile(r'^q(\d+)\d{3}$')

                    for grade_entry in grades_data:
                        email = grade_entry.get("emailaddress")
                        if email and grade_entry.get("state") == "Finished":
                            # Extract individual question grades
                            question_grades = {}

                            # Cerca tutte le chiavi che matchano il pattern q{numero}{3cifre}
                            for key, value in grade_entry.items():
                                match = question_grade_pattern.match(key)
                                if match:
                                    question_num = int(match.group(1))
                                    try:
                                        question_grades[question_num] = float(value)
                                    except (ValueError, TypeError):
                                        pass  # Skip invalid grades

                            grades_by_email[email] = {
                                "total_grade": float(grade_entry.get("grade2700", 0)),
                                "question_grades": question_grades
                            }

                # Parse questions
                questions = []
                for key, value in questions_data.items():
                    if key.startswith("Question"):
                        questions.append({
                            "number": key,
                            "id": value.get("id"),
                            "text": value.get("text"),
                            "score": value.get("score", 3.0)
                        })

                # Parse students
                students = []
                for student_data in responses_data:
                    if student_data.get("state") != "Finished":
                        continue

                    email = student_data.get("emailaddress", "unknown")

                    # Extract responses based on number of questions
                    responses = {}
                    for i in range(1, len(questions) + 1):
                        response_key = f"response{i}"
                        if response_key in student_data:
                            response_text = student_data[response_key]
                            if response_text and response_text.strip() != '-':
                                responses[i] = response_text

                    students.append({
                        "email": email,
                        "started": student_data.get("startedon"),
                        "completed": student_data.get("completed"),
                        "time_taken": student_data.get("timetaken"),
                        "moodle_grade": student_data.get("grade2700"),
                        "responses": responses,
                        "num_responses": len(responses),
                        "original_grades": grades_by_email.get(email, {})  # Add original grades
                    })

                # Store in context
                exam_id = f"{questions_path.stem}_{responses_path.stem}"
                self.context.loaded_exams[exam_id] = {
                    "questions": questions,
                    "students": students,
                    "questions_file": str(questions_path),
                    "responses_file": str(responses_path)
                }

                return json.dumps({
                    "exam_id": exam_id,
                    "loaded_from": str(self.exams_dir),
                    "questions_file": questions_path.name,
                    "responses_file": responses_path.name,
                    "num_questions": len(questions),
                    "num_students": len(students),
                    "questions": questions,
                    "students_preview": [
                        {
                            "email": s["email"] ,
                            "num_responses": s["num_responses"],
                            "time_taken": s["time_taken"]
                        }
                        for s in students[:5]
                    ],
                    "message": f"Loaded exam with {len(questions)} questions and {len(students)} students from {self.exams_dir}"
                }, indent=2)

            except Exception as e:
                import traceback
                return json.dumps({"error": str(e), "traceback": traceback.format_exc()})

        tools["load_exam_from_yaml"] = load_exam_from_yaml

        # TOOL: Assess Student Exam (REFACTORIZZATO)
        async def assess_student_exam(student_email: str) -> str:
            """
            Assess all responses for a single student from loaded exam.
            Results are automatically saved to evaluations/{email_hash}/assessment.json

            REFACTORIZZATO: Ora usa la classe Assessor per la logica di valutazione.

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
                student_dir = self.evaluations_dir / student_email_full
                student_dir.mkdir(parents=True, exist_ok=True)

                print(f"[ASSESS] Matched student: {student_email_full}")

                # =========================================================
                # REFACTORING: Usa la classe Assessor invece di codice inline
                # =========================================================

                assessor = Assessor()

                result = await assessor.assess_student_exam(
                    student_email=student_email_full,
                    exam_questions=questions,
                    student_responses=student_data["responses"],
                    questions_store=self.questions_store,
                    context=self.context
                )

                # Aggiungi metadati Moodle
                result["moodle_grade"] = student_data.get("moodle_grade")

                # =========================================================
                # Fine refactoring - il resto è solo salvataggio file
                # =========================================================

                # Salva assessment completo in JSON
                assessment_file = student_dir / "assessment.json"
                with open(assessment_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                # Salva anche un summary leggibile
                summary_file = student_dir / "summary.txt"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"STUDENT ASSESSMENT SUMMARY\n")
                    f.write(f"{'='*70}\n\n")
                    f.write(f"Student: {student_email_full}\n")
                    f.write(f"Calculated Score: {result['calculated_score']:.2f}/{result['max_score']}\n")
                    f.write(f"Calculated Percentage: {result['percentage']}%\n")

                    # Add comparison with original grades if available
                    original_grades = student_data.get("original_grades", {})
                    if original_grades:
                        original_total = original_grades.get("total_grade", 0)
                        f.write(f"Original Moodle Grade: {original_total:.2f}/27.00\n")

                        # Calculate difference
                        score_diff = result['calculated_score'] - original_total
                        f.write(f"Difference: {score_diff:+.2f} ")
                        if abs(score_diff) < 0.5:
                            f.write("( Very close)\n")
                        elif abs(score_diff) < 2.0:
                            f.write("( Reasonable)\n")
                        else:
                            f.write("(Significant difference)\n")
                    else:
                        f.write(f"Moodle Grade: {student_data.get('moodle_grade', 'N/A')}\n")

                    f.write(f"Scoring System: {result['scoring_system']}\n\n")
                    f.write(f"{'='*70}\n\n")

                    for assessment in result["assessments"]:
                        question_num = assessment['question_number']
                        f.write(f"Question {question_num}: {assessment['question_id']}\n")
                        f.write(f"{'-'*70}\n")

                        if assessment['status'] == 'assessed':
                            f.write(f"Calculated Score: {assessment['score']:.2f}/{assessment['max_score']}\n")

                            # Add comparison with original grade if available
                            if original_grades and 'question_grades' in original_grades:
                                orig_q_grade = original_grades['question_grades'].get(question_num)
                                if orig_q_grade is not None:
                                    diff = assessment['score'] - orig_q_grade
                                    f.write(f"Original Grade: {orig_q_grade:.2f}/{assessment['max_score']}\n")
                                    f.write(f"Difference: {diff:+.2f}\n")

                            f.write(f"Breakdown: {assessment['breakdown']}\n\n")

                            # Raggruppa per tipo
                            core_features = [fa for fa in assessment['feature_assessments']
                                        if fa['feature_type'] == 'CORE']
                            important_features = [fa for fa in assessment['feature_assessments']
                                                if fa['feature_type'] == 'DETAILS_IMPORTANT']

                            if core_features:
                                f.write("CORE Elements:\n")
                                for fa in core_features:
                                    status = "✓ OK" if fa['satisfied'] else "✗ MISSING"
                                    f.write(f"  [{status}] {fa['feature']}\n")
                                    f.write(f"       {fa['motivation']}\n\n")

                            if important_features:
                                f.write("Important Details:\n")
                                for fa in important_features:
                                    status = "✓ OK" if fa['satisfied'] else "✗ MISSING"
                                    f.write(f"  [{status}] {fa['feature']}\n")
                                    f.write(f"       {fa['motivation']}\n\n")

                        else:
                            f.write(f"Status: {assessment['status']}\n")
                            if 'error' in assessment:
                                f.write(f"Error: {assessment['error']}\n")

                        f.write(f"\n{'='*70}\n\n")

                # Aggiungi info sui file salvati al risultato
                result["saved_files"] = {
                    "assessment": str(assessment_file),
                    "summary": str(summary_file)
                }

                return json.dumps(result, indent=2)

            except Exception as e:
                import traceback
                return json.dumps({"error": str(e), "traceback": traceback.format_exc()})

        tools["assess_student_exam"] = assess_student_exam
        return tools