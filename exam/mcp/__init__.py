"""
MCP Server con Context Condiviso per collaborazione tra tool.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, field

from exam import get_questions_store, Question
from exam.solution import Answer, load_cache as load_answer_cache
from exam.assess import FeatureAssessment, Feature, FeatureType, enumerate_features


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
    
    def __init__(self, exam_dir: Path = None):
        self.exam_dir = exam_dir
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
        
        # TOOL: Load Student Answer (ATOMICO)
        async def load_student_answer(question_id: str, student_code: str) -> str:
            """
            Load a student's answer into memory.
            The answer will be available for other tools to use.
            
            Args:
                question_id: The question ID (e.g., "CI-5")
                student_code: The student code (e.g., "280944")
            
            Returns:
                JSON with answer preview and confirmation it's loaded
            """
            if not self.exam_dir:
                return json.dumps({"error": "No exam directory configured"})
            
            # Check if already loaded
            cached = self.context.get_answer(question_id, student_code)
            if cached:
                return json.dumps({
                    "status": "already_loaded",
                    "question_id": question_id,
                    "student_code": student_code,
                    "preview": cached[:200] + "..." if len(cached) > 200 else cached,
                    "length": len(cached)
                })
            
            # Load from file
            pattern = f"Q* - {question_id}/{student_code} - *"
            matching_dirs = list(self.exam_dir.glob(pattern))
            
            if not matching_dirs:
                return json.dumps({"error": f"No answer found for student {student_code} on question {question_id}"})
            
            student_dir = matching_dirs[0]
            answer_file = next(student_dir.glob("Attempt*_textresponse"), None)
            
            if not answer_file:
                return json.dumps({"error": "Answer file not found"})
            
            answer_text = answer_file.read_text(encoding="utf-8")
            student_name = student_dir.name.split(" - ")[1] if " - " in student_dir.name else "Unknown"
            
            # Store in context
            self.context.store_answer(question_id, student_code, answer_text)
            
            return json.dumps({
                "status": "loaded",
                "question_id": question_id,
                "student_code": student_code,
                "student_name": student_name,
                "preview": answer_text[:200] + "..." if len(answer_text) > 200 else answer_text,
                "length": len(answer_text),
                "word_count": len(answer_text.split()),
                "message": "Answer loaded into memory. Use assess_all_features to evaluate it."
            })
        
        tools["load_student_answer"] = load_student_answer
        
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
        
        # TOOL : Assess All Features
        async def assess_all_features(question_id: str, student_code: str) -> str:
            """
            Assess ALL features for a student's answer in one step.
            Automatically loads answer and checklist if needed.
            
            This is the FASTEST way to get a complete assessment.
            
            Use this when you need:
            - Quick complete assessment
            - Full score breakdown
            - All feature evaluations at once
            
            Returns: Complete assessment with score and all feature feedback.
            """
            from exam.llm_provider import llm_client
            from exam.assess import TEMPLATE, calculate_score_from_assessments
            
            # Ensure answer is loaded
            if not self.context.get_answer(question_id, student_code):
                load_result = await load_student_answer(question_id, student_code)
                result_data = json.loads(load_result)
                if "error" in result_data:
                    return load_result
            
            # Ensure checklist is loaded
            if not self.context.get_checklist(question_id):
                checklist_result = await load_checklist(question_id)
                result_data = json.loads(checklist_result)
                if "error" in result_data:
                    return checklist_result
            
            # Get data from context
            answer_text = self.context.get_answer(question_id, student_code)
            checklist = self.context.get_checklist(question_id)
            question = self.questions_store.question(question_id)
            
            # Assess all features
            assessments_list = []
            assessments_dict = {}
            
            for index, feature in enumerate_features(checklist):
                prompt = TEMPLATE.format(
                    class_name="FeatureAssessment",
                    question=question.text,
                    feature_type=feature.type.value,
                    feature_verb_ideal=feature.verb_ideal,
                    feature_verb_actual=feature.verb_actual,
                    feature=feature.description,
                    answer=answer_text
                )
                
                llm, _, _ = llm_client(structured_output=FeatureAssessment)
                result = llm.invoke(prompt)
                
                assessments_list.append({
                    "feature": feature.description,
                    "feature_type": feature.type.name,
                    "satisfied": result.satisfied,
                    "motivation": result.motivation
                })
                
                assessments_dict[feature] = result
            
            # Store assessments in context
            self.context.store_assessments(question_id, student_code, assessments_list)
            
            # Calculate score using centralized function
            score, breakdown, stats = calculate_score_from_assessments(assessments_dict, question.weight)
            
            return json.dumps({
                "question_id": question_id,
                "student_code": student_code,
                "assessments": assessments_list,
                "statistics": stats,
                "estimated_score": {
                    "score": score,
                    "max_score": question.weight,
                    "percentage": round((score / question.weight * 100) if question.weight > 0 else 0, 1),
                    "breakdown": breakdown
                }
            }, indent=2)
        
        tools["assess_all_features"] = assess_all_features
        
        # TOOL: Load Exam from YAML
        async def load_exam_from_yaml(questions_file: str, responses_file: str) -> str:
            """
            Load an entire exam from YAML files in static/se-exams directory.
            
            Args:
                questions_file: Filename of questions YAML (e.g., "se-2025-06-05-questions.yml")
                responses_file: Filename of responses YAML (e.g., "se-2025-06-05-responses.yml")
            
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
                    
                    # Extract responses
                    responses = {}
                    for i in range(1, 10):
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
                        "num_responses": len(responses)
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
        
        # TOOL: Assess Student Exam
        async def assess_student_exam(student_email: str) -> str:
            """
            Assess all responses for a single student from loaded exam.
            Results are automatically saved to evaluations/{email_hash}/assessment.json
            
            Args:
                student_email: Student's email (can use first 20 chars)
            
            Returns:
                Complete assessment for all student's responses
            """
            try:
                from exam.llm_provider import llm_client
                from exam.assess import TEMPLATE, enumerate_features, FeatureAssessment, calculate_score_from_assessments, Feature
                from exam.solution import load_cache as load_answer_cache

                student_email_clean = student_email.rstrip('.').strip()

                # Find student with FLEXIBLE matching
                student_data = None
                questions = None
                matched_email = None

                for exam_data in self.context.loaded_exams.values():
                    for student in exam_data["students"]:
                        full_email = student["email"]

                        # Match flessibile:
                        # 1. Exact match (case insensitive)
                        # 2. Partial match (inizio email, min 10 chars)
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
                
                # Assess each response
                assessments = []
                total_score = 0.0
                total_max_score = 0.0
                
                for question in questions:
                    question_num = int(question["number"].replace("Question ", ""))
                    
                    if question_num not in student_data["responses"]:
                        assessments.append({
                            "question_number": question_num,
                            "question_id": question["id"],
                            "question_text": question.get("text", ""),
                            "status": "no_response",
                            "score": 0.0,
                            "max_score": question["score"]
                        })
                        total_max_score += question["score"]
                        continue
                    
                    try:
                        q = self.questions_store.question(question["id"])
                        response_text = student_data["responses"][question_num]
                        
                        # Load checklist if not in context
                        if not self.context.get_checklist(question["id"]):
                            checklist = load_answer_cache(q)
                            if checklist:
                                self.context.store_checklist(question["id"], checklist)
                        
                        checklist = self.context.get_checklist(question["id"])
                        if not checklist:
                            assessments.append({
                                "question_number": question_num,
                                "question_id": question["id"],
                                "question_text": question.get("text", ""),
                                "status": "no_checklist",
                                "score": 0.0,
                                "max_score": question["score"]
                            })
                            total_max_score += question["score"]
                            continue
                        
                        # Assess features
                        feature_assessments_list = []
                        feature_assessments_dict = {}
                        
                        for index, feature in enumerate_features(checklist):
                            prompt = TEMPLATE.format(
                                class_name="FeatureAssessment",
                                question=q.text,
                                feature_type=feature.type.value,
                                feature_verb_ideal=feature.verb_ideal,
                                feature_verb_actual=feature.verb_actual,
                                feature=feature.description,
                                answer=response_text
                            )
                            
                            llm, _, _ = llm_client(structured_output=FeatureAssessment)
                            result = llm.invoke(prompt)
                            
                            feature_assessments_list.append({
                                "feature": feature.description,
                                "feature_type": feature.type.name,
                                "satisfied": result.satisfied,
                                "motivation": result.motivation
                            })
                            
                            feature_assessments_dict[feature] = result
                        
                        score, breakdown, stats = calculate_score_from_assessments(
                            feature_assessments_dict, 
                            question["score"],
                        )
                        
                        assessments.append({
                            "question_number": question_num,
                            "question_id": question["id"],
                            "question_text": q.text,
                            "student_response": response_text,
                            "status": "assessed",
                            "score": score,
                            "max_score": question["score"],
                            "statistics": stats,
                            "breakdown": breakdown,
                            "feature_assessments": feature_assessments_list
                        })
                        
                        total_score += score
                        total_max_score += question["score"]
                        
                    except Exception as e:
                        assessments.append({
                            "question_number": question_num,
                            "question_id": question["id"],
                            "question_text": question.get("text", ""),
                            "status": "error",
                            "error": str(e),
                            "score": 0.0,
                            "max_score": question["score"]
                        })
                        total_max_score += question["score"]
                
                # Prepara risultato finale
                result = {
                    "student_email": student_email_full,
                    "calculated_score": total_score,
                    "max_score": total_max_score,
                    "percentage": round((total_score / total_max_score * 100) if total_max_score > 0 else 0, 1),
                    "moodle_grade": student_data.get("moodle_grade"),
                    "scoring_system": "70% Core + 30% Important_Details",
                    "assessments": assessments
                }
                
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
                    f.write(f"Score: {total_score}/{total_max_score} ({result['percentage']}%)\n")
                    f.write(f"Moodle Grade: {student_data.get('moodle_grade', 'N/A')}\n")
                    f.write(f"Scoring: 70% Core + 35% Important\n\n")
                    f.write(f"{'='*70}\n\n")
                    
                    for assessment in assessments:
                        f.write(f"Question {assessment['question_number']}: {assessment['question_id']}\n")
                        f.write(f"{'-'*70}\n")
                        
                        if assessment['status'] == 'assessed':
                            f.write(f"Score: {assessment['score']}/{assessment['max_score']}\n")
                            f.write(f"Breakdown: {assessment['breakdown']}\n\n")
                            
                            # Raggruppa per tipo
                            core_features = [fa for fa in assessment['feature_assessments'] 
                                        if fa['feature_type'] == 'CORE']
                            important_features = [fa for fa in assessment['feature_assessments'] 
                                                if fa['feature_type'] == 'DETAILS_IMPORTANT']

                            
                            if core_features:
                                f.write("CORE Elements:\n")
                                for fa in core_features:
                                    status = "OK" if fa['satisfied'] else "MISSING"
                                    f.write(f"  [{status}] {fa['feature']}\n")
                                    f.write(f"       {fa['motivation']}\n\n")
                            
                            if important_features:
                                f.write("Important Details:\n")
                                for fa in important_features:
                                    status = "OK" if fa['satisfied'] else "MISSING"
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
        
        # TOOL: Assess Full Exam Batch
        async def assess_full_exam(questions_file: str, responses_file: str, max_students: int = None) -> str:
            """
            Assess entire exam for all students.
            
            Args:
                questions_file: Path to questions YAML
                responses_file: Path to responses YAML
                max_students: Limit number of students (optional, for testing)
            
            Returns:
                Summary of all assessments with statistics
            """
            try:
                # Load exam
                load_result = await load_exam_from_yaml(questions_file, responses_file)
                load_data = json.loads(load_result)
                
                if "error" in load_data:
                    return load_result
                
                # Get students
                exam_id = load_data["exam_id"]
                exam_data = self.context.loaded_exams[exam_id]
                students = exam_data["students"]
                
                if max_students:
                    students = students[:max_students]
                
                # Assess each student
                results = []
                for idx, student in enumerate(students):
                    print(f"[BATCH] Assessing student {idx+1}/{len(students)}...")
                    
                    assessment_result = await assess_student_exam(student["email"])
                    assessment_data = json.loads(assessment_result)
                    
                    if "error" not in assessment_data:
                        results.append({
                            "student_index": idx + 1,
                            "email_preview": student["email"][:30] + "...",
                            "calculated_score": assessment_data["calculated_score"],
                            "max_score": assessment_data["max_score"],
                            "percentage": assessment_data["percentage"],
                            "moodle_grade": assessment_data["moodle_grade"]
                        })
                
                # Statistics
                scores = [r["calculated_score"] for r in results]
                avg_score = sum(scores) / len(scores) if scores else 0
                
                return json.dumps({
                    "exam": {
                        "questions_file": questions_file,
                        "responses_file": responses_file,
                        "num_students": len(results)
                    },
                    "statistics": {
                        "average_score": round(avg_score, 2),
                        "max_score": max(scores) if scores else 0,
                        "min_score": min(scores) if scores else 0,
                        "total_possible": results[0]["max_score"] if results else 0
                    },
                    "results": results
                }, indent=2)
                
            except Exception as e:
                import traceback
                return json.dumps({"error": str(e), "traceback": traceback.format_exc()})
        
        tools["assess_full_exam"] = assess_full_exam
        
        return tools