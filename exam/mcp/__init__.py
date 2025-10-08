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
        self.tools = self._create_tools()
    
    def _create_tools(self):
        """Create all available tools."""
        tools = {}
        
        # TOOL 1: List Questions
        async def list_questions() -> str:
            """List all available questions in the question bank."""
            questions = []
            for q in self.questions_store.questions:
                questions.append({
                    "id": q.id,
                    "category": q.category.name,
                    "text": q.text,
                    "weight": q.weight
                })
            return json.dumps(questions, indent=2)
        
        tools["list_questions"] = list_questions
        
        # TOOL 2: List Students
        async def list_students(question_id: str = None) -> str:
            """List all students who submitted answers."""
            if not self.exam_dir:
                return json.dumps({"error": "No exam directory configured"})
            
            students = {}
            pattern = f"Q* - {question_id}/*" if question_id else "Q*/*"
            
            for student_dir in self.exam_dir.glob(pattern):
                if student_dir.is_dir() and " - " in student_dir.name:
                    parts = student_dir.name.split(" - ")
                    if len(parts) >= 2:
                        code = parts[0]
                        name = parts[1]
                        
                        question_folder = student_dir.parent.name
                        q_id = question_folder.split(" - ")[1] if " - " in question_folder else "unknown"
                        
                        if code not in students:
                            students[code] = {
                                "code": code,
                                "name": name,
                                "questions": []
                            }
                        students[code]["questions"].append(q_id)
            
            return json.dumps(list(students.values()), indent=2)
        
        tools["list_students"] = list_students
        
        # TOOL 3: Load Student Answer (ATOMICO)
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
        
        # TOOL 4: Load Checklist (ATOMICO)
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
                    "additional_count": len(cached.details_additional)
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
                        "additional": len(checklist.details_additional)
                    },
                    "core_items": checklist.core,
                    "important_items": checklist.details_important,
                    "additional_items": checklist.details_additional,
                    "message": "Checklist loaded into memory. Use assess_all_features to evaluate."
                })
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        tools["load_checklist"] = load_checklist
        
        # TOOL 5: Assess All Features (COMPOSTO)
        async def assess_all_features(question_id: str, student_code: str) -> str:
            """
            Assess ALL features for a student's answer.
            Automatically loads answer and checklist if needed, then evaluates all features.
            
            Args:
                question_id: The question ID (e.g., "CI-5")
                student_code: The student code (e.g., "280944")
            
            Returns:
                Complete assessment with score breakdown
            """
            from exam.openai import llm_client
            from exam.assess import TEMPLATE
            
            # Ensure answer is loaded
            # if not self.context.get_answer(question_id, student_code):
            #     load_result = await load_student_answer(question_id, student_code)
            #     result_data = json.loads(load_result)
            #     if "error" in result_data:
            #         return load_result
            
            # # Ensure checklist is loaded
            # if not self.context.get_checklist(question_id):
            #     checklist_result = await load_checklist(question_id)
            #     result_data = json.loads(checklist_result)
            #     if "error" in result_data:
            #         return checklist_result
            
            # Get data from context
            answer_text = self.context.get_answer(question_id, student_code)
            checklist = self.context.get_checklist(question_id)
            question = self.questions_store.question(question_id)
            
            # Assess all features
            assessments = []
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
                
                assessments.append({
                    "feature": feature.description,
                    "feature_type": feature.type.name,
                    "satisfied": result.satisfied,
                    "motivation": result.motivation
                })
            
            # Store assessments in context
            self.context.store_assessments(question_id, student_code, assessments)
            
            # Calculate statistics
            core_count = sum(1 for a in assessments if a["feature_type"] == "CORE")
            core_satisfied = sum(1 for a in assessments if a["feature_type"] == "CORE" and a["satisfied"])
            
            important_count = sum(1 for a in assessments if a["feature_type"] == "DETAILS_IMPORTANT")
            important_satisfied = sum(1 for a in assessments if a["feature_type"] == "DETAILS_IMPORTANT" and a["satisfied"])
            
            additional_count = sum(1 for a in assessments if a["feature_type"] == "DETAILS_ADDITIONAL")
            additional_satisfied = sum(1 for a in assessments if a["feature_type"] == "DETAILS_ADDITIONAL" and a["satisfied"])
            
            # Calculate score
            core_percentage = (core_satisfied / core_count * 0.70) if core_count > 0 else 0.0
            important_percentage = (important_satisfied / important_count * 0.20) if important_count > 0 else 0.20
            additional_percentage = (additional_satisfied / additional_count * 0.10) if additional_count > 0 else 0.10
            
            final_percentage = core_percentage + important_percentage + additional_percentage
            estimated_score = round(final_percentage * question.weight, 3)
            
            return json.dumps({
                "question_id": question_id,
                "student_code": student_code,
                "assessments": assessments,
                "statistics": {
                    "core": {
                        "total": core_count,
                        "satisfied": core_satisfied,
                        "percentage": round((core_satisfied / core_count * 100) if core_count > 0 else 0, 1)
                    },
                    "details_important": {
                        "total": important_count,
                        "satisfied": important_satisfied,
                        "percentage": round((important_satisfied / important_count * 100) if important_count > 0 else 0, 1)
                    },
                    "details_additional": {
                        "total": additional_count,
                        "satisfied": additional_satisfied,
                        "percentage": round((additional_satisfied / additional_count * 100) if additional_count > 0 else 0, 1)
                    }
                },
                "estimated_score": {
                    "score": estimated_score,
                    "max_score": question.weight,
                    "percentage": round(final_percentage * 100, 1),
                    "breakdown": f"Core: {core_satisfied}/{core_count} (70%) + Important: {important_satisfied}/{important_count} (20%) + Additional: {additional_satisfied}/{additional_count} (10%)"
                }
            }, indent=2)
        
        tools["assess_all_features"] = assess_all_features
        
        # TOOL 6: Calculate Score
        async def calculate_score(question_id: str, student_code: str) -> str:
            """
            Calculate final score from stored assessments.
            Requires that assess_all_features has been run first.
            
            Args:
                question_id: The question ID
                student_code: The student code
            
            Returns:
                Score breakdown
            """
            assessments = self.context.get_assessments(question_id, student_code)
            if not assessments:
                return json.dumps({
                    "error": "No assessments found. Run assess_all_features first."
                })
            
            question = self.questions_store.question(question_id)
            
            # Calculate statistics
            core_count = sum(1 for a in assessments if a["feature_type"] == "CORE")
            core_satisfied = sum(1 for a in assessments if a["feature_type"] == "CORE" and a["satisfied"])
            
            important_count = sum(1 for a in assessments if a["feature_type"] == "DETAILS_IMPORTANT")
            important_satisfied = sum(1 for a in assessments if a["feature_type"] == "DETAILS_IMPORTANT" and a["satisfied"])
            
            additional_count = sum(1 for a in assessments if a["feature_type"] == "DETAILS_ADDITIONAL")
            additional_satisfied = sum(1 for a in assessments if a["feature_type"] == "DETAILS_ADDITIONAL" and a["satisfied"])
            
            # Calculate score
            core_percentage = (core_satisfied / core_count * 0.70) if core_count > 0 else 0.0
            important_percentage = (important_satisfied / important_count * 0.20) if important_count > 0 else 0.20
            additional_percentage = (additional_satisfied / additional_count * 0.10) if additional_count > 0 else 0.10
            
            final_percentage = core_percentage + important_percentage + additional_percentage
            estimated_score = round(final_percentage * question.weight, 2)
            
            return json.dumps({
                "question_id": question_id,
                "student_code": student_code,
                "score": estimated_score,
                "max_score": question.weight,
                "percentage": round(final_percentage * 100, 1),
                "breakdown": {
                    "core": f"{core_satisfied}/{core_count} (70% weight)",
                    "important": f"{important_satisfied}/{important_count} (20% weight)",
                    "additional": f"{additional_satisfied}/{additional_count} (10% weight)"
                }
            }, indent=2)
        
        tools["calculate_score"] = calculate_score
        
        return tools
    
    def get_tool_descriptions(self):
        """Get descriptions of all available tools."""
        descriptions = {}
        for name, func in self.tools.items():
            descriptions[name] = {
                "name": name,
                "description": func.__doc__.strip() if func.__doc__ else "No description",
            }
        return descriptions


async def run_mcp_server(exam_dir: Path = None):
    """
    Run the MCP server.
    
    Args:
        exam_dir: Directory containing student submissions
    """
    server = ExamMCPServer(exam_dir)
    
    print("# MCP Server initialized with tools:")
    for tool_name in server.tools.keys():
        print(f"#   - {tool_name}")
    print("# MCP Server ready")
    
    # Keep server running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n# MCP Server shutting down")