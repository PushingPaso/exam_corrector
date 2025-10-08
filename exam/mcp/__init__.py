"""
MCP (Model Context Protocol) Server for Exam Correction System.
Exposes tools that LLMs can use to interact with the exam assessment system.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from exam import get_questions_store, Question
from exam.solution import Answer, load_cache as load_answer_cache
from exam.assess import Assessor, FeatureAssessment, Feature, FeatureType
from exam.rag import sqlite_vector_store


class ExamMCPServer:
    """MCP-style server that exposes exam assessment tools to LLMs."""
    
    def __init__(self, exam_dir: Path = None):
        """
        Initialize the MCP server.
        
        Args:
            exam_dir: Directory containing student submissions
        """
        self.exam_dir = exam_dir
        self.questions_store = get_questions_store() 
        self.vector_store = None
        
        # Try to initialize vector store only if it has content
        try:
            from exam import DIR_ROOT
            db_file = DIR_ROOT / "slides-rag.db"
            
            # Check if database exists and has content
            if db_file.exists() and db_file.stat().st_size > 0:
                vs = sqlite_vector_store()
                
                # Try to check if it has data by getting dimensionality
                try:
                    dim = vs.get_dimensionality()
                    if dim > 0:
                        self.vector_store = vs
                        print(f"# MCP Server: RAG vector store loaded ({dim} dimensions)")
                    else:
                        print("# MCP Server: RAG database exists but is empty (skipped)")
                except:
                    # If we can't get dimensionality, assume it's usable
                    self.vector_store = vs
                    print("# MCP Server: RAG vector store loaded")
            else:
                print("# MCP Server: RAG database not found or empty (skipped)")
        except Exception as e:
            print(f"# MCP Server: RAG not available: {e}")
        
        # Register all tools
        self.tools = self._create_tools()
    
    def _create_tools(self):
        """Create all available tools as a dictionary."""
        
        tools = {}
        
        # Wrapper to add error logging to all tools
        def tool_wrapper(tool_name, tool_func):
            async def wrapped(*args, **kwargs):
                try:
                    print(f"\n[TOOL] {tool_name} called with args={args}, kwargs={kwargs}")
                    result = await tool_func(*args, **kwargs)
                    print(f"[TOOL] {tool_name} returned: {result[:200] if len(result) > 200 else result}...")
                    return result
                except Exception as e:
                    error_msg = f"[TOOL ERROR] {tool_name} failed: {type(e).__name__}: {str(e)}"
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                    return json.dumps({"error": f"{type(e).__name__}: {str(e)}", "tool": tool_name})
            return wrapped
        
        # Tool 1: list_questions
        async def list_questions() -> str:
            """
            List all available questions in the question bank.
            Returns a JSON string with question IDs, categories, and text.
            """
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
        
        # Tool 2: get_question
        async def get_question(question_id: str) -> str:
            """
            Get details of a specific question.
            
            Args:
                question_id: The ID of the question (e.g., "Foundations-1")
            
            Returns:
                JSON string with question details
            """
  
            try:
                question = self.questions_store.question(question_id)
                return json.dumps({
                    "id": question.id,
                    "category": question.category.name,
                    "text": question.text,
                    "weight": question.weight,
                    "max_lines": question.max_lines
                }, indent=2)
            except KeyError:
                return json.dumps({"error": f"Question '{question_id}' not found"})
        
        tools["get_question"] = get_question
        
        # Tool 3: list_students
        async def list_students(question_id: str = None) -> str:
            """
            List all students who submitted answers.
            
            Args:
                question_id: Optional - filter by specific question ID
            
            Returns:
                JSON string with student codes, names, and questions answered
            """
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
                        
                        # Extract question ID from parent
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
        
        # Tool 4: read_student_answer
        async def read_student_answer(question_id: str, student_code: str) -> str:
            """
            Read a student's answer to a specific question.
            
            Args:
                question_id: The ID of the question (e.g., "Foundations-1")
                student_code: The student's code (e.g., "123456")
            
            Returns:
                JSON string with answer text and metadata
            """
            if not self.exam_dir:
                return json.dumps({"error": "No exam directory configured"})
            
            # Find the student's answer file
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
            
            return json.dumps({
                "question_id": question_id,
                "student_code": student_code,
                "student_name": student_name,
                "answer": answer_text,
                "length": len(answer_text),
                "word_count": len(answer_text.split())
            }, indent=2)
        
        tools["read_student_answer"] = read_student_answer
        
        # Tool 5: get_checklist
        async def get_checklist(question_id: str) -> str:
            """
            Get the assessment checklist for a specific question.
            
            Args:
                question_id: The ID of the question (e.g., "Foundations-1")
            
            Returns:
                JSON string with CORE and DETAILS
            """
            try:
                question = self.questions_store.question(question_id)
                answer = load_answer_cache(question)
                
                if not answer:
                    return json.dumps({"error": f"No checklist found for question {question_id}"})
                
                return json.dumps({
                    "question_id": question_id,
                    "question_text": question.text,
                    "core": answer.core,
                    "details_important": answer.details_important,
                    "details_additional": answer.details_additional,
                }, indent=2)
            except KeyError:
                return json.dumps({"error": f"Question {question_id} not found"})
        
        tools["get_checklist"] = get_checklist
        
        # Tool 6: assess_feature (con auto-save)
        async def assess_and_save_feature(
            question_id: str,
            student_answer: str,
            student_code: str = None
        ) -> str:
            """
            Assess ALL features of a student answer against the checklist.
            Automatically saves results if student_code is provided.
            
            Args:
                question_id: The ID of the question
                student_answer: The student's answer text
                student_code: Optional student code (if provided, saves results)
            
            Returns:
                JSON string with list of all feature assessments
            """
            try:
                from exam.openai import llm_client
                from exam.assess import TEMPLATE, enumerate_features
                from exam.solution import load_cache as load_answer_cache
                
                question = self.questions_store.question(question_id)
                answer_cache = load_answer_cache(question)
                
                if not answer_cache:
                    return json.dumps({"error": f"No checklist found for {question_id}"})
                
                assessments = []
                
                print(f"[TOOL] assess_feature: Evaluating {question_id}...")
                
                for index, feature in enumerate_features(answer_cache):
                    prompt = TEMPLATE.format(
                        class_name="FeatureAssessment",
                        question=question.text,
                        feature_type=feature.type.value,
                        feature_verb_ideal=feature.verb_ideal,
                        feature_verb_actual=feature.verb_actual,
                        feature=feature.description,
                        answer=student_answer
                    )
                    
                    llm, _, _ = llm_client(structured_output=FeatureAssessment)
                    result = llm.invoke(prompt)
                    
                    assessments.append({
                        "feature": feature.description,
                        "feature_type": feature.type.name,
                        "satisfied": result.satisfied,    
                        "motivation": result.motivation    
                    })
                
                print(f"[TOOL] assess_feature: Completed {len(assessments)} assessments")
                
                # Calcola statistiche per tipo di feature
                core_count = sum(1 for a in assessments if a["feature_type"] == "CORE")
                core_satisfied = sum(1 for a in assessments if a["feature_type"] == "CORE" and a["satisfied"])
                
                important_count = sum(1 for a in assessments if a["feature_type"] == "DETAILS_IMPORTANT")
                important_satisfied = sum(1 for a in assessments if a["feature_type"] == "DETAILS_IMPORTANT" and a["satisfied"])
                
                additional_count = sum(1 for a in assessments if a["feature_type"] == "DETAILS_ADDITIONAL")
                additional_satisfied = sum(1 for a in assessments if a["feature_type"] == "DETAILS_ADDITIONAL" and a["satisfied"])
                
                # Calcola score come nel sistema assess
                core_percentage = (core_satisfied / core_count * 0.70) if core_count > 0 else 0.0
                important_percentage = (important_satisfied / important_count * 0.20) if important_count > 0 else 0.20
                additional_percentage = (additional_satisfied / additional_count * 0.10) if additional_count > 0 else 0.10
                
                final_percentage = core_percentage + important_percentage + additional_percentage
                estimated_score = round(final_percentage * question.weight, 2)
                
                result = {
                    "question_id": question_id,
                    "question_text": question.text,
                    "assessments": assessments,
                    "statistics": {
                        "total_features": len(assessments),
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
                }
                
                # Auto-save if student_code provided
                if student_code and self.exam_dir:
                    try:
                        pattern = f"Q* - {question_id}/{student_code} - *"
                        matching_dirs = list(self.exam_dir.glob(pattern))
                        
                        if matching_dirs:
                            student_dir = matching_dirs[0]
                            summary_file = student_dir / "assessment_summary.json"
                            
                            with open(summary_file, "w", encoding="utf-8") as f:
                                json.dump(result, f, indent=2, ensure_ascii=False)
                            
                            print(f"[TOOL] assess_feature: Auto-saved to {summary_file}")
                            result["saved_to"] = str(summary_file)
                    except Exception as e:
                        print(f"[TOOL] assess_feature: Auto-save failed: {e}")
                
                return json.dumps(result, indent=2, ensure_ascii=False)
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"[ERROR] assess_feature: {error_details}")
                return json.dumps({
                    "error": str(e),
                    "details": error_details
                })

        tools["assess_and_save_feature"] = assess_and_save_feature

        # Wrap all tools with error logging
        wrapped_tools = {}
        for name, func in tools.items():
            wrapped_tools[name] = tool_wrapper(name, func)
        
        return wrapped_tools
    
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