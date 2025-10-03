"""
MCP (Model Context Protocol) Server for Exam Correction System.
Exposes tools that LLMs can use to interact with the exam assessment system.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from exam import QuestionsStore, Question
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
        self.questions_store = QuestionsStore()
        self.vector_store = None
        
        # Try to initialize vector store (optional)
        try:
            self.vector_store = sqlite_vector_store()
            print("# MCP Server: RAG vector store loaded")
        except Exception as e:
            print(f"# MCP Server: RAG not available: {e}")
        
        # Register all tools
        self.tools = self._create_tools()
    
    def _create_tools(self):
        """Create all available tools as a dictionary."""
        
        tools = {}
        
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
                return json.dumps({"error": f"Question {question_id} not found"})
        
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
                JSON string with SHOULD, SHOULDNT, EXAMPLE, and SEE_ALSO features
            """
            try:
                question = self.questions_store.question(question_id)
                print(question+"!!!!")
                answer = load_answer_cache(question)


                
                if not answer:
                    return json.dumps({"error": f"No checklist found for question {question_id}"})
                
                return json.dumps({
                    "question_id": question_id,
                    "question_text": question.text,
                    "should": answer.should,
                    "should_not": answer.should_not,
                    "examples": answer.examples,
                    "see_also": answer.see_also
                }, indent=2)
            except KeyError:
                return json.dumps({"error": f"Question {question_id} not found"})
        
        tools["get_checklist"] = get_checklist
        
        # Tool 6: assess_feature
        async def assess_feature(
            question_id: str,
            feature_description: str,
            feature_type: str,
            student_answer: str
        ) -> str:
            """
            Assess whether a specific feature is present in a student's answer.
            
            Args:
                question_id: The ID of the question
                feature_description: Description of what to look for
                feature_type: Type of feature ("SHOULD", "SHOULDNT", "EXAMPLE", "SEE_ALSO")
                student_answer: The student's answer text
            
            Returns:
                JSON string with assessment result (satisfied, motivation)
            """
            try:
                from exam.openai import llm_client
                from exam.assess import TEMPLATE
                question = self.questions_store.question(question_id)

                # Create feature object
                feature_type_lower = feature_type.lower().replace("_", " ")
                if feature_type_lower == "shouldnt":
                    feature_type_lower = "mistake"
                feature_type_enum = FeatureType(feature_type_lower)
                feature = Feature(type=feature_type_enum, description=feature_description)
                
                # Prepare prompt
                prompt = TEMPLATE.format(
                    class_name="FeatureAssessment",
                    question=question.text,
                    feature_type=feature.type.value,
                    feature_verb_ideal=feature.verb_ideal,
                    feature_verb_actual=feature.verb_actual,
                    feature=feature.description,
                    answer=student_answer
                )
                
                # Get LLM client
                llm, _, _ = llm_client(structured_output=FeatureAssessment)
                
                # Invoke assessment
                result = llm.invoke(prompt)
                
                return json.dumps({
                    "feature": feature_description,
                    "feature_type": feature_type,
                    "satisfied": result.satisfied,
                    "motivation": result.motivation
                }, indent=2)
                
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        tools["assess_feature"] = assess_feature
        
        # Tool 7: calculate_score
        async def calculate_score(assessments_json: str, max_score: float) -> str:
            """
            Calculate the score based on feature assessments.
            
            Args:
                assessments_json: JSON string with list of assessments
                                 [{"feature_type": "SHOULD", "satisfied": true}, ...]
                max_score: Maximum possible score for the question
            
            Returns:
                JSON string with calculated score and breakdown
            """
            try:
                assessments = json.loads(assessments_json)
                
                should_total = sum(1 for a in assessments if a["feature_type"] == "SHOULD")
                should_satisfied = sum(1 for a in assessments 
                                     if a["feature_type"] == "SHOULD" and a["satisfied"])
                
                shouldnt_total = sum(1 for a in assessments if a["feature_type"] == "SHOULDNT")
                shouldnt_violated = sum(1 for a in assessments 
                                      if a["feature_type"] == "SHOULDNT" and not a["satisfied"])
                
                if should_total == 0:
                    return json.dumps({"error": "No SHOULD features to evaluate"})
                
                base_percentage = should_satisfied / should_total
                penalty_per_error = 0.15
                error_penalty = shouldnt_violated * penalty_per_error
                final_percentage = max(0.0, base_percentage - error_penalty)
                score = round(final_percentage * max_score, 2)
                
                breakdown = f"Base: {should_satisfied}/{should_total} SHOULD ({base_percentage*100:.0f}%)"
                if shouldnt_violated > 0:
                    breakdown += f" - Errors: {shouldnt_violated}/{shouldnt_total} mistakes (-{error_penalty*100:.0f}%)"
                breakdown += f" = {final_percentage*100:.0f}% of {max_score} = {score}"
                
                return json.dumps({
                    "score": score,
                    "max_score": max_score,
                    "percentage": round(final_percentage * 100, 1),
                    "breakdown": breakdown,
                    "details": {
                        "should_satisfied": should_satisfied,
                        "should_total": should_total,
                        "errors": shouldnt_violated,
                        "error_total": shouldnt_total
                    }
                }, indent=2)
                
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        tools["calculate_score"] = calculate_score
        
        # Tool 8: search_course_material
        async def search_course_material(query: str, max_results: int = 5) -> str:
            """
            Search course materials using RAG (Retrieval-Augmented Generation).
            
            Args:
                query: Search query
                max_results: Maximum number of results to return (default: 5)
            
            Returns:
                JSON string with relevant course material snippets
            """
            if not self.vector_store:
                return json.dumps({"error": "RAG vector store not available"})
            
            try:
                results = self.vector_store.similarity_search(query, k=max_results)
                
                materials = []
                for doc in results:
                    materials.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "lines": doc.metadata.get("lines", [0, 0]),
                        "index": doc.metadata.get("index", 0)
                    })
                
                return json.dumps({
                    "query": query,
                    "results_count": len(materials),
                    "materials": materials
                }, indent=2)
                
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        tools["search_course_material"] = search_course_material
        
        # Tool 9: generate_feedback
        async def generate_feedback(assessments_json: str) -> str:
            """
            Generate constructive feedback based on assessments.
            
            Args:
                assessments_json: JSON string with list of assessments
            
            Returns:
                JSON string with generated feedback text
            """
            try:
                assessments = json.loads(assessments_json)
                
                feedback_parts = []
                
                # Positive feedback
                satisfied = [a for a in assessments 
                           if a.get("feature_type") == "SHOULD" and a.get("satisfied")]
                if satisfied:
                    feedback_parts.append("**Strengths:**")
                    for a in satisfied[:3]:  # Top 3
                        feedback_parts.append(f"- {a.get('motivation', 'Good work')}")
                
                # Areas for improvement
                unsatisfied = [a for a in assessments 
                             if a.get("feature_type") == "SHOULD" and not a.get("satisfied")]
                if unsatisfied:
                    feedback_parts.append("\n**Areas for Improvement:**")
                    for a in unsatisfied:
                        feedback_parts.append(f"- {a.get('motivation', 'Needs work')}")
                
                # Errors to avoid
                errors = [a for a in assessments 
                         if a.get("feature_type") == "SHOULDNT" and not a.get("satisfied")]
                if errors:
                    feedback_parts.append("\n**Errors to Correct:**")
                    for a in errors:
                        feedback_parts.append(f"- {a.get('motivation', 'Avoid this error')}")
                
                feedback_text = "\n".join(feedback_parts)
                
                return json.dumps({
                    "feedback": feedback_text,
                    "strengths_count": len(satisfied),
                    "improvements_needed": len(unsatisfied),
                    "errors_found": len(errors)
                }, indent=2)
                
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        tools["generate_feedback"] = generate_feedback
        
        # Tool 10: save_assessment
        async def save_assessment(
            question_id: str,
            student_code: str,
            score: float,
            assessments_json: str,
            feedback: str = ""
        ) -> str:
            """
            Save the assessment results for a student.
            
            Args:
                question_id: The ID of the question
                student_code: The student's code
                score: The calculated score
                assessments_json: JSON string with all feature assessments
                feedback: Optional feedback text
            
            Returns:
                JSON string with save confirmation
            """
            if not self.exam_dir:
                return json.dumps({"error": "No exam directory configured"})
            
            try:
                # Find student directory
                pattern = f"Q* - {question_id}/{student_code} - *"
                matching_dirs = list(self.exam_dir.glob(pattern))
                
                if not matching_dirs:
                    return json.dumps({"error": f"Student directory not found"})
                
                student_dir = matching_dirs[0]
                
                # Save summary
                summary_file = student_dir / "assessment_summary.json"
                summary = {
                    "question_id": question_id,
                    "student_code": student_code,
                    "score": score,
                    "assessments": json.loads(assessments_json),
                    "feedback": feedback
                }
                
                with open(summary_file, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                
                return json.dumps({
                    "status": "success",
                    "saved_to": str(summary_file),
                    "score": score
                }, indent=2)
                
            except Exception as e:
                return json.dumps({"error": str(e)})
        
        tools["save_assessment"] = save_assessment
        
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