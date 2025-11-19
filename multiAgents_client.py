"""
Multi-Agent PARALLELO con Send API (versione corretta).
"""

import asyncio
import operator
import time
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from exam.llm_provider import llm_client
from exam.mcp import ExamMCPServer


# ============================================================================
# STATO CONDIVISO
# ============================================================================

class ExamData(BaseModel):
    """A movie with details."""
    exam_id: str = Field(..., description="exam id")
    question_ids: list[str] = Field(..., description="the ids of the questions")
    student_emails: list[str]= Field(..., description="the list of student emails")

class MultiAgentAssessmentState(TypedDict):
    """Global state"""
    exam_id : str
    question_ids: list[str]
    student_emails: list[str]
    loaded_checklists: list[str]
    assessments: Annotated[list, operator.add]



class WorkerState(TypedDict):
    """Stato privato di ogni worker."""
    worker_id: int
    batch: list
    assessments: list


# ============================================================================
# SISTEMA MULTI-AGENTE CON SEND API
# ============================================================================

class ExamAssessment:
    """Sistema con workers VERAMENTE paralleli usando Send."""

    def __init__(self, exam_date: str):
        self.mcp_server = ExamMCPServer()
        self.exam_date = exam_date
        self.graph = self._build_graph()
        self.llm,_,_ = llm_client()
        self.tools= [self.mcp_server.load_exam_from_yaml_tool,self.mcp_server.load_checklist,self.mcp_server.assess_student_exam]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    # ------------------------------------------------------------------------
    # NODO SETUP
    # ------------------------------------------------------------------------

    async def setup_node(self, state: MultiAgentAssessmentState) -> dict:
        """Carica esame e checklist chiamando direttamente i tool."""
        import json

        print("\n" + "=" * 70)
        print(f"[SETUP] Caricamento esame del {self.exam_date}...")
        print("=" * 70)

        # 1. PREPARE ARGUMENTS FOR EXAM TOOL
        exam_tool_args = {
            "questions_file": f"se-{self.exam_date}-questions.yml",
            "responses_file": f"se-{self.exam_date}-responses.yml",
            "grades_file": f"se-{self.exam_date}-grades.yml"
        }

        try:
            # 2. INVOKE EXAM TOOL DIRECTLY
            print(f"[SETUP] Loading Exam Files: {exam_tool_args}")
            exam_json = await self.mcp_server.load_exam_from_yaml_tool.ainvoke(exam_tool_args)
            exam_data = json.loads(exam_json)

            if "error" in exam_data:
                print(f"[SETUP] ✗ Error loading exam: {exam_data['error']}")
                return {"exam_loaded": False}

            # 3. PREPARE ARGUMENTS FOR CHECKLIST TOOL
            # The tool expects a dictionary with the argument name 'question_ids'
            question_ids = exam_data["question_ids"]
            checklist_args = {"question_ids": question_ids}

            # 4. INVOKE CHECKLIST TOOL DIRECTLY
            print(f"[SETUP] Loading Checklists for {len(question_ids)} questions...")
            checklist_json = await self.mcp_server.load_checklist.ainvoke(checklist_args)
            checklist_data = json.loads(checklist_json)

            if "status" not in checklist_data or checklist_data["status"] != "batch_completed":
                print(f"[SETUP] ⚠ Warning loading checklists: {checklist_json}")

            # 5. UPDATE STATE
            # With TypedDict, we return a dictionary with the keys we want to update
            print(f"[SETUP] Success! Exam ID: {exam_data['exam_id']}")

            return {
                "exam_loaded": True,
                "exam_id": exam_data["exam_id"],
                "question_ids": exam_data["question_ids"],
                "student_emails": exam_data["student_email"],  # Tool returns singular 'email' list
                "loaded_checklists": checklist_data.get("details", {}).get("loaded", [])
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[SETUP] Critical Error: {e}")
            return {"exam_loaded": False}

    # ------------------------------------------------------------------------
    # NODO WORKER
    # ------------------------------------------------------------------------
    async def worker_node(self, state: MultiAgentAssessmentState) -> dict:
        """
        Worker che valuta gli studenti usando direttamente il tool assess_student_exam.
        """
        import json

        # Access state using dictionary syntax for TypedDict
        students = state["student_emails"]

        print(f"\n[WORKER] Processing {len(students)} students...")
        start = time.time()
        results = []

        for student_email in students:
            print(f"[WORKER] Assessing: {student_email} ...")

            try:
                # 1. INVOKE ASSESSMENT TOOL DIRECTLY
                # No LLM involved here, just pure python logic execution via the tool wrapper
                assessment_json = await self.mcp_server.assess_student_exam.ainvoke(
                    {"student_email": student_email}
                )

                # 2. PARSE JSON
                assessment_data = json.loads(assessment_json)

                if "error" in assessment_data:
                    print(f"[WORKER] ✗ Error for {student_email}: {assessment_data['error']}")
                    continue

                # 3. EXTRACT RELEVANT METRICS
                # Assuming the assessment tool returns a dict with these keys
                score = assessment_data.get("calculated_score", 0)
                max_score = assessment_data.get("max_score", 30)
                percentage = assessment_data.get("percentage", "0%")

                results.append({
                    "student": student_email,
                    "score": score,
                    "max_score": max_score,
                    "percentage": percentage
                })
                print(f"[WORKER] Done. Score: {score:.2f}/{max_score}")

            except Exception as e:
                print(f"[WORKER] Critical Error processing {student_email}: {e}")

        elapsed = time.time() - start
        print(f"[WORKER] BATCH COMPLETED in {elapsed:.2f}s\n")

        # Return the update for the 'assessments' key in state
        return {"assessments": results}



    def _build_graph(self) -> StateGraph:
        """Costruisce grafo con Send API."""

        workflow = StateGraph(MultiAgentAssessmentState)

        # Aggiungi nodi
        workflow.add_node("setup", self.setup_node)
        workflow.add_node("assess", self.worker_node)

        # Flusso
        workflow.set_entry_point("setup")
        workflow.add_edge("setup", "assess")
        workflow.add_edge("assess", END)

        return workflow.compile()

    # ------------------------------------------------------------------------
    # ESECUZIONE
    # ------------------------------------------------------------------------

    async def run(self):
        """Esegue la valutazione parallela."""

        print("\n" + "=" * 70)
        print("MULTI-AGENT PARALLEL ASSESSMENT (Send API)")
        print(f"Esame del: {self.exam_date}")
        print("=" * 70)

        initial_state = MultiAgentAssessmentState(
            exam_id= "",
            question_ids = [],
            student_emails = [],
            loaded_checklists = [],
            assessments = []
        )

        start = time.time()
        final_state = await self.graph.ainvoke(initial_state)
        elapsed = time.time() - start

        print(f"\n{'=' * 70}")
        print(f"⚡ COMPLETATO IN {elapsed:.2f} SECONDI ⚡")
        print(f"{'=' * 70}\n")

        return final_state


# ============================================================================
# DEMO
# ============================================================================

async def main():
    """Entry point."""

    import os
    if not os.environ.get("GROQ_API_KEY"):
        print("\nGROQ_API_KEY not set!")
        return

    print("\n Multi-Agent Exam Assessment")
    print("=" * 70)

    # Chiedi la data dell'esame
    exam_date = input("\nData dell'esame (formato YYYY-MM-DD, es. 2025-06-05): ").strip()

    # Validazione base del formato
    if not exam_date:
        exam_date = "2025-06-05"  # Default
        print(f"Usando data di default: {exam_date}")


    system = ExamAssessment(exam_date=exam_date)
    await system.run()


if __name__ == "__main__":
    asyncio.run(main())