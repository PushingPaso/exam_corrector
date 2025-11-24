"""
Multi-Agent PARALLELO con Send API e Integrazione MLflow.
"""

import asyncio
import operator
import time
import json
import statistics  # Aggiunto per calcoli statistici
from typing import TypedDict, Annotated

# --- MLFLOW IMPORTS ---
import mlflow
import mlflow.langchain
# ----------------------

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from exam.llm_provider import llm_client
from exam.mcp import ExamMCPServer


# ============================================================================
# STATO CONDIVISO (Invariato)
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


class ExamAssessment:
    """Sistema con workers VERAMENTE paralleli usando Send."""

    def __init__(self, exam_date: str):
        self.mcp_server = ExamMCPServer()
        self.exam_date = exam_date
        self.graph = self._build_graph()
        self.llm, self.model_name, _ = llm_client() # Recupero anche il nome modello
        self.tools= [self.mcp_server.load_exam_from_yaml_tool,self.mcp_server.load_checklist,self.mcp_server.assess_student_exam]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    # ------------------------------------------------------------------------
    # NODO SETUP
    # ------------------------------------------------------------------------

    async def setup_node(self, state: MultiAgentAssessmentState) -> dict:
        """Carica esame e checklist chiamando direttamente i tool."""
        print("\n" + "=" * 70)
        print(f"[SETUP] Caricamento esame del {self.exam_date}...")
        print("=" * 70)

        exam_tool_args = {
            "questions_file": f"se-{self.exam_date}-questions.yml",
            "responses_file": f"se-{self.exam_date}-responses.yml",
            "grades_file": f"se-{self.exam_date}-grades.yml"
        }

        try:
            print(f"[SETUP] Loading Exam Files: {exam_tool_args}")
            exam_json = await self.mcp_server.load_exam_from_yaml_tool.ainvoke(exam_tool_args)
            exam_data = json.loads(exam_json)

            if "error" in exam_data:
                print(f"[SETUP] ✗ Error loading exam: {exam_data['error']}")
                return {"exam_loaded": False}

            question_ids = exam_data["question_ids"]
            checklist_args = {"question_ids": question_ids}

            print(f"[SETUP] Loading Checklists for {len(question_ids)} questions...")
            checklist_json = await self.mcp_server.load_checklist.ainvoke(checklist_args)
            checklist_data = json.loads(checklist_json)

            print(f"[SETUP] Success! Exam ID: {exam_data['exam_id']}")

            return {
                "exam_loaded": True,
                "exam_id": exam_data["exam_id"],
                "question_ids": exam_data["question_ids"],
                "student_emails": exam_data["student_email"],
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
        students = state["student_emails"]
        print(f"\n[WORKER] Processing {len(students)} students...")
        start = time.time()
        results = []

        for student_email in students:
            print(f"[WORKER] Assessing: {student_email} ...")
            try:
                assessment_json = await self.mcp_server.assess_student_exam.ainvoke(
                    {"student_email": student_email}
                )
                assessment_data = json.loads(assessment_json)

                if "error" in assessment_data:
                    print(f"[WORKER] ✗ Error for {student_email}: {assessment_data['error']}")
                    continue

                score = assessment_data.get("calculated_score", 0)
                max_score = assessment_data.get("max_score", 30)
                percentage = assessment_data.get("percentage", 0) # Assicurati sia float/int

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
        return {"assessments": results}

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(MultiAgentAssessmentState)
        workflow.add_node("setup", self.setup_node)
        workflow.add_node("assess", self.worker_node)
        workflow.set_entry_point("setup")
        workflow.add_edge("setup", "assess")
        workflow.add_edge("assess", END)
        return workflow.compile()

    async def run(self):
        # Questo metodo ora è chiamato dentro il contesto MLflow nel main
        initial_state = MultiAgentAssessmentState(
            exam_id= "",
            question_ids = [],
            student_emails = [],
            loaded_checklists = [],
            assessments = []
        )
        return await self.graph.ainvoke(initial_state)


# ============================================================================
# DEMO CON MLFLOW
# ============================================================================

async def main():
    import os
    if not os.environ.get("GROQ_API_KEY"):
        print("\nGROQ_API_KEY not set!")
        return

    print("\n Multi-Agent Exam Assessment with MLflow")
    print("=" * 70)

    exam_date = input("\nData dell'esame (formato YYYY-MM-DD, es. 2025-06-05): ").strip()
    if not exam_date:
        exam_date = "2025-06-05"

    # Inizializza il sistema
    system = ExamAssessment(exam_date=exam_date)

    # 1. SETUP MLFLOW
    # Imposta un esperimento per raggruppare le run
    mlflow.set_experiment("Exam_Corrector_Agents")
    mlflow.set_tracking_uri("http://localhost:5000")
    # Abilita l'autologging per LangChain (cattura prompt, chain, tool calls)
    mlflow.langchain.autolog()

    print("\n[MLFLOW] Avvio tracciamento run...")

    # 2. AVVIO RUN MLFLOW
    with mlflow.start_run(run_name=f"Assessment-{exam_date}") as run:

        # Log parametri iniziali
        mlflow.log_param("exam_date", exam_date)
        mlflow.log_param("model_name", system.model_name)
        mlflow.log_param("agent_type", "LangGraph-Parallel-Simulation")

        start_time = time.time()

        # --- ESECUZIONE AGENTE ---
        final_state = await system.run()
        # -------------------------

        total_duration = time.time() - start_time

        # 3. CALCOLO E LOG METRICHE
        assessments = final_state.get("assessments", [])
        num_students = len(assessments)

        if num_students > 0:
            scores = [a['score'] for a in assessments]
            percentages = [float(a['percentage']) for a in assessments]

            avg_score = statistics.mean(scores)
            avg_percentage = statistics.mean(percentages)

            # Log metriche aggregate
            mlflow.log_metric("total_duration_seconds", total_duration)
            mlflow.log_metric("students_processed", num_students)
            mlflow.log_metric("average_score", avg_score)
            mlflow.log_metric("average_percentage", avg_percentage)
            mlflow.log_metric("min_score", min(scores))
            mlflow.log_metric("max_score", max(scores))

            # 4. SALVATAGGIO ARTIFATTI (Risultati completi)
            # Salviamo il JSON finale come artefatto consultabile nella UI
            results_file = "final_assessment_results.json"
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(final_state, f, indent=2, ensure_ascii=False)

            mlflow.log_artifact(results_file)
            print(f"[MLFLOW] Artifact {results_file} saved.")

        else:
            print("[MLFLOW] Nessuno studente valutato, nessuna metrica salvata.")
            mlflow.log_metric("students_processed", 0)

        print(f"\n[MLFLOW] Run completata. ID: {run.info.run_id}")
        print(f"Per visualizzare i risultati esegui nel terminale: mlflow ui")

if __name__ == "__main__":
    asyncio.run(main())