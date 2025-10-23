"""
MCP Client con LangGraph per orchestrazione multi-step.
"""

import asyncio
import os
from pathlib import Path
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.tools import tool

from exam.llm_provider import llm_client
from exam.mcp import ExamMCPServer


# ============================================================================
# DEFINIZIONE DELLO STATO CONDIVISO
# ============================================================================

class AssessmentState(TypedDict):
    """Stato condiviso tra i nodi del grafo."""

    # Input/Output
    messages: Annotated[List[BaseMessage], "Chat history"]
    task: str
    result: str

    # Dati caricati (per evitare reload)
    loaded_answers: dict  # {(question_id, student_code): answer_text}
    loaded_checklists: dict  # {question_id: checklist}

    # Dati dell'esame
    exam_loaded: bool
    exam_questions: list
    exam_students: list

    # Risultati valutazioni
    assessments: list  # Lista di assessment completati

    # Stato workflow
    current_question_id: str
    current_student_code: str
    students_to_assess: list
    next_action: str  # "assess_next", "done", "error"


# ============================================================================
# NODI DEL GRAFO
# ============================================================================

class ExamAssessmentGraph:
    """Grafo LangGraph per valutazione esami."""

    def __init__(self):
        self.mcp_server = ExamMCPServer()
        self.llm, _, _ = llm_client()
        self.tools = self._create_tools()
        self.graph = self._build_graph()

    def _create_tools(self):
        """Crea i tool base (stessi di prima)."""

        @tool
        async def load_student_answer(question_id: str, student_code: str) -> str:
            """Load a student's answer into memory."""
            return await self.mcp_server.tools["load_student_answer"](question_id, student_code)

        @tool
        async def load_checklist(question_id: str) -> str:
            """Load assessment checklist for a question."""
            return await self.mcp_server.tools["load_checklist"](question_id)

        @tool
        async def assess_all_features(question_id: str, student_code: str) -> str:
            """Assess all features for a student's answer."""
            return await self.mcp_server.tools["assess_all_features"](question_id, student_code)

        @tool
        async def load_exam_from_yaml(questions_file: str, responses_file: str) -> str:
            """Load exam from YAML files."""
            return await self.mcp_server.tools["load_exam_from_yaml"](questions_file, responses_file)

        return [load_student_answer, load_checklist, assess_all_features, load_exam_from_yaml]

    # ------------------------------------------------------------------------
    # NODI: Ogni nodo è una funzione che modifica lo stato
    # ------------------------------------------------------------------------

    async def load_exam_node(self, state: AssessmentState) -> AssessmentState:
        """Nodo: Carica l'esame dai file YAML."""
        print("\n[LOAD_EXAM] Caricamento esame...")

        import json
        result = await self.mcp_server.tools["load_exam_from_yaml"](
            "se-2025-06-05-questions.yml",
            "se-2025-06-05-responses.yml"
        )

        data = json.loads(result)

        if "error" in data:
            state["next_action"] = "error"
            state["result"] = f"Errore caricamento: {data['error']}"
            return state

        # Estrai domande e studenti
        exam_id = data["exam_id"]
        exam_data = self.mcp_server.context.loaded_exams[exam_id]

        state["exam_loaded"] = True
        state["exam_questions"] = exam_data["questions"]
        state["exam_students"] = exam_data["students"]
        state["students_to_assess"] = [s["email"] for s in exam_data["students"]]
        state["next_action"] = "assess_next"

        print(f"[LOAD_EXAM] ✓ Caricati {len(state['exam_students'])} studenti")

        return state

    async def prepare_student_node(self, state: AssessmentState) -> AssessmentState:
        """Nodo: Prepara prossimo studente da valutare."""
        print("\n[PREPARE_STUDENT] Preparazione studente...")

        if not state["students_to_assess"]:
            state["next_action"] = "done"
            return state

        # Prendi prossimo studente
        student_email = state["students_to_assess"][0]
        state["current_student_code"] = student_email

        print(f"[PREPARE_STUDENT] → {student_email[:30]}...")

        state["next_action"] = "load_checklists"
        return state

    async def load_checklists_node(self, state: AssessmentState) -> AssessmentState:
        """Nodo: Carica tutte le checklist necessarie."""
        print("\n[LOAD_CHECKLISTS] Caricamento checklist...")

        for question in state["exam_questions"]:
            question_id = question["id"]

            # Salta se già caricata
            if question_id in state.get("loaded_checklists", {}):
                continue

            print(f"[LOAD_CHECKLISTS] Loading {question_id}...")
            await self.mcp_server.tools["load_checklist"](question_id)

            if "loaded_checklists" not in state:
                state["loaded_checklists"] = {}
            state["loaded_checklists"][question_id] = True

        print(f"[LOAD_CHECKLISTS] ✓ {len(state['loaded_checklists'])} checklist caricate")

        state["next_action"] = "assess_student"
        return state

    async def assess_student_node(self, state: AssessmentState) -> AssessmentState:
        """Nodo: Valuta tutte le risposte dello studente corrente."""
        print("\n[ASSESS_STUDENT] Valutazione in corso...")

        import json
        student_email = state["current_student_code"]

        result = await self.mcp_server.tools["assess_student_exam"](student_email)
        assessment_data = json.loads(result)

        if "error" not in assessment_data:
            if "assessments" not in state:
                state["assessments"] = []

            state["assessments"].append({
                "student": student_email,
                "score": assessment_data["calculated_score"],
                "max_score": assessment_data["max_score"],
                "percentage": assessment_data["percentage"]
            })

            print(f"[ASSESS_STUDENT] ✓ Score: {assessment_data['calculated_score']}/{assessment_data['max_score']}")

        # Rimuovi studente completato
        state["students_to_assess"] = state["students_to_assess"][1:]

        # Decidi prossima azione
        if state["students_to_assess"]:
            state["next_action"] = "assess_next"
        else:
            state["next_action"] = "done"

        return state

    async def generate_report_node(self, state: AssessmentState) -> AssessmentState:
        """Nodo: Genera report finale."""
        print("\n[REPORT] Generazione report...")

        if not state.get("assessments"):
            state["result"] = "Nessuna valutazione completata"
            return state

        # Calcola statistiche
        scores = [a["score"] for a in state["assessments"]]
        avg_score = sum(scores) / len(scores)

        report = f"""
REPORT VALUTAZIONE ESAME
{'='*70}

Studenti valutati: {len(state['assessments'])}
Punteggio medio: {avg_score:.2f}
Punteggio massimo: {max(scores):.2f}
Punteggio minimo: {min(scores):.2f}

DETTAGLIO STUDENTI:
"""

        for i, assessment in enumerate(state["assessments"], 1):
            email_preview = assessment["student"][:30] + "..."
            report += f"\n{i}. {email_preview}"
            report += f"\n   Score: {assessment['score']:.2f}/{assessment['max_score']} ({assessment['percentage']:.1f}%)"

        state["result"] = report
        print(report)

        return state

    # ------------------------------------------------------------------------
    # FUNZIONI DI ROUTING (decidono il prossimo nodo)
    # ------------------------------------------------------------------------

    def route_after_prepare(self, state: AssessmentState) -> str:
        """Decide dove andare dopo prepare_student."""
        if state["next_action"] == "done":
            return "generate_report"
        return "load_checklists"

    def route_after_assess(self, state: AssessmentState) -> str:
        """Decide dove andare dopo assess_student."""
        if state["next_action"] == "assess_next":
            return "prepare_student"
        return "generate_report"

    # ------------------------------------------------------------------------
    # COSTRUZIONE DEL GRAFO
    # ------------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        """Costruisce il grafo di valutazione."""

        # Crea il grafo
        workflow = StateGraph(AssessmentState)

        # Aggiungi i nodi
        workflow.add_node("load_exam", self.load_exam_node)
        workflow.add_node("prepare_student", self.prepare_student_node)
        workflow.add_node("load_checklists", self.load_checklists_node)
        workflow.add_node("assess_student", self.assess_student_node)
        workflow.add_node("generate_report", self.generate_report_node)

        # Definisci le transizioni
        workflow.set_entry_point("load_exam")

        workflow.add_edge("load_exam", "prepare_student")

        # Routing condizionale dopo prepare_student
        workflow.add_conditional_edges(
            "prepare_student",
            self.route_after_prepare,
            {
                "load_checklists": "load_checklists",
                "generate_report": "generate_report"
            }
        )

        workflow.add_edge("load_checklists", "assess_student")

        # Routing condizionale dopo assess_student
        workflow.add_conditional_edges(
            "assess_student",
            self.route_after_assess,
            {
                "prepare_student": "prepare_student",
                "generate_report": "generate_report"
            }
        )

        workflow.add_edge("generate_report", END)

        return workflow.compile()

    # ------------------------------------------------------------------------
    # ESECUZIONE
    # ------------------------------------------------------------------------

    async def run(self, task: str = "Assess full exam", max_students: int = None):
        """Esegue il grafo di valutazione."""

        print("\n" + "="*70)
        print(f"LANGGRAPH WORKFLOW: {task}")
        print("="*70)

        # Stato iniziale
        initial_state = AssessmentState(
            messages=[HumanMessage(content=task)],
            task=task,
            result="",
            loaded_answers={},
            loaded_checklists={},
            exam_loaded=False,
            exam_questions=[],
            exam_students=[],
            assessments=[],
            current_question_id="",
            current_student_code="",
            students_to_assess=[],
            next_action="load_exam"
        )

        import time
        start = time.time()

        try:
            # Esegui il grafo
            final_state = await self.graph.ainvoke(initial_state)

            elapsed = time.time() - start

            print("\n" + "="*70)
            print("WORKFLOW COMPLETATO")
            print("="*70)
            print(final_state["result"])
            print(f"\nCompletato in {elapsed:.2f} secondi")
            print("="*70 + "\n")

            return final_state

        except Exception as e:
            print(f"\nErrore: {e}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# DEMO
# ============================================================================

async def demo_langgraph():
    """Demo con LangGraph."""
    print("\nDEMO: LangGraph Workflow")
    print("="*70)

    graph = ExamAssessmentGraph(Path("mock_exam_submissions"))

    # Esegui workflow automatico
    await graph.run(
        task="Assess June 2025 exam for first 3 students",
        max_students=3
    )


async def main():
    """Entry point."""

    if not os.environ.get("GROQ_API_KEY"):
        print("\nGROQ_API_KEY not set!")
        return

    print("\nExam Assessment with LangGraph")
    print("This will automatically:")
    print("  1. Load exam from YAML")
    print("  2. Load all checklists")
    print("  3. Assess each student")
    print("  4. Generate final report")

    input("\nPress Enter to start...")

    await demo_langgraph()


if __name__ == "__main__":
    asyncio.run(main())