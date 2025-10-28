"""
Multi-Agent PARALLELO con Send API (versione corretta).
"""

import asyncio
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.types import Send  # ✓ Import corretto
from langchain_core.messages import BaseMessage, HumanMessage
import operator
import time

from exam.mcp import ExamMCPServer


# ============================================================================
# STATO CONDIVISO
# ============================================================================

class MultiAgentAssessmentState(TypedDict):
    """Stato globale condiviso."""

    exam_loaded: bool
    exam_questions: list
    exam_students: list
    loaded_checklists: dict

    student_batches: list
    num_workers: int

    assessments: Annotated[list, operator.add]

    result: str


class WorkerState(TypedDict):
    """Stato privato di ogni worker."""
    worker_id: int
    batch: list
    assessments: list


# ============================================================================
# SISTEMA MULTI-AGENTE CON SEND API
# ============================================================================

class TrueParallelExamAssessment:
    """Sistema con workers VERAMENTE paralleli usando Send."""

    def __init__(self, exam_date: str, num_workers: int = 3):
        self.mcp_server = ExamMCPServer()
        self.num_workers = num_workers
        self.exam_date = exam_date
        self.graph = self._build_graph()

    # ------------------------------------------------------------------------
    # NODO SETUP
    # ------------------------------------------------------------------------

    async def setup_node(self, state: MultiAgentAssessmentState) -> dict:
        """Carica esame e checklist."""
        print("\n" + "=" * 70)
        print(f"[SETUP] Caricamento esame del {self.exam_date}...")
        print("=" * 70)

        import json

        # Costruisce i nomi dei file basandosi sulla data
        questions_file = f"se-{self.exam_date}-questions.yml"
        responses_file = f"se-{self.exam_date}-responses.yml"
        grades_file = f"se-{self.exam_date}-grades.yml"  # Add grades file

        print(f"[SETUP] File domande: {questions_file}")
        print(f"[SETUP] File risposte: {responses_file}")
        print(f"[SETUP] File voti: {grades_file}")

        result = await self.mcp_server.tools["load_exam_from_yaml"](
            questions_file,
            responses_file,
            grades_file
        )
        data = json.loads(result)

        if "error" in data:
            print(f"[SETUP] ✗ Errore: {data['error']}")
            return {"exam_loaded": False}

        exam_id = data["exam_id"]
        exam_data = self.mcp_server.context.loaded_exams[exam_id]

        print(f"[SETUP] Caricati {len(exam_data['students'])} studenti")

        # Carica checklist in parallelo
        checklist_tasks = [
            self.mcp_server.tools["load_checklist"](q["id"])
            for q in exam_data["questions"]
        ]
        await asyncio.gather(*checklist_tasks)

        loaded_checklists = {q["id"]: True for q in exam_data["questions"]}

        print(f"[SETUP] ✓ {len(loaded_checklists)} checklist caricate")
        print("=" * 70 + "\n")

        return {
            "exam_loaded": True,
            "exam_questions": exam_data["questions"],
            "exam_students": exam_data["students"],
            "loaded_checklists": loaded_checklists
        }

    # ------------------------------------------------------------------------
    # NODO DISTRIBUTOR (ritorna dict, NON Send!)
    # ------------------------------------------------------------------------

    async def distribute_node(self, state: MultiAgentAssessmentState) -> dict:
        """Divide studenti in batch e aggiorna lo stato."""
        print("\n[DISTRIBUTOR] Divisione studenti tra workers...")

        students = [s["email"] for s in state["exam_students"]]
        batch_size = len(students) // self.num_workers + 1

        batches = []
        for i in range(0, len(students), batch_size):
            batch = students[i:i + batch_size]
            batches.append(batch)

        for i, batch in enumerate(batches):
            print(f"[DISTRIBUTOR] Worker {i}: {len(batch)} studenti")

        print(f"[DISTRIBUTOR] ✓ Creazione di {len(batches)} branch paralleli\n")

        # ⚡ Ritorna SOLO un dict per aggiornare lo stato
        return {
            "student_batches": batches,
            "num_workers": len(batches)
        }

    # ------------------------------------------------------------------------
    # FUNZIONE CHE CREA I SEND (usata in conditional_edges)
    # ------------------------------------------------------------------------

    def create_worker_sends(self, state: MultiAgentAssessmentState) -> list[Send]:
        """Crea i Send objects per i branch paralleli."""
        sends = []

        for worker_id, batch in enumerate(state["student_batches"]):
            # Crea stato privato per ogni worker
            worker_state = WorkerState(
                worker_id=worker_id,
                batch=batch,
                assessments=[]
            )

            # Crea un Send per ogni worker
            sends.append(Send("worker", worker_state))

        print(f"[DISPATCHER] Lancio di {len(sends)} workers in parallelo!\n")
        return sends

    # ------------------------------------------------------------------------
    # NODO WORKER (eseguito in parallelo)
    # ------------------------------------------------------------------------

    async def worker_node(self, state: WorkerState) -> dict:
        """Worker che valuta il suo batch."""
        worker_id = state["worker_id"] + 1
        batch = state["batch"]

        print(f"[WORKER {worker_id}] AVVIO in parallelo! ({len(batch)} studenti)")

        import json
        start = time.time()

        results = []

        for idx, student_email in enumerate(batch, 1):
            print(f"[WORKER {worker_id}] [{idx}/{len(batch)}] {student_email[:30]}...")

            try:
                result = await self.mcp_server.tools["assess_student_exam"](student_email)
                assessment_data = json.loads(result)

                if "error" not in assessment_data:
                    results.append({
                        "worker_id": worker_id,
                        "student": student_email,
                        "score": assessment_data["calculated_score"],
                        "max_score": assessment_data["max_score"],
                        "percentage": assessment_data["percentage"]
                    })
                    print(f"[WORKER {worker_id}] Score: {assessment_data['calculated_score']:.2f}")

            except Exception as e:
                print(f"[WORKER {worker_id}] ✗ Errore: {e}")

        elapsed = time.time() - start
        print(f"[WORKER {worker_id+1}] COMPLETATO in {elapsed:.2f}s ({len(results)} valutazioni)\n")

        # Ritorna dict che verrà aggiunto allo stato globale grazie a operator.add
        return {"assessments": results}

    # ------------------------------------------------------------------------
    # NODO REPORT
    # ------------------------------------------------------------------------

    async def report_node(self, state: MultiAgentAssessmentState) -> dict:
        """Genera report finale."""
        print("\n" + "=" * 70)
        print("[REPORT] Aggregazione risultati...")
        print("=" * 70)

        if not state.get("assessments"):
            return {"result": "Nessuna valutazione completata"}

        assessments = sorted(state["assessments"], key=lambda x: x["student"])

        scores = [a["score"] for a in assessments]
        avg_score = sum(scores) / len(scores)

        # Statistiche per worker
        worker_stats = {}
        for a in assessments:
            wid = a["worker_id"]
            if wid not in worker_stats:
                worker_stats[wid] = []
            worker_stats[wid].append(a["score"])

        report = f"""
{'=' * 70}
REPORT VALUTAZIONE MULTI-AGENTE PARALLELO (Send API)
{'=' * 70}

ESAME DEL: {self.exam_date}

CONFIGURAZIONE:
  Workers in parallelo: {state['num_workers']}
  Studenti totali: {len(assessments)}

STATISTICHE GLOBALI:
  Punteggio medio: {avg_score:.2f}
  Punteggio massimo: {max(scores):.2f}
  Punteggio minimo: {min(scores):.2f}

STATISTICHE PER WORKER:
"""

        for wid in sorted(worker_stats.keys()):
            wscores = worker_stats[wid]
            wavg = sum(wscores) / len(wscores)
            report += f"  Worker {wid}: {len(wscores)} studenti, media {wavg:.2f}\n"

        report += f"\nDETTAGLIO STUDENTI:\n"

        for i, assessment in enumerate(assessments, 1):
            email_preview = assessment["student"][:40] + "..."
            report += f"\n{i:2d}. {email_preview}"
            report += f"\n     Score: {assessment['score']:.2f}/{assessment['max_score']} "
            report += f"({assessment['percentage']:.1f}%) [Worker {assessment['worker_id']}]"

        report += "\n" + "=" * 70

        print(report)

        return {"result": report}

    # ------------------------------------------------------------------------
    # COSTRUZIONE GRAFO (LA CHIAVE È QUI!)
    # ------------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        """Costruisce grafo con Send API."""

        workflow = StateGraph(MultiAgentAssessmentState)

        # Aggiungi nodi
        workflow.add_node("setup", self.setup_node)
        workflow.add_node("distribute", self.distribute_node)
        workflow.add_node("worker", self.worker_node)  # Verrà chiamato N volte in parallelo
        workflow.add_node("report", self.report_node)

        # Flusso
        workflow.set_entry_point("setup")
        workflow.add_edge("setup", "distribute")

        # ⚡ QUESTO È IL TRUCCO: add_conditional_edges con funzione che ritorna Send[]
        workflow.add_conditional_edges(
            "distribute",
            self.create_worker_sends,  # Funzione che ritorna lista di Send
            # Non serve path_map perché i Send specificano già il nodo target
        )

        # Dopo che tutti i worker finiscono, vai al report
        workflow.add_edge("worker", "report")
        workflow.add_edge("report", END)

        return workflow.compile()

    # ------------------------------------------------------------------------
    # ESECUZIONE
    # ------------------------------------------------------------------------

    async def run(self):
        """Esegue la valutazione parallela."""

        print("\n" + "=" * 70)
        print("MULTI-AGENT PARALLEL ASSESSMENT (Send API)")
        print(f"Esame del: {self.exam_date}")
        print(f"Workers: {self.num_workers}")
        print("=" * 70)

        initial_state = MultiAgentAssessmentState(
            exam_loaded=False,
            exam_questions=[],
            exam_students=[],
            loaded_checklists={},
            student_batches=[],
            num_workers=self.num_workers,
            assessments=[],
            result=""
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

    print("\n True Parallel Multi-Agent Assessment (Send API)")
    print("=" * 70)

    # Chiedi la data dell'esame
    exam_date = input("\nData dell'esame (formato YYYY-MM-DD, es. 2025-06-05): ").strip()

    # Validazione base del formato
    if not exam_date:
        exam_date = "2025-06-05"  # Default
        print(f"Usando data di default: {exam_date}")

    num_workers = int(input("Numero di workers (default 3): ") or "3")

    system = TrueParallelExamAssessment(exam_date=exam_date, num_workers=num_workers)
    await system.run()


if __name__ == "__main__":
    asyncio.run(main())