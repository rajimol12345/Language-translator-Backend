import logging
import os
from typing import Dict, List, TypedDict, Optional
from pathlib import Path

# LangGraph Imports
from langgraph.graph import StateGraph, END

logger = logging.getLogger("translation_workflow")
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------
# 1. DEFINE THE GRAPH STATE
# --------------------------------------------------
class AgentState(TypedDict, total=False):
    """Represents the state of the translation job."""
    job_id: str
    file_path: Path
    languages: List[str]
    formats: List[str]
    paragraphs: List[Dict[str, str]]  # {"text": str, "style": str}
    translations: Dict[str, List[Dict[str, str]]]
    status: str
    progress: int
    error: Optional[str]
    complete: bool

# --------------------------------------------------
# 2. DEFINE THE NODES (WORKERS)
# --------------------------------------------------
class WorkflowNodes:
    def __init__(self, processor, translator, jobs_db: Dict):
        self.processor = processor
        self.translator = translator
        self.jobs_db = jobs_db

    def _update_db(self, state: AgentState, message: str):
        """Syncs the internal graph state with your FastAPI jobs_db."""
        jid = state["job_id"]
        if jid in self.jobs_db:
            self.jobs_db[jid].update({
                "status": state["status"],
                "progress": state["progress"],
                "message": message,
                "complete": state.get("complete", False),
                "error": state.get("error") is not None
            })

    def extract_node(self, state: AgentState) -> AgentState:
        try:
            state["status"] = "Processing"
            state["progress"] = 10
            self._update_db(state, "Extracting content and structure...")

            if not state["file_path"].exists():
                raise FileNotFoundError(f"Source file missing: {state['file_path']}")

            state["paragraphs"] = self.processor.extract_paragraphs(str(state["file_path"]))
            if not state["paragraphs"]:
                raise ValueError("Document appears to be empty or unreadable.")

            return state
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            state["error"] = str(e)
            state["status"] = "Failed"
            return state

    def translate_node(self, state: AgentState) -> AgentState:
        if state.get("error"):
            return state

        try:
            state["status"] = "Translating"
            state["translations"] = {}
            total_langs = len(state["languages"])

            for idx, lang in enumerate(state["languages"]):
                self._update_db(state, f"Translating to {lang.title()}...")

                translated_data = []
                for p in state.get("paragraphs", []):
                    text_to_translate = p.get("text", "").strip()
                    style = p.get("style", "Normal")
                    if not text_to_translate:
                        translated_data.append({"text": "", "style": style})
                        continue

                    trans_text = self.translator.translate(text_to_translate, lang)
                    translated_data.append({"text": trans_text, "style": style})

                state["translations"][lang] = translated_data
                state["progress"] = int(10 + ((idx + 1) / total_langs) * 70)

            return state
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            state["error"] = str(e)
            state["status"] = "Failed"
            return state

    def export_node(self, state: AgentState) -> AgentState:
        if state.get("error"):
            return state

        try:
            state["status"] = "Exporting"
            self._update_db(state, "Generating final files...")

            for lang, paras in state.get("translations", {}).items():
                for fmt in state["formats"]:
                    out_path = Path(self.processor.output_dir) / f"translated_{lang}_{state['job_id']}.{fmt}"

                    if fmt == "docx":
                        self.processor.export_docx(paras, out_path)
                    elif fmt == "pdf":
                        self.processor.export_pdf(paras, out_path, language=lang)
                    elif fmt == "epub":
                        self.processor.export_epub(paras, out_path, language=lang)

            state["status"] = "Completed"
            state["progress"] = 100
            state["complete"] = True
            self._update_db(state, "Job finished successfully.")
            return state
        except Exception as e:
            logger.error(f"Export failed: {e}")
            state["error"] = str(e)
            state["status"] = "Failed"
            return state

# --------------------------------------------------
# 3. ROUTING LOGIC
# --------------------------------------------------
def should_continue(state: AgentState):
    """Router: Ends execution if an error is present in the state."""
    return "end" if state.get("error") else "continue"

# --------------------------------------------------
# 4. CONSTRUCT THE GRAPH
# --------------------------------------------------
def create_translation_graph(processor, translator, jobs_db: Dict):
    nodes = WorkflowNodes(processor, translator, jobs_db)
    workflow = StateGraph(AgentState)

    workflow.add_node("extractor", nodes.extract_node)
    workflow.add_node("translator", nodes.translate_node)
    workflow.add_node("exporter", nodes.export_node)

    workflow.set_entry_point("extractor")

    workflow.add_conditional_edges("extractor", should_continue, {"continue": "translator", "end": END})
    workflow.add_conditional_edges("translator", should_continue, {"continue": "exporter", "end": END})
    workflow.add_edge("exporter", END)

    return workflow  # no .compile(), LangGraph uses the instance directly

# --------------------------------------------------
# 5. WRAPPER CLASS
# --------------------------------------------------
class TranslationWorkflow:
    def __init__(self, job_id, file_path, languages, formats, processor, translator, jobs_db):
        self.initial_state: AgentState = {
            "job_id": job_id,
            "file_path": Path(file_path),
            "languages": [l.lower() for l in languages],
            "formats": [f.lower() for f in formats],
            "paragraphs": [],
            "translations": {},
            "status": "Starting",
            "progress": 0,
            "error": None,
            "complete": False
        }
        self.graph = create_translation_graph(processor, translator, jobs_db)
        self.jobs_db = jobs_db

    def execute(self):
        """Runs the compiled LangGraph workflow."""
        try:
            final_state = self.graph.invoke(self.initial_state)

            # Cleanup source file safely
            if final_state["file_path"].exists():
                try:
                    final_state["file_path"].unlink()
                except Exception as cleanup_err:
                    logger.warning(f"Source file cleanup failed: {cleanup_err}")

            if final_state.get("error"):
                logger.error(f"Workflow terminated with error: {final_state['error']}")

        except Exception as e:
            logger.exception(f"Critical workflow failure: {e}")
