from typing import Dict, Any, Optional
import os

from app.agents.base import BaseAgent
from app.tools.memory import MemoryManager

# Lazy import socratic module - only import when needed
_socratic_module = None

def _lazy_import_socratic():
    """Lazy import of socratic module to speed up module loading"""
    global _socratic_module
    if _socratic_module is None:
        from app.tools import socratic
        _socratic_module = socratic
    return _socratic_module

class HypothesisAgent(BaseAgent):
    def __init__(self, name, desc, question: str):
        super().__init__(name, desc)
        self.question = question
        self.memory = MemoryManager()
        # Don't import socratic here - lazy load when needed

    def confidence(self, params: Dict[str, Any]) -> float:
        """Return confidence score for hypothesis tasks."""
        # Low confidence by default (hypothesis agent usually needs user input)
        return 0.3

    def initial_process(self, question, experimental_mode=False, experimental_constraints=None):
        socratic = _lazy_import_socratic()

        resolved_api_key = (
            self.memory.get_var("api_key")
            or os.getenv("HUGGINGFACE_API_KEY")
            or os.getenv("HF_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )

        if not resolved_api_key:
            raise ValueError("API key not set. Configure it in Settings or set HUGGINGFACE_API_KEY / GEMINI_API_KEY in the environment.")

        if not question or not question.strip():
            raise ValueError("Question is empty. Provide a valid research question.")

        try:
            clarified_question = socratic.clarify_question(question)
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error generating clarified question: {e}") from e

        if not clarified_question or not clarified_question.strip():
            raise RuntimeError(
                "LLM returned an empty clarified question. "
                "Check your API key, quota, and network connectivity."
            )

        try:
            socratic_questions = socratic.socratic_pass(clarified_question)
        except Exception as e:
            raise RuntimeError(f"Error generating Socratic questions: {e}") from e

        if not socratic_questions or not socratic_questions.strip():
            raise RuntimeError("LLM returned empty Socratic questions. Please try again.")

        socratic_answers = None
        try:
            socratic_answers = socratic.socratic_answer_questions(clarified_question, socratic_questions)
        except Exception as e:
            raise RuntimeError(f"Error generating Socratic answers: {e}") from e

        if experimental_mode and experimental_constraints:
            thoughts = socratic.tot_generation_experimental_plan(
                socratic_questions, clarified_question, experimental_constraints)
        else:
            thoughts = socratic.tot_generation(socratic_questions, clarified_question, socratic_answers)
            if not thoughts:
                raise RuntimeError("Tree-of-Thought generation returned empty results. Please try again.")

        if len(thoughts) < 3:
            thoughts = list(thoughts) + [""] * (3 - len(thoughts))

        return clarified_question, socratic_questions, thoughts[:3], socratic_answers

    def build_conversation_context(self, prompt_session_id: Optional[str] = None):
        """ Build full conversation context from all interactions """
        if prompt_session_id is None:
            try:
                prompt_session_id = self.memory.get_var("current_prompt_session_id")
            except Exception:
                prompt_session_id = None
        
        context_parts = []

        # Get initial question
        initial_q = self.memory.view_component("initial_question")
        if initial_q:
            context_parts.append(f"Initial Question: {initial_q}")

        # Get clarified question
        clarified = self.memory.view_component("clarified_question")
        if clarified:
            context_parts.append(f"Clarified Question: {clarified}")

        # Get socratic pass
        socratic_pass = self.memory.view_component("socratic_pass")
        if socratic_pass:
            context_parts.append(f"Socratic Analysis: {socratic_pass}")

        # Get all thoughts
        thought1 = self.memory.view_component("first_thought_1")
        thought2 = self.memory.view_component("second_thought_1")
        thought3 = self.memory.view_component("third_thought_1")
        if thought1 or thought2 or thought3:
            thoughts = [t for t in [thought1, thought2, thought3] if t]
            context_parts.append(f"Initial Thoughts: {'; '.join(thoughts)}")

        # Get all selected options and their responses
        selected_options = []
        for i in self.memory.get_var("interactions", []):
            if i.get("component") == "option_choice":
                selected_options.append(f"Selected: {i.get('message')}")
            elif i.get("component") == "next_step_option_1":
                selected_options.append(f"Option 1: {i.get('message')}")
            elif i.get("component") == "next_step_option_2":
                selected_options.append(f"Option 2: {i.get('message')}")
            elif i.get("component") == "next_step_option_3":
                selected_options.append(f"Option 3: {i.get('message')}")
            elif i.get("component") == "additional_question":
                selected_options.append(f"Additional Question: {i.get('message')}")

        if selected_options:
            context_parts.append(f"Conversation Flow: {'; '.join(selected_options[-10:])}")  # Last 10 interactions

        # Get socratic questions from iterations
        retry_q = self.memory.view_component("retry_thinking_question")
        if retry_q:
            context_parts.append(f"Latest Socratic Question: {retry_q}")

        return "\n\n".join(context_parts) if context_parts else "No previous context available."

    def get_context_for_followup(self):
        """ Get context from previous conversations for follow-up questions """
        if self.memory.get_var("conversation_events", []):
            latest = self.memory.get_latest_history()
            if not latest:
                return ""
            payload = latest.get("payload", {})
            context = f"Previous question: {payload.get('question', '')}\n"
            if payload.get("thoughts"):
                context += f"Previous thoughts: {str(payload['thoughts'])[:200]}...\n"
            if payload.get("hypothesis"):
                context += f"Previous hypothesis: {payload['hypothesis']}\n"
            return context
        return ""

    def _build_negative_hypotheses_context(self) -> str:
        """Build context from past negative hypotheses for model learning."""
        negative = self.memory.get_negative_hypotheses(limit=None)
        if not negative:
            return ""
        parts = [
            "=== Past hypotheses that did not work ===",
            "IMPORTANT: Do NOT repeat these approaches. Your new hypothesis must differ and address why these failed.",
            "",
        ]
        for i, nh in enumerate(negative, 1):
            hyp_preview = (nh.get("hypothesis_text") or "")[:600]
            if len(nh.get("hypothesis_text", "") or "") > 600:
                hyp_preview += "..."
            parts.append(f"[{i}] Status: {nh.get('status', 'unknown')}")
            parts.append(f"    Hypothesis: {hyp_preview}")
            if nh.get("analysis_summary"):
                summary = nh["analysis_summary"][:500] + ("..." if len(nh["analysis_summary"]) > 500 else "")
                parts.append(f"    Why it failed: {summary}")
            parts.append("")
        return "\n".join(parts)

    def generate_hypothesis_with_context(self, socratic_question, next_step_option, previous_option_1,
                                         previous_option_2, conversation_context):
        """ Generate hypothesis with full conversation context """
        try:
            # Include past negative hypotheses so model can learn from them
            negative_ctx = self._build_negative_hypotheses_context()
            full_context = conversation_context
            if negative_ctx:
                full_context = negative_ctx + "\n\n--- Current conversation ---\n\n" + conversation_context
            socratic = _lazy_import_socratic()
            return socratic.hypothesis_synthesis(
                socratic_question,
                next_step_option,
                previous_option_1,
                previous_option_2,
                full_context
            )
        except Exception as e:
            raise RuntimeError(f"Error generating hypothesis: {str(e)}. Please check your API key and try again.") from e

    def run_agent(self, memory) -> Dict[str, Any]:
        """Run hypothesis agent headlessly via the API."""
        return {
            "status": "ready",
            "message": "Use POST /api/v1/agents/hypothesis/chat or /chat/stream for the interactive hypothesis flow.",
            "stage": str(memory.get_var("stage") or "initial"),
            "hypothesis_preview": (memory.view_component("hypothesis") or "")[:500] or None,
        }

