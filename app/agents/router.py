
from typing import Any, Dict, List, Optional
import inspect
import logging

from app.tools import socratic

logger = logging.getLogger(__name__)

class AgentRouter:
    """
    Central router coordinating which agent should run next.

    Modes (configured via `memory.routing_mode`):
    - **Autonomous (LLM)**: use per‑agent `confidence` plus an LLM tie‑breaker
      that looks at current session context (files, hypothesis, experiment data).
    - **Manual Workflow**: follow the user‑defined ordered workflow in
      `memory.manual_workflow`.

    The router does not render UI itself; pages/agents can call it when they
    want the “next” agent to run.
    """

    def __init__(self, agents: List[Any], fallback_agent: Optional[Any] = None):
        self.agents = agents
        self.fallback_agent = fallback_agent

    def _find_agent_by_name(self, name: str) -> Optional[Any]:
        for agent in self.agents:
            if getattr(agent, "name", "").lower() == name.lower():
                return agent
        return None

    def _llm_select_agent(self, payload: Dict[str, Any], memory: Any) -> Optional[Any]:
        """
        Ask the LLM which agent should run next, given available agents and
        high‑level context from the current session (memory or payload.session_context).
        """
        agent_names = [getattr(a, "name", "Unnamed Agent") for a in self.agents]

        # Use session_context from payload (watcher) when available; else memory
        session_ctx = payload.get("session_context") or {}
        uploaded_files = []
        last_hypothesis = session_ctx.get("hypothesis_preview")
        experimental_outputs = None
        experimental_constraints = {}
        curve_fitting_results = None

        if memory:
            try:
                uploaded_files = memory.get_var("uploaded_files", [])
                last_hypothesis = last_hypothesis or memory.get_var("last_hypothesis")
                experimental_outputs = memory.get_var("experimental_outputs")
                experimental_constraints = memory.get_var("experimental_constraints", {})
                curve_fitting_results = memory.get_var("curve_fitting_results")
            except (RuntimeError, AttributeError, NameError):
                pass

        hyp_preview = (last_hypothesis[:300] + "...") if last_hypothesis and len(last_hypothesis) > 300 else last_hypothesis

        context = {
            "trigger_file": payload.get("trigger_file"),
            "source": payload.get("source"),
            "session_context": session_ctx,
            "uploaded_files": uploaded_files,
            "last_hypothesis": hyp_preview,
            "experimental_outputs": experimental_outputs,
            "experimental_constraints": experimental_constraints,
            "curve_fitting_results": curve_fitting_results is not None,
            "has_hypothesis": session_ctx.get("has_hypothesis"),
            "has_experimental_outputs": session_ctx.get("has_experimental_outputs"),
            "has_curve_fitting_results": session_ctx.get("has_curve_fitting_results"),
            "has_analysis_results": session_ctx.get("has_analysis_results"),
            "has_gp_results": session_ctx.get("has_gp_results"),
            "stage": session_ctx.get("stage"),
            "workflow_step": session_ctx.get("workflow_step"),
            "research_goal": session_ctx.get("research_goal"),
            "hypothesis_ready": session_ctx.get("hypothesis_ready"),
        }

        prompt = f"""
You are a routing controller for a lab-assistant application with multiple agents.

Available agents (choose exactly one):
- {"; ".join(agent_names)}

You are given JSON-like context about the current state (files, hypothesis, experiment data, etc.).
Decide which single agent from the list above should run *next*.

Context:
{context}

Return ONLY the exact name of the chosen agent from the list above, with no explanation, no formatting.
"""
        try:
            choice_raw = socratic.generate_text_with_llm(prompt).strip()
        except Exception as e:
            logger.warning(f"Unable to generate agent name from LLM: {e}")
            return None

        # Normalise and try to map back to a known agent
        for name in agent_names:
            if name.lower() in choice_raw.lower():
                return self._find_agent_by_name(name)

        return None

    def _route_manual(self, memory: Any) -> Optional[Any]:
        """
        Manual workflow: follow `memory.manual_workflow` and
        `memory.workflow_index`.
        """
        if not memory:
            workflow = [
                "Hypothesis Agent",
                "Experiment Agent",
                "Curve Fitting",
                "ML Models",
                "Analysis Agent",
            ]
            index = 0
        else:
            try:
                workflow = memory.get_var(
                    "manual_workflow",
                    ["Hypothesis Agent", "Experiment Agent", "Curve Fitting Agent", "ML Models", "Analysis Agent"],
                )
                index = memory.get_var("workflow_index", 0)
            except (RuntimeError, AttributeError):
                workflow = [
                "Hypothesis Agent",
                "Experiment Agent",
                "Curve Fitting",
                "ML Models",
                "Analysis Agent",
            ]
                index = 0

        if not workflow or index >= len(workflow):
            return None

        target_name = workflow[index]
        from app.services.workflow_followups import resolve_agent_name

        return self._find_agent_by_name(resolve_agent_name(target_name))

    def suggest_next_agent(self, payload: Dict[str, Any], memory: Any) -> Dict[str, Any]:
        """
        Suggest which agent should run next without executing it.
        Returns {"agent": str, "confidence": float, "session_context": dict}.
        """
        try:
            routing_mode = memory.get_var("routing_mode", "Autonomous (LLM)") if memory else "Autonomous (LLM)"
        except (RuntimeError, AttributeError):
            routing_mode = "Autonomous (LLM)"

        if routing_mode == "Manual":
            agent = self._route_manual(memory)
            return {
                "agent": getattr(agent, "name", "Fallback Agent") if agent else "Fallback Agent",
                "confidence": 1.0,
                "mode": "manual",
                "session_context": payload.get("session_context") or {},
            }

        scored = []
        for agent in self.agents:
            try:
                conf = agent.confidence(payload)
                conf = float(conf) if conf is not None else 0.0
                scored.append((conf, agent))
            except Exception:
                scored.append((0.0, agent))

        if not scored:
            return {"agent": "Fallback Agent", "confidence": 0.0, "mode": "autonomous", "session_context": payload.get("session_context") or {}}

        scored.sort(reverse=True, key=lambda x: x[0])
        score, top_agent = scored[0]

        llm_agent = None
        try:
            import os
            api_key = (
                os.getenv("HUGGINGFACE_API_KEY")
                or os.getenv("HF_API_KEY")
                or os.getenv("LLM_API_KEY")
                or os.getenv("DASHSCOPE_API_KEY")
            )
            if api_key:
                llm_agent = self._llm_select_agent(payload, memory)
        except Exception as e:
            logger.warning("LLM agent selection failed (suggest_next_agent): %s", e)

        chosen = llm_agent or top_agent
        return {
            "agent": getattr(chosen, "name", "Unknown"),
            "confidence": float(score) if score is not None else 0.0,
            "mode": "autonomous",
            "llm_override": llm_agent is not None,
            "session_context": payload.get("session_context") or {},
        }

    def route(self, payload: Dict[str, Any], memory: Any) -> Any:
        """
        Route to the next agent based on configured routing mode.

        - In autonomous mode, use agents' `confidence` plus an optional LLM
          decision using the current session context.
        - In manual mode, follow the user-configured workflow order.
        """
        # Store payload in memory so agents can access it
        try:
            if hasattr(memory, 'log_event'):
                memory.log_event(
                    "router",
                    {"current_payload": payload},
                    mode="router"
                )
        except Exception:
            pass
        
        try:
            routing_mode = memory.get_var("routing_mode", "Autonomous (LLM)") if memory else "Autonomous (LLM)"
        except (RuntimeError, AttributeError):
            routing_mode = "Autonomous (LLM)"

        # Manual workflow mode – ignore confidence and LLM, follow user order.
        if routing_mode == "Manual":
            agent = self._route_manual(memory)
            if agent is None and self.fallback_agent:
                return self.fallback_agent.run_agent(memory)
            if agent is None:
                raise RuntimeError("Manual workflow is empty or exhausted.")

            # Advance workflow index for next call (SQLite-backed, always safe)
            try:
                memory.set_var("workflow_index", memory.get_var("workflow_index", 0) + 1)
                counts = memory.get_var("agent_usage_counts", {})
                counts["router"] = counts.get("router", 0) + 1
                agent_key = getattr(agent, "name", "unknown").split()[0].lower()
                counts[agent_key] = counts.get(agent_key, 0) + 1
                memory.set_var("agent_usage_counts", counts)
            except (RuntimeError, AttributeError):
                pass
            return agent.run_agent(memory)

        # Autonomous mode: start with confidence-based scoring
        scored = []
        for agent in self.agents:
            try:
                conf = agent.confidence(payload)
                # Ensure confidence is a float (handle None)
                if conf is None:
                    conf = 0.0
                conf = float(conf)
                scored.append((conf, agent))
            except Exception as e:
                # If confidence method fails, assign low confidence
                logger.warning(f"Agent {getattr(agent, 'name', 'unknown')} confidence failed: {e}")
                scored.append((0.0, agent))
        
        if not scored:
            if self.fallback_agent:
                return self.fallback_agent.run_agent(memory)
            raise RuntimeError("No agents available.")
        
        scored.sort(reverse=True, key=lambda x: x[0])

        score, top_agent = scored[0]

        # If all agents are uncertain, fall back
        if score is None or score < 0.4:
            if self.fallback_agent:
                return self.fallback_agent.run_agent(memory)
            raise RuntimeError("No agent is confident enough.")

        # Let the LLM override / confirm the top choice using global context
        # Use LLM when API key is available (Streamlit or headless watcher with session_context)
        llm_agent = None
        try:
            import os
            api_key = (
                os.getenv("HUGGINGFACE_API_KEY")
                or os.getenv("HF_API_KEY")
                or os.getenv("LLM_API_KEY")
                or os.getenv("DASHSCOPE_API_KEY")
            )
            if api_key:
                llm_agent = self._llm_select_agent(payload, memory)
            else:
                logger.debug("Skipping LLM agent selection - no API key in environment")
        except Exception as e:
            logger.warning(f"LLM agent selection failed: {e}")
        
        chosen_agent = llm_agent or top_agent

        try:
            # Update usage counts (SQLite-backed, always safe)
            try:
                counts = memory.get_var("agent_usage_counts", {})
                counts["router"] = counts.get("router", 0) + 1
                key = getattr(chosen_agent, "name", "unknown").split()[0].lower()
                counts[key] = counts.get(key, 0) + 1
                memory.set_var("agent_usage_counts", counts)
            except (RuntimeError, AttributeError):
                pass
            
            # Pass payload to run_agent if it accepts it
            sig = inspect.signature(chosen_agent.run_agent)
            if 'payload' in sig.parameters:
                return chosen_agent.run_agent(memory, payload=payload)
            else:
                return chosen_agent.run_agent(memory)
        except Exception as e:
            logger.error(f"Error running agent {getattr(chosen_agent, 'name', 'unknown')}: {e}")
            if self.fallback_agent:
                return self.fallback_agent.run_agent(memory)
            raise