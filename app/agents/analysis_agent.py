from typing import Dict, Any, Optional, List
import json
import os

from app.agents.base import BaseAgent
from app.tools.mcp_orchestrator_bridge import sync_hypothesis_outcome
from app.tools.memory import MemoryManager
from app.tools.experiment_memory import get_experiment_memory


# Lazy import socratic module - only import when needed
_socratic_module = None

def _lazy_import_socratic():
    global _socratic_module
    if _socratic_module is None:
        from app.tools import socratic
        _socratic_module = socratic
    return _socratic_module


class AnalysisAgent(BaseAgent):
    """
    Analysis Agent that evaluates curve fitting results, relates them to hypothesis
    and experimental data, decides if more experiments are needed, and provides
    literature-backed explanations of results and impacts.
    """
    
    def __init__(self, name: str = "Analysis Agent", desc: str = "Analyzes curve fitting results and relates them to hypothesis and experiments"):
        super().__init__(name, desc)
        self.memory = MemoryManager()
    
    def confidence(self, payload: Dict[str, Any]) -> float:
        """
        Return confidence score for analysis tasks.
        High confidence when curve fitting or ML results are available.
        Can run with goal-only (no hypothesis/experiment required).
        """
        # Check if curve fitting results exist
        curve_fitting_results = self._get_curve_fitting_results()
        if curve_fitting_results:
            return 0.9

        # High confidence if GP or Monte Carlo results exist (can analyze goal + ML only)
        try:
            gp_results = self.memory.get_var("gp_results")
            mc_results = self.memory.get_var("monte_carlo_results")
            research_goal = self.memory.get_var("research_goal", "")
            if (gp_results or mc_results) and research_goal:
                return 0.85
        except (RuntimeError, AttributeError):
            pass

        # Medium confidence if hypothesis and experiment data exist
        try:
            hypothesis = self.memory.view_component("hypothesis")
            if not hypothesis:
                hypothesis = self.memory.get_var("last_hypothesis")
            experimental_outputs = self.memory.get_var("experimental_outputs")
        except (RuntimeError, AttributeError):
            hypothesis = None
            experimental_outputs = None

        if hypothesis and experimental_outputs:
            return 0.7

        # Low confidence otherwise
        return 0.3
    
    def _get_curve_fitting_results(self) -> Optional[Dict[str, Any]]:
        """Retrieve curve fitting results from memory or results directory."""
        import logging
        stored = self.memory.get_var("curve_fitting_results")
        if stored:
            return stored

        results_dir = "results"
        if os.path.exists(results_dir):
            json_files = [f for f in os.listdir(results_dir) if f.endswith("_peak_fit_results.json")]
            if json_files:
                latest_file = max(json_files, key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
                file_path = os.path.join(results_dir, latest_file)
                try:
                    with open(file_path, "r") as f:
                        return json.load(f)
                except Exception as e:
                    logging.warning("Could not load curve fitting results from %s: %s", file_path, e)

        return None
    
    def _get_gp_results(self) -> Optional[Dict[str, Any]]:
        """Retrieve Gaussian Process model results from session state."""
        return self.memory.get_var("gp_results")
    
    def _get_hypothesis_context(self) -> str:
        """Build context string from hypothesis agent outputs."""
        context_parts = []
        
        # Get hypothesis
        hypothesis = self.memory.view_component("hypothesis") or self.memory.get_var("last_hypothesis")
        if hypothesis:
            context_parts.append(f"**Hypothesis:**\n{hypothesis}\n")
        
        # Get clarified question
        clarified_q = self.memory.view_component("clarified_question")
        if clarified_q:
            context_parts.append(f"**Research Question:**\n{clarified_q}\n")
        
        # Get socratic questions
        socratic_q = self.memory.view_component("socratic_pass")
        if socratic_q:
            context_parts.append(f"**Socratic Analysis:**\n{socratic_q}\n")
        
        return "\n".join(context_parts) if context_parts else "No hypothesis context available."
    
    def _get_experimental_context(self) -> str:
        """Build context string from experimental agent outputs."""
        experimental_outputs = self.memory.get_var("experimental_outputs")
        if not experimental_outputs:
            return "No experimental data available."
        
        context_parts = []
        
        if experimental_outputs.get("plan"):
            context_parts.append(f"**Experimental Plan:**\n{experimental_outputs['plan']}\n")
        
        if experimental_outputs.get("worklist"):
            context_parts.append(f"**Worklist:**\n{experimental_outputs['worklist'][:500]}...\n")
        
        return "\n".join(context_parts) if context_parts else "No experimental data available."
    
    def _summarize_curve_fitting_results(self, results: Dict[str, Any]) -> str:
        """Create a summary of curve fitting results for LLM analysis."""
        summary_parts = []
        
        # Extract key information from results
        if isinstance(results, dict):
            # Check if it's the format from run_complete_analysis
            if "results" in results:
                all_results = results["results"]
                summary_parts.append(f"**Total Wells Analyzed:** {len(all_results)}\n")
                
                # Count successful fits
                successful = sum(1 for r in all_results if r.get('fit_result', {}).get('success', False))
                summary_parts.append(f"**Successful Fits:** {successful}/{len(all_results)}\n")
                
                # Extract peak information
                peak_summaries = []
                for result in all_results[:10]:  # Limit to first 10 for summary
                    well_name = result.get('well_name', 'Unknown')
                    read = result.get('read', '')
                    fit_result = result.get('fit_result', {})
                    
                    if fit_result.get('success'):
                        r2 = fit_result.get('stats', {}).get('r2', 0)
                        peaks = fit_result.get('peaks', [])
                        peak_info = f"  - {well_name} (Read {read}): R²={r2:.3f}, {len(peaks)} peaks"
                        if peaks:
                            peak_centers = [p.get('center', 0) for p in peaks[:3]]
                            peak_info += f" at {[f'{c:.1f}nm' for c in peak_centers]}"
                        peak_summaries.append(peak_info)
                
                if peak_summaries:
                    summary_parts.append("**Sample Results:**\n" + "\n".join(peak_summaries) + "\n")
            else:
                # Try to extract from other formats
                summary_parts.append(f"**Results Summary:**\n{json.dumps(results, indent=2)[:1000]}...\n")
        
        return "\n".join(summary_parts) if summary_parts else "No detailed results available."
    
    def _summarize_gp_results(self, gp_results: Dict[str, Any]) -> str:
        """Create a summary of GP model results for LLM analysis, including acquisition batch."""
        summary_parts = []
        
        model_type = gp_results.get("model_type", "Unknown")
        summary_parts.append(f"**Model Type:** {model_type}\n")
        if model_type == "DualTorchGP":
            summary_parts.append(
                f"**Performance Target:** {gp_results.get('performance_target', gp_results.get('target', 'Unknown'))}\n"
            )
            summary_parts.append(
                f"**Stability Target:** {gp_results.get('stability_target', 'Unknown')}\n"
            )
            summary_parts.append(
                f"**Acquisition Method:** {gp_results.get('acquisition_method', 'UCB + stability')}\n"
            )
            acq = gp_results.get("acquisition_stats", {})
            if acq:
                summary_parts.append(
                    f"**Acquisition Score:** mean={acq.get('mean', 0):.4f}, max={acq.get('max', 0):.4f}\n"
                )
        else:
            summary_parts.append(f"**Kernel:** {gp_results.get('kernel', 'Unknown')}\n")
            summary_parts.append(f"**Target Variable:** {gp_results.get('target', 'Unknown')}\n")
            summary_parts.append(
                f"**Acquisition Method:** {gp_results.get('acquisition_method', 'UCB')}\n"
            )
            summary_parts.append(
                f"**Cross-Validation R²:** {gp_results.get('cv_score', 0):.4f} ± {gp_results.get('cv_std', 0):.4f}\n"
            )
        
        uncertainty_stats = gp_results.get('uncertainty_stats', {})
        if uncertainty_stats:
            summary_parts.append(f"**Uncertainty:** Mean={uncertainty_stats.get('mean', 0):.4f}, Max={uncertainty_stats.get('max', 0):.4f}\n")
        
        top_candidates = gp_results.get('top_candidates', [])
        if top_candidates:
            summary_parts.append(f"**Top Exploration Candidates:** {len(top_candidates)}\n")
            for i, candidate in enumerate(top_candidates[:5], 1):
                summary_parts.append(f"  {i}. {candidate.get('Candidate', candidate.get('candidate', 'Unknown'))}: Predicted={candidate.get('Predicted Value', candidate.get('predicted_value', 'N/A'))}, Uncertainty={candidate.get('Uncertainty', candidate.get('uncertainty', 'N/A'))}, Acquisition={candidate.get('Acquisition Score', candidate.get('acquisition_score', 'N/A'))}\n")
        
        # Include full batch from acquisition function for literature validation
        suggested = gp_results.get('suggested_compositions', [])
        if suggested:
            summary_parts.append(f"\n**Acquisition Batch (prioritized compositions for next experiments):** {len(suggested)} candidates\n")
            for i, sc in enumerate(suggested[:10], 1):
                comps = sc.get('compositions', {})
                pred = sc.get('predicted_value', 'N/A')
                unc = sc.get('uncertainty', 'N/A')
                acq = sc.get('acquisition_score', 'N/A')
                comp_str = ", ".join(f"{k}={v:.2f}" for k, v in comps.items()) if isinstance(comps, dict) else str(comps)
                summary_parts.append(f"  {i}. {comp_str} | pred={pred}, unc={unc}, acq={acq}\n")
        
        return "\n".join(summary_parts)
    
    def _summarize_monte_carlo_results(self, mc_results: Dict[str, Any]) -> str:
        """Create a summary of Monte Carlo Decision Tree results for LLM analysis."""
        summary_parts = []
        
        summary_parts.append(f"**Model Type:** Monte Carlo Decision Tree\n")
        
        # Optimization stats from stdout
        stats = mc_results.get("optimization_stats", {})
        if stats:
            summary_parts.append(f"**Total Cycles:** {stats.get('total_cycles', 'N/A')}\n")
            summary_parts.append(f"**Best Quality Achieved:** {stats.get('best_quality', 'N/A'):.2f}\n")
            summary_parts.append(f"**Total Improvement:** {stats.get('total_improvement_pct', 'N/A'):+.2f}%\n")
            summary_parts.append(f"**Average Improvement/Cycle:** {stats.get('avg_improvement_per_cycle_pct', 'N/A'):+.2f}%\n")
        
        # Top candidates
        top_candidates = mc_results.get("top_candidates", [])
        if top_candidates:
            summary_parts.append(f"**Top Exploration Candidates:** {len(top_candidates)}\n")
            for i, candidate in enumerate(top_candidates[:5], 1):
                candidate_str = candidate.get("candidate", "Unknown")
                # Add any quality/score metrics if available
                metrics = []
                for key, value in candidate.items():
                    if key not in ["rank", "candidate"] and isinstance(value, (int, float)):
                        metrics.append(f"{key}={value:.3f}")
                metrics_str = f" ({', '.join(metrics)})" if metrics else ""
                summary_parts.append(f"  {i}. {candidate_str}{metrics_str}\n")
        
        # Optimization history
        history = mc_results.get("optimization_history", {})
        if history:
            summary_parts.append(f"**Optimization History:** {history.get('total_experiments', 'N/A')} experiments")
            if history.get("latest_cycle"):
                summary_parts.append(f" (Latest Cycle: {history['latest_cycle']})")
            summary_parts.append("\n")
        
        # LLM Agent Restrictions/Constraints (if available)
        llm_agent = mc_results.get("llm_agent_results", {})
        if llm_agent:
            summary_parts.append(f"**LLM Agent Analysis (Parameter Restrictions/Constraints):**\n")
            
            # Patterns (learned restrictions)
            patterns = llm_agent.get("patterns", [])
            if patterns:
                summary_parts.append(f"**Learned Patterns ({len(patterns)}):**\n")
                for i, pattern in enumerate(patterns[:5], 1):
                    param = pattern.get("parameter", "Unknown")
                    obs = pattern.get("observation", "")
                    conf = pattern.get("confidence", 0)
                    summary_parts.append(f"  {i}. **{param}**: {obs} (confidence: {conf:.2f})\n")
            
            # Config suggestions (parameter restrictions/changes)
            suggestions = llm_agent.get("suggestions", [])
            if suggestions:
                summary_parts.append(f"**Parameter Restrictions/Changes Suggested ({len(suggestions)}):**\n")
                for i, sugg in enumerate(suggestions[:5], 1):
                    param_path = sugg.get("parameter_path", "Unknown")
                    current = sugg.get("current_value", "?")
                    suggested = sugg.get("suggested_value", "?")
                    conf = sugg.get("confidence", 0)
                    summary_parts.append(f"  {i}. **{param_path}**: {current} → {suggested} (confidence: {conf:.2f})\n")
            
            # Recommendation summary
            rec_summary = llm_agent.get("recommendation_summary")
            if rec_summary:
                summary_parts.append(f"**LLM Agent Recommendation Summary:**\n{rec_summary}\n")
            elif llm_agent.get("num_patterns") or llm_agent.get("num_suggestions"):
                summary_parts.append(f"LLM agent identified {llm_agent.get('num_patterns', 0)} patterns and "
                                   f"{llm_agent.get('num_suggestions', 0)} parameter restriction suggestions.\n")
        
        return "".join(summary_parts) if summary_parts else "Monte Carlo results available but no summary generated."
    
    def _analyze_results_with_llm(
        self,
        hypothesis_context: str,
        experimental_context: str,
        curve_fitting_summary: str,
        full_results: Optional[Dict[str, Any]] = None,
        gp_results: Optional[Dict[str, Any]] = None,
        monte_carlo_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze results, relate to hypothesis/experiments,
        decide on next steps, and provide literature-backed explanations.
        """
        socratic = _lazy_import_socratic()
        
        # Import analysis instructions
        from app.tools.instruct import ANALYSIS_INSTRUCTIONS
        
        # Summarize GP results if available
        gp_summary = self._summarize_gp_results(gp_results) if gp_results else "Not provided"
        
        # Summarize Monte Carlo results if available
        mc_summary = self._summarize_monte_carlo_results(monte_carlo_results) if monte_carlo_results else "Not provided"
        
        # Build prompt with available context (hypothesis and experiment are optional)
        prompt_parts = [ANALYSIS_INSTRUCTIONS]
        if hypothesis_context and hypothesis_context.strip() and "No hypothesis context" not in hypothesis_context:
            prompt_parts.append(f"\n## Hypothesis Context\n{hypothesis_context}")
        else:
            prompt_parts.append("\n## Hypothesis Context\nNot provided - analyze based on goal and results only.")
        if experimental_context and experimental_context.strip() and "No experimental data" not in experimental_context:
            prompt_parts.append(f"\n## Experimental Context\n{experimental_context}")
        else:
            prompt_parts.append("\n## Experimental Context\nNot provided.")
        prompt_parts.append(f"\n## Curve Fitting Results Summary\n{curve_fitting_summary}")
        prompt_parts.append(f"\n## ML Model Results (Gaussian Process)\n{gp_summary}")
        prompt_parts.append(f"\n## ML Model Results (Monte Carlo Decision Tree)\n{mc_summary}")
        prompt_parts.append(f"\n## Full Results (if needed for detailed analysis)\n{json.dumps(full_results, indent=2)[:2000] if full_results else 'Not provided'}")

        prompt = "".join(prompt_parts)
        
        try:
            analysis_response = socratic.generate_text_with_llm(prompt)
            return {
                "success": True,
                "analysis": analysis_response,
                "raw_response": analysis_response
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": None
            }
    
    def _parse_analysis_response(self, analysis_text: str) -> Dict[str, Any]:
        """
        Parse the LLM analysis response to extract structured information:
        - Hypothesis validation status
        - Need for more experiments
        - Literature explanations
        - Impact assessment
        """
        parsed = {
            "hypothesis_status": "unknown",  # "confirmed", "needs_revision", "rejected", "needs_more_data"
            "more_experiments_needed": False,
            "experiment_recommendations": [],
            "literature_explanations": "",
            "impact_assessment": "",
            "key_findings": []
        }
        
        # Simple parsing - look for key phrases
        analysis_lower = analysis_text.lower()
        
        # Check hypothesis status
        if "hypothesis is confirmed" in analysis_lower or "hypothesis is supported" in analysis_lower:
            parsed["hypothesis_status"] = "confirmed"
        elif "hypothesis needs revision" in analysis_lower or "hypothesis should be modified" in analysis_lower:
            parsed["hypothesis_status"] = "needs_revision"
        elif "hypothesis is rejected" in analysis_lower or "hypothesis is not supported" in analysis_lower:
            parsed["hypothesis_status"] = "rejected"
        elif "more data" in analysis_lower or "additional experiments" in analysis_lower:
            parsed["hypothesis_status"] = "needs_more_data"
        
        # Check if more experiments needed
        if "more experiments" in analysis_lower or "additional experiments" in analysis_lower:
            parsed["more_experiments_needed"] = True
        
        # Extract experiment recommendations (look for numbered lists or bullet points)
        import re
        exp_patterns = [
            r"experiment.*?:?\s*(.+?)(?:\n|$)",
            r"recommend.*?:?\s*(.+?)(?:\n|$)",
            r"should.*?:?\s*(.+?)(?:\n|$)"
        ]
        for pattern in exp_patterns:
            matches = re.findall(pattern, analysis_text, re.IGNORECASE | re.MULTILINE)
            if matches:
                parsed["experiment_recommendations"].extend(matches[:3])  # Limit to 3
        
        # Extract literature explanations (look for "literature" or "research" sections)
        lit_pattern = r"(?:literature|research|previous studies|according to).*?:(.+?)(?=\n\n|\n##|$)"
        lit_matches = re.findall(lit_pattern, analysis_text, re.IGNORECASE | re.DOTALL)
        if lit_matches:
            parsed["literature_explanations"] = lit_matches[0][:1000]  # Limit length
        
        # Extract impact assessment
        impact_pattern = r"(?:impact|implications|significance).*?:(.+?)(?=\n\n|\n##|$)"
        impact_matches = re.findall(impact_pattern, analysis_text, re.IGNORECASE | re.DOTALL)
        if impact_matches:
            parsed["impact_assessment"] = impact_matches[0][:1000]  # Limit length
        
        return parsed
    
    def _generate_new_question(
        self,
        research_goal: str,
        analysis_summary: str,
        current_hypothesis: str,
        completed_experiments: str,
    ) -> Optional[str]:
        """Generate a new research question using Hypothesis Agent approach."""
        socratic = _lazy_import_socratic()
        
        from app.tools.instruct import ANALYSIS_NEW_QUESTION_INSTRUCTIONS
        
        prompt = ANALYSIS_NEW_QUESTION_INSTRUCTIONS.format(
            research_goal=research_goal,
            analysis_summary=analysis_summary,
            completed_experiments=completed_experiments,
            current_hypothesis=current_hypothesis,
        )
        
        try:
            new_question = socratic.generate_text_with_llm(prompt)
            return new_question.strip() if new_question else None
        except Exception as e:
            import logging
            logging.warning("Error generating new research question: %s", e)
            return None

    def run_agent(self, memory: MemoryManager) -> Dict[str, Any]:
        """Run analysis agent headlessly via the API."""
        return {
            "status": "ready",
            "message": "Use POST /api/v1/agents/analysis to run analysis.",
            "has_curve_results": self._get_curve_fitting_results() is not None,
            "has_gp_results": self.memory.get_var("gp_results") is not None,
        }
