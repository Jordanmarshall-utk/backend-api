from app.tools import polaris_orchestrator, runtime_state, socratic


class FakeMemory:
    def __init__(self):
        self.values = {}

    def set_var(self, key, value):
        self.values[key] = value

    def get_var(self, key, default=None):
        return self.values.get(key, default)


def test_selected_branch_packet_is_reused_for_synthesis_and_analysis() -> None:
    token = socratic._last_literature_handoff.set(
        {
            "query": "Unverified continuation: test DAP2+ against a monovalent spacer",
            "retrieval_query": "Unverified continuation: test DAP2+ against a monovalent spacer",
            "selected_scope": "Unverified continuation: test DAP2+ against a monovalent spacer",
            "formatted_context": "No strong direct local support.",
            "fresh_retrieval_performed": True,
        }
    )
    try:
        context = socratic._reuse_selected_branch_context(
            "Unverified continuation: test DAP2+ against a monovalent spacer"
        )
        socratic._mark_synthesis_evidence_context(context)
        handoff = socratic.get_last_literature_handoff()

        assert context == "No strong direct local support."
        assert handoff["synthesis_reused_selected_branch_packet"] is True
        assert handoff["used_for_hypothesis_synthesis"] is True
        assert handoff["formatted_context"] == context
    finally:
        socratic._last_literature_handoff.reset(token)


def test_hypothesis_retrieval_has_bounded_server_timeout(monkeypatch) -> None:
    captured = {}

    class FakeOrchestrator:
        def literature_evidence(self, query, **kwargs):
            captured.update(kwargs)
            return {
                "formatted_context": "No strong direct local support.",
                "provenance": {
                    "strong_evidence_count": 0,
                    "claim_count": 0,
                    "grounding_status": "unavailable_after_retries",
                    "fresh_retrieval_required": True,
                    "fresh_retrieval_performed": True,
                },
            }

    monkeypatch.setattr(polaris_orchestrator, "PolarisOrchestrator", FakeOrchestrator)
    token = socratic._last_literature_handoff.set(None)
    try:
        context = socratic.get_local_literature_context(
            "test selected branch",
            require_direct_support=True,
            retrieval_attempts=1,
        )
        assert context == "No strong direct local support."
        assert captured["retrieval_timeout_seconds"] == 120
    finally:
        socratic._last_literature_handoff.reset(token)


def test_handoff_can_be_restored_across_api_request_contexts() -> None:
    handoff = {
        "query": "selected DAP2+ branch",
        "retrieval_query": "selected DAP2+ branch",
        "selected_scope": "selected DAP2+ branch",
        "formatted_context": "branch-specific packet",
    }
    first = socratic._last_literature_handoff.set(handoff)
    try:
        persisted = socratic.get_last_literature_handoff()
    finally:
        socratic._last_literature_handoff.reset(first)

    second = socratic._last_literature_handoff.set(None)
    try:
        socratic.set_last_literature_handoff(persisted)
        assert socratic._reuse_selected_branch_context("selected DAP2+ branch") == "branch-specific packet"
    finally:
        socratic._last_literature_handoff.reset(second)


def test_literature_handoff_is_ephemeral_runtime_state() -> None:
    assert runtime_state.is_runtime_key("last_literature_handoff")
