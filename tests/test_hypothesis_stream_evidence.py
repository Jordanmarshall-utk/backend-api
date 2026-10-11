from unittest.mock import patch

from app.services.hypothesis_stream import iter_generate_hypothesis


def test_streamed_final_generation_preserves_shared_evidence_bubble():
    expected = {
        "stage": "analysis",
        "messages": [
            {"role": "assistant", "title": "Hypothesis", "content": "Hypothesis text"},
            {
                "role": "assistant",
                "title": "LiteratureAgent grounding",
                "content": "Evidence used: 10.example/source",
            },
            {"role": "assistant", "title": "Analysis report", "content": "Analysis"},
        ],
        "assistant_message": "Hypothesis text",
        "options": [],
    }

    with patch(
        "app.services.hypothesis_stream.generate_hypothesis",
        return_value=expected,
    ) as generate:
        events = list(iter_generate_hypothesis(object()))

    generate.assert_called_once()
    assert events[-1]["type"] == "complete"
    assert events[-1]["messages"] == expected["messages"]
    assert any(
        event.get("step") == "literature_handoff"
        and event.get("messages", [{}])[0].get("title") == "LiteratureAgent grounding"
        for event in events
    )
