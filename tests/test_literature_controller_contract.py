import subprocess
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = (
    BACKEND_ROOT
    / "literature_agent"
    / "controller"
    / "literature_agent_full_end_to_end_v21_3_english_sanitizer.py"
)


def test_controller_help_exposes_cross_domain_runtime_flags():
    result = subprocess.run(
        [sys.executable, str(CONTROLLER), "--help"],
        cwd=CONTROLLER.parent,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert "--domain_profile" in result.stdout
    assert "--ontology_path" in result.stdout
    assert "--knowledge_graph_builder" in result.stdout


def test_controller_directory_supports_sibling_domain_pack_fallback():
    probe = """
try:
    from literature_agent.domain_pack import load_structured_output_contract
except ModuleNotFoundError as exc:
    if exc.name not in {"literature_agent", "literature_agent.domain_pack"}:
        raise
    from domain_pack import load_structured_output_contract

contract = load_structured_output_contract(sys.argv[1])
assert "domain_measurements" in contract.fields
print(contract.source_path)
"""
    pack = (
        BACKEND_ROOT
        / "literature_agent"
        / "config"
        / "domain_packs"
        / "custom_domain_template.json"
    )
    result = subprocess.run(
        [sys.executable, "-c", "import sys\n" + probe, str(pack)],
        cwd=CONTROLLER.parent,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert str(pack.resolve()) in result.stdout
