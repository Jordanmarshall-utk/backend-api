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
assert contract.pack_id == "custom_domain"
assert contract.pack_version == "1.0"
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


def test_cross_domain_controller_rejects_legacy_inline_vision():
    result = subprocess.run(
        [
            sys.executable,
            str(CONTROLLER),
            "--base_csv",
            str(BACKEND_ROOT / "data" / "Perovskite_database_content_all_data.csv"),
            "--domain_profile",
            "general_science",
            "--inline_vision",
            "1",
        ],
        cwd=CONTROLLER.parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2
    assert "Inline vision uses the legacy halide-PV prompt" in result.stderr


def test_embedded_controller_contains_no_personal_retrieval_identity():
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "Jmarsh39" + "@vols.utk.edu" not in source
    assert "client_secret_" + "298117362376" not in source
