"""Unit tests for per-user phase-mapper path helpers."""
import pytest

from app.tools import phase_mapper_paths as pmp


def test_slugify_matches_reference_app_behavior():
    assert pmp.slugify("DMA-Rb") == "DMA-Rb"
    assert pmp.slugify("DMA / Rb library!") == "DMA_Rb_library"
    assert pmp.slugify("   ") == "library"


def test_library_root_is_namespaced_by_user(monkeypatch, tmp_path):
    monkeypatch.setattr(pmp, "get_user_data_dir", lambda: str(tmp_path))
    root_a = pmp.library_root("user-1", "lib-a")
    root_b = pmp.library_root("user-2", "lib-a")
    assert root_a != root_b
    assert str(tmp_path) in str(root_a)
    assert "user-1" in str(root_a)


def test_uploads_dir_variants(monkeypatch, tmp_path):
    monkeypatch.setattr(pmp, "get_user_data_dir", lambda: str(tmp_path))
    plate_dir = pmp.library_uploads_dir("user-1", "lib-a", "plate_reader")
    xrd_dir = pmp.library_uploads_dir("user-1", "lib-a", "raw_xrd")
    assert plate_dir != xrd_dir
    assert plate_dir.exists()  # created eagerly
    assert xrd_dir.exists()


def test_new_run_dir_is_timestamped_and_not_created(monkeypatch, tmp_path):
    monkeypatch.setattr(pmp, "get_user_data_dir", lambda: str(tmp_path))
    run_dir = pmp.new_run_dir("user-1", "lib-a")
    assert run_dir.parent == pmp.library_runs_dir("user-1", "lib-a")
    assert not run_dir.exists()


@pytest.mark.parametrize(
    "bad_user_id",
    [
        "../../etc",
        "../escape",
        "/etc/passwd",
        "user/../../escape",
        "user/inner",
        "",
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        pmp.library_root,
        pmp.library_inputs_dir,
        pmp.library_runs_dir,
        pmp.new_run_dir,
    ],
)
def test_malicious_user_id_is_rejected(monkeypatch, tmp_path, bad_user_id, func):
    monkeypatch.setattr(pmp, "get_user_data_dir", lambda: str(tmp_path))
    with pytest.raises(ValueError):
        func(bad_user_id, "lib-a")


def test_malicious_user_id_is_rejected_for_uploads_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(pmp, "get_user_data_dir", lambda: str(tmp_path))
    with pytest.raises(ValueError):
        pmp.library_uploads_dir("../escape", "lib-a", "plate_reader")


@pytest.mark.parametrize(
    "malicious_slug,expected_fragment",
    [
        ("../../etc/passwd", "etc_passwd"),
        ("/etc/passwd", "etc_passwd"),
        ("../escape", "escape"),
        ("a/b/../../c", "a_b_c"),
    ],
)
def test_malicious_slug_is_neutralized_and_stays_sandboxed(
    monkeypatch, tmp_path, malicious_slug, expected_fragment
):
    monkeypatch.setattr(pmp, "get_user_data_dir", lambda: str(tmp_path))
    root = pmp.library_root("user-1", malicious_slug)

    sandbox = (tmp_path / "phase_mapper").resolve()
    resolved = root.resolve()
    assert sandbox in resolved.parents or resolved == sandbox
    assert str(sandbox) in str(resolved)
    # No path traversal segments survive into the resulting directory name.
    assert ".." not in root.parts
    assert not any(part.startswith("/") and part != "/" for part in root.parts)


def test_malicious_slug_cannot_escape_via_uploads_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(pmp, "get_user_data_dir", lambda: str(tmp_path))
    path = pmp.library_uploads_dir("user-1", "/etc/passwd", "plate_reader")
    sandbox = (tmp_path / "phase_mapper").resolve()
    assert sandbox in path.resolve().parents
