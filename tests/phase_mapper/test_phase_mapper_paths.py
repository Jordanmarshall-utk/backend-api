"""Unit tests for per-user phase-mapper path helpers."""
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
