"""Unit tests for phase-mapper upload persistence and validation."""
import io

import pytest
from fastapi import UploadFile

from app.services import phase_mapper_uploads as uploads


def _upload(name: str, content: bytes) -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(content))


@pytest.fixture(autouse=True)
def _isolate_user_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("app.tools.phase_mapper_paths.get_user_data_dir", lambda: str(tmp_path))


@pytest.mark.asyncio
async def test_save_composition_csv_round_trips(tmp_path):
    csv_bytes = b"well,row,col,Cs_fraction,Rb_fraction,FA_fraction,DMA_fraction,I_fraction,Br_fraction,Cl_fraction\nA1,A,1,0,0,1,0,1,0,0\n"
    path = await uploads.save_composition_csv("user-1", "lib-a", _upload("composition.csv", csv_bytes))
    assert path.endswith("composition.csv")
    assert "user-1" in path and "lib-a" in path


@pytest.mark.asyncio
async def test_save_composition_csv_rejects_missing_well_column(tmp_path):
    csv_bytes = b"row,col,Cs_fraction\nA,1,0\n"
    with pytest.raises(ValueError, match="well"):
        await uploads.save_composition_csv("user-1", "lib-a", _upload("bad.csv", csv_bytes))


def test_validate_composition_csv_warns_without_failing(tmp_path):
    path = tmp_path / "composition.csv"
    path.write_text("well,row,col,Cs_fraction,Rb_fraction,FA_fraction,DMA_fraction,I_fraction,Br_fraction,Cl_fraction\nA1,A,1,0.5,0,0.2,0,1,0,0\n")
    warnings = uploads.validate_composition_csv(str(path))
    assert any("Cs+Rb+FA+DMA" in w for w in warnings)


@pytest.mark.asyncio
async def test_save_plate_reader_file(tmp_path):
    path = await uploads.save_plate_reader_file("user-1", "lib-a", _upload("plate.xlsx", b"binary-stub"))
    assert path.endswith("plate.xlsx")
    assert "plate_reader" in path
