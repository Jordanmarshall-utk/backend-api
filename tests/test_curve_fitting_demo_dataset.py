from app.tools.demo_data_generator import generate_demo_dataset
from app.tools.fitting_agent import build_agent_config, curate_dataset


def test_demo_dataset_curates_despite_duplicate_padding_headers(tmp_path):
    spectral_path, composition_path = generate_demo_dataset(
        n_wells=15,
        output_dir=str(tmp_path),
    )
    config = build_agent_config(
        data_csv=spectral_path,
        composition_csv=composition_path,
        read_selection="all",
        wells_to_ignore=[],
        fill_na_value=0.0,
    )
    curated = curate_dataset(config)
    assert curated["tensor"].shape == (1, 451, 15)
    assert curated["wells"] == [
        "A1", "A10", "A11", "A12", "A2", "A3", "A4", "A5",
        "A6", "A7", "A8", "A9", "B1", "B2", "B3",
    ]
