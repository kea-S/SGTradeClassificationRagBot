from src.parser.ingestion import build_and_persist_index
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex


def test_build_and_persist_index_success(tmp_path):
    """
    Build an index from a markdown directory and persist directly into the
    provided `processed` directory. The function now persists files directly
    under `index_out_dir`
    """
    md_dir = tmp_path / "stcced2022"
    md_dir.mkdir()
    md_file = md_dir / "stcced2022.md"
    md_file.write_text("# Example\ncontent", encoding="utf-8")

    processed_dir = tmp_path / "processed"

    returned = build_and_persist_index(md_dir, processed_dir)

    # build_and_persist_index now returns the processed_dir (where files are written)
    assert returned == processed_dir
    assert processed_dir.exists() and processed_dir.is_dir()

    # persisted directory should contain at least one file
    entries = list(processed_dir.iterdir())
    assert len(entries) > 0, "Persisted index directory is empty"

    storage_context = StorageContext.from_defaults(persist_dir=str(processed_dir))
    loaded = load_index_from_storage(storage_context, index_id=f"{md_dir.stem}_index")

    # basic sanity checks for the loaded index
    assert loaded is not None
    assert isinstance(loaded, VectorStoreIndex)


def test_build_and_persist_index_skips_when_marker_present(tmp_path):
    """
    If the index marker file exists in the output directory, the function should
    skip building/persisting and return early without creating new files.
    """
    md_dir = tmp_path / "stcced2022"
    md_dir.mkdir()
    md_file = md_dir / "stcced2022.md"
    md_file.write_text("# Example\ncontent", encoding="utf-8")

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)

    index_id = f"{md_dir.stem}_index"
    marker = processed_dir / f"{index_id}.ingested"
    marker.write_text("Ingested: 2026-01-01T00:00:00Z\n", encoding="utf-8")

    before = sorted([p.name for p in processed_dir.iterdir()])

    returned = build_and_persist_index(md_dir, processed_dir)

    assert returned == processed_dir

    after = sorted([p.name for p in processed_dir.iterdir()])
    # No new files should have been created when marker exists
    assert after == before

    # marker still exists and contains our sentinel
    assert marker.exists()
    assert "Ingested" in marker.read_text()
