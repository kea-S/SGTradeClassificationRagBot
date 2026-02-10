from src.parser.ingestion import build_and_persist_index


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

    # attempt to reload the index using the official API
    from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex

    storage_context = StorageContext.from_defaults(persist_dir=str(processed_dir))
    loaded = load_index_from_storage(storage_context, index_id=f"{md_dir.stem}_index")

    # basic sanity checks for the loaded index
    assert loaded is not None
    assert isinstance(loaded, VectorStoreIndex)
