from src.parser.ingestion import build_and_persist_index


def test_build_and_persist_index_success(tmp_path):
    """
    Use the real llama_index components to build and persist an index from a markdown file.

    This test:
    - writes a markdown file
    - calls build_and_persist_index (which uses SimpleDirectoryReader, MarkdownNodeParser, VectorStoreIndex)
    - asserts the returned persist directory exists and contains files
    - attempts to reload the index from the persisted storage to ensure persistence worked
    """
    md_dir = tmp_path / "stcced2022"
    md_dir.mkdir()
    md_file = md_dir / "stcced2022.md"
    md_file.write_text("# Example\ncontent", encoding="utf-8")

    index_out_dir = tmp_path / "processed"

    returned = build_and_persist_index(md_dir, index_out_dir)

    # expected index dir
    expected_index_dir = index_out_dir / f"{md_file.stem}_index"
    assert returned == expected_index_dir
    assert expected_index_dir.exists() and expected_index_dir.is_dir()

    # persisted directory should contain at least one file
    entries = list(expected_index_dir.iterdir())
    assert len(entries) > 0, "Persisted index directory is empty"

    # attempt to reload the index using the official API
    from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex

    storage_context = StorageContext.from_defaults(persist_dir=str(expected_index_dir))
    loaded = load_index_from_storage(storage_context, index_id=f"{md_file.stem}_index")

    # basic sanity checks for the loaded index
    assert loaded is not None
    assert isinstance(loaded, VectorStoreIndex)
