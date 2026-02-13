from sg_trade_ragbot.parser.ingestion import build_and_persist_index, pdf_to_markdown
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


def test_pdf_to_markdown_skips_when_marker_present(tmp_path):
    """
    Ensure pdf_to_markdown skips converting when a marker file exists and the
    markdown file is present. Verify skip by asserting marker/md mtimes are unchanged.
    """

    out_dir = tmp_path / "intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)

    # create a dummy PDF file
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake-pdf\n")

    # create the expected markdown file and marker file before calling
    md_path = out_dir / f"{pdf_path.stem}.md"
    md_path.write_text("previously converted markdown", encoding="utf-8")

    marker_file = out_dir / f"{out_dir.stem}_md.ingested"
    marker_file.write_text("Ingested: test\n", encoding="utf-8")

    # record mtimes before the call
    md_mtime_before = md_path.stat().st_mtime
    marker_mtime_before = marker_file.stat().st_mtime

    # call the function under test
    result = pdf_to_markdown(pdf_path, out_dir)

    # assertions: should return the existing markdown path and marker should still exist
    assert result == md_path
    assert result.exists()
    assert marker_file.exists()

    # ensure files were not modified (skip-path)
    assert md_path.stat().st_mtime == md_mtime_before
    assert marker_file.stat().st_mtime == marker_mtime_before


def test_build_and_persist_index_skips_when_marker_present(tmp_path):
    """
    Ensure build_and_persist_index skips building/persisting the index when a
    marker file exists in the output directory. Verify skip by asserting the
    marker mtime and directory entries are unchanged.
    """

    # prepare a markdown directory with one markdown file
    md_dir = tmp_path / "mds"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_file = md_dir / "doc.md"
    md_file.write_text("# title\n\ncontent", encoding="utf-8")

    # prepare the processed/index output dir and pre-create the marker file
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    index_id = f"{md_dir.stem}_index"
    marker_file = processed_dir / f"{index_id}.ingested"
    marker_file.write_text("Ingested: test\n", encoding="utf-8")

    # snapshot directory listing and marker mtime before running
    entries_before = sorted(p.name for p in processed_dir.iterdir())
    marker_mtime_before = marker_file.stat().st_mtime

    # call the function under test
    result = build_and_persist_index(md_dir, processed_dir)

    # assertions: function should return the processed_dir and should not have
    # modified the marker or created new files
    assert result == processed_dir
    assert marker_file.exists()
    assert marker_file.stat().st_mtime == marker_mtime_before
    entries_after = sorted(p.name for p in processed_dir.iterdir())
    assert entries_after == entries_before
