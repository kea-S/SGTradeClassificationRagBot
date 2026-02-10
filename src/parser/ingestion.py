import logging
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser

from pymupdf4llm import to_markdown

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()


def pdf_to_markdown(pdf_path: Path, out_dir: Path) -> Path:
    """
    Convert a PDF to markdown and write the result to out_dir/<stem>.md.
    Returns the path to the written markdown file.
    """
    pdf_path = pdf_path.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Converting PDF %s -> markdown", pdf_path)
    md_text = to_markdown(str(pdf_path))

    md_path = out_dir / f"{pdf_path.stem}.md"
    md_path.write_text(md_text, encoding="utf-8")
    logger.info("Wrote markdown to %s", md_path)
    return md_path


def build_and_persist_index(md_dir: Path, index_out_dir: Path) -> Path:
    """
    Build a VectorStoreIndex from a directory of markdown files and persist it under
    index_out_dir/<stem>_index. Returns the path to the persisted index directory.

    Note: md_dir must be a directory (not a single file). The index name is derived
    from md_dir.stem.
    """
    md_dir = Path(md_dir).resolve()
    if not md_dir.exists() or not md_dir.is_dir():
        raise ValueError(f"md_dir must be an existing directory containing markdown files: {md_dir}")

    index_out_dir = Path(index_out_dir).resolve()
    index_dir = index_out_dir / f"{md_dir.stem}_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading documents from markdown directory %s", md_dir)
    documents = SimpleDirectoryReader(str(md_dir)).load_data()

    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    logger.info("Building VectorStoreIndex from parsed nodes")

    # NOTE: llamaindex automatically uses ada, openai's embedding model,
    # to change, change in the settings:
    # https://developers.llamaindex.ai/python/framework/module_guides/models/embeddings/
    index = VectorStoreIndex(nodes)

    # Official persistence: set index id and persist storage_context
    index_id = f"{md_dir.stem}_index"
    index.set_index_id(index_id)
    index.storage_context.persist(persist_dir=str(index_dir))
    logger.info("Index persisted to %s using storage_context.persist", index_dir)
    return index_dir


def main(
    pdf_path: Optional[Path] = None,
    *,
    data_dir: Optional[Path] = None,
) -> None:
    """
    Simple script entrypoint.
    If no pdf_path is provided, defaults to data/raw/stcced2022.pdf in the repository root.
    The markdown and index will be placed into data/processed/.
    """
    repo_root = Path(__file__).resolve().parents[2]  # project root
    data_dir = (Path(data_dir) if data_dir else repo_root / "data").resolve()
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    pdf_path = Path(pdf_path) if pdf_path else raw_dir / "stcced2022.pdf"

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    md_file = pdf_to_markdown(pdf_path, processed_dir)

    # build_and_persist_index now expects a directory containing markdown files
    md_dir = md_file.parent
    build_and_persist_index(md_dir, processed_dir)


if __name__ == "__main__":
    main()
