import os
import json
from pathlib import Path

# -------- CONFIG --------
BASE_DIR = r"C:\Capstone insurance"
DOCS_DIR = os.path.join(BASE_DIR, "Documents")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
ALL_METADATA_FILE = os.path.join(METADATA_DIR, "all_metadata.json")
# ------------------------


def read_file_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def load_document_metadata() -> dict:
    """
    Load all_metadata.json and index by file_name for quick lookup.
    """
    meta_path = Path(ALL_METADATA_FILE)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        all_meta = json.load(f)

    by_file_name = {}
    for md in all_meta:
        file_name = md["file_name"]
        by_file_name[file_name] = md
    return by_file_name


def split_into_sections(text: str):
    """
    Split document into (section_title, section_body) pairs based on lines starting with '## '.
    Returns:
        title: document title (from '# ' line)
        sections: list of (section_title, section_body)
    """
    lines = text.splitlines()

    doc_title = ""
    sections = []

    current_section_title = None
    current_lines = []

    for line in lines:
        stripped = line.strip()

        # Document title
        if stripped.startswith("# ") and not doc_title:
            doc_title = stripped[2:].strip()
            continue

        # Section heading
        if stripped.startswith("## "):
            # Save previous section if exists
            if current_section_title is not None:
                section_body = "\n".join(current_lines).strip()
                sections.append((current_section_title, section_body))

            current_section_title = stripped[3:].strip()
            current_lines = []
        else:
            if current_section_title is not None:
                current_lines.append(line)

    # Save last section
    if current_section_title is not None:
        section_body = "\n".join(current_lines).strip()
        sections.append((current_section_title, section_body))

    # If no sections found, treat entire doc as one section
    if not sections:
        sections = [("Full Document", text)]

    return doc_title, sections


def ensure_chunks_dir():
    Path(CHUNKS_DIR).mkdir(parents=True, exist_ok=True)


def main():
    ensure_chunks_dir()

    docs_dir = Path(DOCS_DIR)
    metadata_by_file = load_document_metadata()
    all_chunks = []

    txt_files = sorted(docs_dir.glob("*.txt"))

    chunk_counter_global = 0

    for file_path in txt_files:
        file_name = file_path.name
        print(f"Processing document: {file_name}")

        if file_name not in metadata_by_file:
            print(f"  WARNING: No metadata found for file {file_name}, skipping.")
            continue

        doc_meta = metadata_by_file[file_name]
        doc_id = doc_meta["doc_id"]

        text = read_file_text(file_path)
        doc_title, sections = split_into_sections(text)

        # Fall back to metadata title if parsing didn't find one
        if not doc_title:
            doc_title = doc_meta.get("title", file_name)

        for idx, (section_title, section_body) in enumerate(sections, start=1):
            chunk_counter_global += 1
            chunk_id = f"{doc_id}_CHUNK_{idx:03d}"

            # Build chunk text (nice to have title + section)
            chunk_text = f"# {doc_title}\n## {section_title}\n\n{section_body}\n"

            # Write chunk to file
            chunk_file_name = f"{chunk_id}.txt"
            chunk_file_path = Path(CHUNKS_DIR) / chunk_file_name
            with chunk_file_path.open("w", encoding="utf-8") as cf:
                cf.write(chunk_text)

            # Build chunk metadata record
            chunk_record = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "file_name": file_name,
                "chunk_file": chunk_file_name,
                "title": doc_title,
                "section_title": section_title,
                "section_index": idx,
                "department": doc_meta.get("department", ""),
                "doc_type": doc_meta.get("doc_type", ""),
                "version": doc_meta.get("version", ""),
                "effective_date": doc_meta.get("effective_date", ""),
                "text": chunk_text.strip(),
            }

            all_chunks.append(chunk_record)

    # Write all chunks metadata to JSON
    chunks_meta_path = Path(CHUNKS_DIR) / "chunks_metadata.json"
    with chunks_meta_path.open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nTotal chunks created: {len(all_chunks)}")
    print(f"Chunks directory: {CHUNKS_DIR}")
    print(f"Chunks metadata: {chunks_meta_path}")


if __name__ == "__main__":
    main()
