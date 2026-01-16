import os
import json
import re
from pathlib import Path

# -------- CONFIG --------
DOCS_DIR = r"C:\Capstone insurance\documents"
OUTPUT_DIR = r"C:\Capstone insurance\metadata"
# ------------------------


def read_file_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def extract_title(text: str) -> str:
    """
    Title = first line starting with '# '
    """
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def extract_document_control_field(text: str, field_name: str) -> str:
    """
    Extracts lines like: 'Version: 1.0'
    field_name examples: 'Version', 'Effective Date', 'Department', 'Document Type'
    """
    pattern = rf"{field_name}\s*:\s*(.+)$"
    for line in text.splitlines():
        match = re.search(pattern, line.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def extract_purpose_section(text: str) -> str:
    """
    Extract the 'Purpose' section until the next '## ' heading.
    Assumes markdown-like structure with '## 1. Purpose' etc.
    """
    lines = text.splitlines()
    purpose_lines = []
    inside_purpose = False

    for line in lines:
        stripped = line.strip()

        # Start of purpose section
        if stripped.lower().startswith("##") and "purpose" in stripped.lower():
            inside_purpose = True
            # skip the heading line itself
            continue

        # If we were inside purpose and hit a new section, stop
        if inside_purpose and stripped.startswith("## "):
            break

        if inside_purpose:
            purpose_lines.append(stripped)

    summary = " ".join(purpose_lines).strip()
    # Limit summary length a bit if it's too long
    return summary[:500]


def build_keywords(title: str, department: str, doc_type: str) -> list:
    """
    Very simple keyword builder:
    - Split title into words
    - Add department and doc_type words
    - Lowercase, remove very short words, dedupe
    """
    raw_words = []

    for part in [title, department, doc_type]:
        if not part:
            continue
        raw_words.extend(re.split(r"\W+", part))

    cleaned = []
    for w in raw_words:
        w = w.strip().lower()
        if len(w) >= 3:  # ignore tiny words like 'of', 'to'
            cleaned.append(w)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for w in cleaned:
        if w not in seen:
            seen.add(w)
            result.append(w)

    return result


def extract_metadata_from_file(path: Path, doc_index: int) -> dict:
    text = read_file_text(path)

    file_stem = path.stem  # e.g., "Document_01_Adult_Fever_SOP"
    doc_id = f"DOC_{doc_index:03d}"

    title = extract_title(text)
    version = extract_document_control_field(text, "Version")
    effective_date = extract_document_control_field(text, "Effective Date")
    department = extract_document_control_field(text, "Department")
    doc_type = extract_document_control_field(text, "Document Type")
    summary = extract_purpose_section(text)
    keywords = build_keywords(title, department, doc_type)

    metadata = {
        "doc_id": doc_id,
        "file_name": path.name,
        "title": title,
        "department": department,
        "doc_type": doc_type,
        "version": version,
        "effective_date": effective_date,
        "summary": summary,
        "keywords": keywords,
    }

    return metadata


def main():
    docs_dir = Path(DOCS_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_list = []

    # Sort files so numbers 01..05 are consistent
    txt_files = sorted(docs_dir.glob("*.txt"))

    for idx, file_path in enumerate(txt_files, start=1):
        print(f"Processing: {file_path.name}")
        md = extract_metadata_from_file(file_path, doc_index=idx)
        metadata_list.append(md)

        # Write individual file metadata
        out_file = out_dir / f"{md['doc_id']}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(md, f, indent=2)

    # Write combined metadata file
    all_meta_path = out_dir / "all_metadata.json"
    with all_meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    print(f"\nDone. Wrote {len(metadata_list)} metadata files to: {out_dir}")


if __name__ == "__main__":
    main()
