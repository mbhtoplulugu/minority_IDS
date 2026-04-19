import os
import ast
from pathlib import Path
from docx import Document
from docx.shared import Pt

# Files to document
FILES = [
    "unsw_nb15_pipeline.py",
    "fast_binary_expert.py",
    "heuristic_ensemble.py",
    "stack_experts.py",
    "lightgbm_model.py",
    "lightgbm_model_v2.py",
    "ae_cnn_model.py",
    "autoencoder.py",
]

REPO_ROOT = Path(__file__).parent

def extract_docstrings_and_comments(file_path: Path):
    """Return a list of (type, content) tuples.
    type: 'docstring' or 'comment'
    content: string
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    sections = []
    # Module docstring
    module_doc = ast.get_docstring(tree)
    if module_doc:
        sections.append(("docstring", f"Module docstring:\n{module_doc}"))
    # Function / class docstrings and inline comments
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if doc:
                name = node.name
                sections.append(("docstring", f"{node.__class__.__name__} `{name}` docstring:\n{doc}"))
        # Collect comments via source lines
    # Simple line‑based comment extraction (lines starting with #)
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            sections.append(("comment", f"Line {i}: {stripped}"))
    return sections

def add_section(doc: Document, title: str, content_lines):
    doc.add_heading(title, level=2)
    for typ, txt in content_lines:
        if typ == "docstring":
            p = doc.add_paragraph(txt)
            p.style = "Intense Quote"
        else:  # comment
            p = doc.add_paragraph(txt)
            p.style = "Quote"

def main():
    doc = Document()
    doc.add_heading("UNSW‑NB15 IDS Pipeline – Detailed Documentation", level=0)
    for fname in FILES:
        path = REPO_ROOT / fname
        if not path.exists():
            continue
        doc.add_heading(fname, level=1)
        sections = extract_docstrings_and_comments(path)
        # Group by type for readability
        docstrings = [s for s in sections if s[0] == "docstring"]
        comments = [s for s in sections if s[0] == "comment"]
        if docstrings:
            add_section(doc, "Docstrings", docstrings)
        if comments:
            add_section(doc, "Comments", comments)
    out_path = REPO_ROOT / "pipeline_detailed_report.docx"
    doc.save(out_path)
    print(f"Report generated at {out_path}")

if __name__ == "__main__":
    main()
