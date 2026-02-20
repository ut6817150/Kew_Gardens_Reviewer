import streamlit as st
import tempfile
import json
from pathlib import Path
from docx import Document
import io
import contextlib
import pandas as pd
from collections import defaultdict
from dataclasses import asdict, is_dataclass

from preprocessing.assessment_processor import parse_docx_to_dict
from iucn_rules_checker.engine import IUCNRuleChecker

# ----------------------------
# Helpers (must be defined before use)
# ----------------------------
def to_dict_safe(x):
    """Convert dataclass / objects with to_dict() / dict-like into a plain dict."""
    if hasattr(x, "to_dict"):
        return x.to_dict()
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return x
    if hasattr(x, "__dict__"):
        return dict(x.__dict__)
    return {"value": str(x)}


def severity_str(sev):
    """Support Enum severities or plain strings."""
    return sev.value if hasattr(sev, "value") else str(sev)


# ----------------------------
# App
# ----------------------------

st.set_page_config(page_title="IUCN Assessment Feedback Tool", layout="centered")
st.title("IUCN Assessment Feedback Tool")


uploaded = st.file_uploader("Upload a .docx, .html or .htm file", type=["docx", "html", "doc"])

tmp_path: Path | None = None
uploaded_ext: str | None = None

# Handle uploaded file
if uploaded:
    uploaded_ext = Path(uploaded.name).suffix.lower()  # ".docx" or ".html"

    # Doc file -> raise error
    if uploaded_ext == ".doc":
        st.error("The feedback tool only supports .docx files. Please convert .doc to .docx and re-upload. Or, upload a HTML file.")

    # Any other file type uploaded
    elif uploaded_ext not in [".docx", ".html"]:
        st.error("The feedback tool only supports .docx or HTML files.")

    # Docx/HTML File -> to temp directory
    elif uploaded_ext == ".docx" or uploaded_ext == ".html":
        # Save using the same filename (so Path(docx_path).stem matches uploaded name)
        tmp_dir = Path(tempfile.mkdtemp(prefix="upload_"))
        safe_name = Path(uploaded.name).name  # strips any path components
        tmp_path = tmp_dir / safe_name
        tmp_path.write_bytes(uploaded.getbuffer())

        st.success(f"Uploaded: {uploaded.name}")
        st.caption(f"Temp file: {tmp_path}")

# Generate Feedback button
run = st.button("Generate Feedback", type="primary", disabled=(tmp_path is None))


# Generate Feedback
if run and tmp_path is not None:

    # Docx/HTML file process
    if uploaded_ext == ".docx" or uploaded_ext == ".html":
        pass

    #convert to python dict
    data = parse_docx_to_dict(str(tmp_path))

    # ----------------------------
    # Rules-based system output
    # ----------------------------
    checker = IUCNRuleChecker()
    report = checker.check_json(data)

    st.divider()
    st.subheader("Rules Based System Output")

    # Group violations by assessment_section
    violations = getattr(report, "violations", None)
    if violations is None:
        report_dict = to_dict_safe(report)
        violations = report_dict.get("violations", [])

    grouped = defaultdict(list)
    for v in violations:
        vd = to_dict_safe(v)
        section = vd.get("assessment_section") or "Whole Document"
        grouped[section].append(vd)

    st.subheader("Violations by assessment_section")

    # Display grouped sections
    for section in sorted(grouped.keys(), key=lambda s: (s != "Whole Document", s)):
        vs = grouped[section]

        with st.expander(f"{section} ({len(vs)})", expanded=(section == "Whole Document")):
            rows = []
            for vd in vs:
                pos = vd.get("position") or {}
                rows.append({
                    "severity": severity_str(vd.get("severity")),
                    "category": vd.get("category", ""),
                    "rule_name": vd.get("rule_name", ""),
                    "message": vd.get("message", ""),
                    "matched_text": vd.get("matched_text", ""),
                    "line": pos.get("line", None),
                    "column": pos.get("column", None),
                })

            st.dataframe(rows, use_container_width=True)

    # (optional) download the grouped violations as JSON
    st.subheader("Download violations")
    grouped_bytes = json.dumps(grouped, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button(
        label="Download Violations (grouped JSON)",
        data=grouped_bytes,
        file_name=f"{Path(uploaded.name).stem}_violations_grouped.json",
        mime="application/json",
    )
        
        