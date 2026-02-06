import streamlit as st
import tempfile
import json
from pathlib import Path
from docx import Document

from assessment_processor import parse_docx_to_dict

st.set_page_config(page_title="Assessment Feedback Tool", layout="centered")
st.title("Assessment Feedback Tool")

uploaded = st.file_uploader("Upload a Word file (.docx)", type=["docx"])

tmp_path: Path | None = None
if uploaded:
    # Save using the same filename (so Path(docx_path).stem matches uploaded name)
    tmp_dir = Path(tempfile.mkdtemp(prefix="upload_"))
    safe_name = Path(uploaded.name).name  # strips any path components
    tmp_path = tmp_dir / safe_name
    tmp_path.write_bytes(uploaded.getbuffer())

    st.success(f"Uploaded: {uploaded.name}")
    st.caption(f"Temp file: {tmp_path}")

run = st.button("Generate Feedback", type="primary", disabled=(tmp_path is None))

if run and tmp_path is not None:
    if tmp_path.suffix.lower() != ".docx":
        st.error("parse_docx_to_dict currently supports .docx only. Please convert .doc → .docx and re-upload.")
        st.stop()

    with st.spinner("Parsing document..."):
        data = parse_docx_to_dict(str(tmp_path))

    st.success("Parsed successfully!")

    # Display parsed structure
    st.subheader("Parsed JSON (preview)")
    st.json(data)

    # Download parsed JSON
    st.subheader("Download")
    json_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button(
        label="Download JSON",
        data=json_bytes,
        file_name=f"{Path(uploaded.name).stem}.json",
        mime="application/json",
    )