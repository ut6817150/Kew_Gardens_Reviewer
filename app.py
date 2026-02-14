import streamlit as st
import tempfile
import json
from pathlib import Path
from docx import Document

from preprocessing.assessment_processor import parse_docx_to_dict

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
        st.error("The feedback tool only supports .docx files. Please convert .doc → .docx and re-upload. Or, upload a HTML file.")

    # Any other file type uploaded
    elif uploaded_ext not in [".docx", ".html"]:
        st.error("The feedback tool only supports .docx or HTML files.")

    # Docx File -> to temp directory
    elif uploaded_ext == ".docx":
        # Save using the same filename (so Path(docx_path).stem matches uploaded name)
        tmp_dir = Path(tempfile.mkdtemp(prefix="upload_"))
        safe_name = Path(uploaded.name).name  # strips any path components
        tmp_path = tmp_dir / safe_name
        tmp_path.write_bytes(uploaded.getbuffer())

        st.success(f"Uploaded: {uploaded.name}")
        st.caption(f"Temp file: {tmp_path}")

    # HTML file -> TBC
    elif uploaded_ext == ".html":
        pass

# Generate Feedback button
run = st.button("Generate Feedback", type="primary", disabled=(tmp_path is None))


# Generate Feedback
if run and tmp_path is not None:

    # Docx file process
    if uploaded_ext == ".docx":
        #convert to python dict
        data = parse_docx_to_dict(str(tmp_path))

        # Download parsed JSON
        st.subheader("Download")
        json_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(label="Download JSON",
                           data=json_bytes,
                           file_name=f"{Path(uploaded.name).stem}.json",
                           mime="application/json",)   
        

# TEMP: testing display options