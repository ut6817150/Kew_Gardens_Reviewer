import streamlit as st
import tempfile
import concurrent.futures
from pathlib import Path

from dummy_pipelines import llm_pipeline, formatting_pipeline, build_unified_document

st.set_page_config(page_title="Assessment Feedback Tool", layout="centered")
st.title("Assessment Feedback Tool")

uploaded = st.file_uploader("Upload a .doc file", type=["doc"])

tmp_path: Path | None = None
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    st.success(f"Uploaded: {uploaded.name}")
    st.caption(f"Temp file: {tmp_path}")

run = st.button("Generate Feedback", type="primary", disabled=(tmp_path is None))

if run and tmp_path is not None:
    with st.spinner("Running pipelines in parallel..."):
        # Run both pipelines concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_llm = ex.submit(llm_pipeline, tmp_path)
            f_fmt = ex.submit(formatting_pipeline, tmp_path)

            llm_result = f_llm.result()
            fmt_result = f_fmt.result()

        # Combine outputs into ONE unified Word document
        unified_out = tmp_path.with_name(f"{tmp_path.stem}_unified{tmp_path.suffix}")
        unified_path = build_unified_document(
            input_path=tmp_path,
            llm_result=llm_result,
            fmt_result=fmt_result,
            output_path=unified_out,
        )

    st.success("Done! Unified document created.")

    st.download_button(
        label="Download unified output (.doc)",
        data=unified_path.read_bytes(),
        file_name=f"unified_{uploaded.name}",
        mime="application/msword",
    )
