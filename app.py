import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

from iucn_rules_checker.assessment_parser import AssessmentParser
from iucn_rules_checker.assessment_reviewer import IUCNAssessmentReviewer
from preprocessing.assessment_processor import parse_dict


st.set_page_config(page_title="IUCN Assessment Feedback Tool", layout="wide")
st.title("IUCN Assessment Feedback Tool")
st.caption(
    "Upload one assessment, then run the rules-based and LLM systems separately "
    "from the tabs below."
)


def normalize_display_section_name(section_name: str) -> str:
    """Remove parser block markers from a section label for UI grouping."""
    return re.sub(
        r"\s+\[(?:paragraph|table|row)\s+\d+\]",
        "",
        section_name,
        flags=re.IGNORECASE,
    )

if "uploaded_file_signature" not in st.session_state:
    st.session_state["uploaded_file_signature"] = None
if "rules_feedback" not in st.session_state:
    st.session_state["rules_feedback"] = None
if "llm_feedback" not in st.session_state:
    st.session_state["llm_feedback"] = None

uploaded = st.file_uploader(
    "Upload a .docx, .html or .htm file",
    type=["docx", "html", "doc"],
)

tmp_path: Path | None = None
uploaded_ext: str | None = None
uploaded_name = ""
file_signature = None
input_ready = False

# Handle uploaded file
if uploaded:
    uploaded_name = uploaded.name
    uploaded_ext = Path(uploaded.name).suffix.lower()  # ".docx" or ".html"
    file_signature = f"{uploaded.name}:{len(uploaded.getbuffer())}"

    # Doc file -> raise error
    if uploaded_ext == ".doc":
        st.error(
            "The feedback tool only supports .docx files. Please convert .doc to .docx and re-upload. Or, upload a HTML file."
        )

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

if st.session_state["uploaded_file_signature"] != file_signature:
    st.session_state["uploaded_file_signature"] = file_signature
    st.session_state["rules_feedback"] = None
    st.session_state["llm_feedback"] = None

if uploaded is None:
    st.info("Upload a file to enable both feedback systems.")
elif tmp_path is not None:
    input_ready = True

rules_tab, llm_tab = st.tabs(["Rules-based feedback", "LLM feedback"])

with rules_tab:
    st.subheader("Rules-based feedback")
    st.write("Run the deterministic checker suite on the uploaded assessment.")

    if st.button(
        "Generate feedback",
        key="generate_rules_feedback",
        type="primary",
        disabled=not input_ready,
    ):
        with st.spinner("Generating rules-based feedback..."):
            assessment = parse_dict(str(tmp_path))
            parser = AssessmentParser()
            reviewer = IUCNAssessmentReviewer()
            full_report = parser.parse(assessment)
            violations = [
                violation.to_dict()
                for violation in reviewer.review_full_report(full_report)
            ]

            grouped_violations = defaultdict(list)
            for violation in violations:
                section_name = normalize_display_section_name(
                    violation.get("section_name") or "Whole document"
                )
                grouped_violations[section_name].append(violation)

            ordered_grouped_violations = {}
            seen_sections = set()
            for section_name in full_report:
                normalized_name = normalize_display_section_name(section_name)
                if normalized_name in seen_sections:
                    continue
                seen_sections.add(normalized_name)
                if normalized_name in grouped_violations:
                    ordered_grouped_violations[normalized_name] = grouped_violations[normalized_name]

            for section_name, rows in grouped_violations.items():
                if section_name not in ordered_grouped_violations:
                    ordered_grouped_violations[section_name] = rows

            st.session_state["rules_feedback"] = {
                "assessment": assessment,
                "full_report": full_report,
                "violations": violations,
                "grouped_violations": ordered_grouped_violations,
            }

    if st.session_state["rules_feedback"] is None:
        st.info("Click `Generate feedback` in this tab to run the rules-based reviewer.")
    else:
        feedback = st.session_state["rules_feedback"]
        assessment = feedback["assessment"]
        full_report = feedback["full_report"]
        violations = feedback["violations"]
        grouped_violations = feedback["grouped_violations"]

        download_col_1, download_col_2 = st.columns(2)
        with download_col_1:
            st.download_button(
                label="Download parse_dict output",
                data=json.dumps(assessment, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name=f"{Path(uploaded_name).stem}_parse_dict.json",
                mime="application/json",
            )
        with download_col_2:
            st.download_button(
                label="Download assessment parser output",
                data=json.dumps(full_report, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name=f"{Path(uploaded_name).stem}_assessment_parser.json",
                mime="application/json",
            )

        if not violations:
            st.success("No rules-based violations were found for this document.")
        else:
            st.metric("Violations found", len(violations))

            for section_name, rows in grouped_violations.items():
                table = pd.DataFrame(
                    [
                        {
                            "Rule": row.get("rule_class", ""),
                            "Method": row.get("rule_method", ""),
                            "Matched text": row.get("matched_text", ""),
                            "Context": row.get("matched_snippet", ""),
                            "Message": row.get("message", ""),
                            "Suggested fix": row.get("suggested_fix", ""),
                        }
                        for row in rows
                    ]
                )
                table = table[
                    [
                        "Rule",
                        "Method",
                        "Matched text",
                        "Context",
                        "Message",
                        "Suggested fix",
                    ]
                ]

                error_label = "error" if len(rows) == 1 else "errors"
                with st.expander(f"{section_name} ({len(rows)} {error_label})", expanded=False):
                    st.dataframe(table, use_container_width=True, hide_index=True)

            st.download_button(
                label="Download rules feedback (JSON)",
                data=json.dumps(grouped_violations, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name=f"{Path(uploaded_name).stem}_rules_feedback.json",
                mime="application/json",
            )

with llm_tab:
    st.subheader("LLM feedback")
    st.write("Run the language-model review separately from the rules-based checks.")

    if st.button(
        "Generate feedback",
        key="generate_llm_feedback",
        type="primary",
        disabled=not input_ready,
    ):
        with st.spinner("Generating LLM feedback..."):
            assessment = parse_dict(str(tmp_path))
            st.session_state["llm_feedback"] = {
                "status": "placeholder",
                "message": (
                    "The LLM tab is wired up, but the model feedback pipeline is still "
                    "a placeholder. Replace the inline block in `app.py` with your "
                    "Ollama or LangChain call to render real output here."
                ),
                "top_level_sections": [
                    child.get("title")
                    for child in assessment.get("children", [])
                    if child.get("title")
                ],
            }

    if st.session_state["llm_feedback"] is None:
        st.info("Click `Generate feedback` in this tab to run the LLM workflow.")
    else:
        feedback = st.session_state["llm_feedback"]
        if feedback.get("status") == "placeholder":
            st.warning(feedback["message"])
            if feedback.get("top_level_sections"):
                st.caption(
                    "Detected top-level sections: "
                    + ", ".join(feedback["top_level_sections"])
                )
        else:
            st.write(feedback.get("message", "LLM feedback generated."))
