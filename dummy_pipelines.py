from pathlib import Path

def llm_pipeline(input_path: Path) -> dict:
    return {
        "pipeline": "llm",
        "summary": "dummy llm output",
        "section_feedback": [],
        "source_path": str(input_path),
    }

def formatting_pipeline(input_path: Path) -> dict:
    return {
        "pipeline": "formatting",
        "summary": "dummy formatting output",
        "format_issues": [],
        "source_path": str(input_path),
    }

def build_unified_document(
    input_path: Path,
    llm_result: dict,
    fmt_result: dict,
    output_path: Path
) -> Path:
    # Dummy: just copy original doc. Later this will generate a Word doc using llm_result + fmt_result.
    output_path.write_bytes(input_path.read_bytes())
    return output_path
