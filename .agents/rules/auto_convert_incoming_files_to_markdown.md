# Rule: Automatic Internal Document & Data Conversion to Markdown

1. **Automatic Incoming File & Data Ingestion:**
   - Whenever a user shares raw data or attaches files in chat (including `.csv`, `.tsv`, `.xlsx`, `.xls`, `.docx`, `.pdf`, `.json`, `.yaml`, images, etc.), the agent must **automatically detect and convert it internally** into token-optimized Markdown (`.md`).
   - The user does **not** need to repeatedly prompt or request file conversion; handle all conversions proactively in the background using the internal converter module (`file_to_markdown_converter/convert_to_markdown.py`).
   - No external or Streamlit UI is required — execution is entirely internal and automated.

2. **Token Optimization & Representation:**
   - Always produce clean, token-efficient Markdown with:
     - High-level metadata & schema summary (column types, null percentages, cardinality).
     - Compact Markdown table preview (e.g. first 10–50 rows).
     - Elimination of redundant whitespace and repeated blank lines.

3. **Strict Git & Privacy Protection:**
   - All converter tools, uploaded input files, and output Markdown documents in `file_to_markdown_converter/` and `converted_markdown/` MUST remain in `.gitignore`.
   - Never commit or push converter utilities or converted client data files to GitHub.
