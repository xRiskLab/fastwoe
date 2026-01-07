---
title: Python Code Style and Formatting Rules
description: Rules for writing Python code with proper formatting and output standards
globs: **/*.py
alwaysApply: true
---

## Code Quality and Formatting Tools

### Code Review with Sourcery
  - Use Sourcery for automated code review and fixes:
    ```bash
    # Review and fix a specific file
    sourcery review --fix path/to/file.py

    # Review and fix all Python files in a directory
    sourcery review --fix chapters/feature_flags/local/scripts/

    # Review without applying fixes (dry run)
    sourcery review path/to/file.py
    ```
  - Sourcery provides suggestions for:
    - Code simplification and refactoring
    - Removing unnecessary else clauses
    - Simplifying conditionals
    - Improving variable naming
    - Reducing complexity
  - Run Sourcery before committing code to catch common issues early.
  - Sourcery suggestions should be reviewed and applied when they improve code readability and maintainability.

## Code Output and Formatting

### Prohibited Patterns

- **DO NOT USE decorative separator lines:**
    - `print("=" * 80)`
    - `print("-" * 50)`
    - Any decorative print statements using repeated characters
- **DO NOT USE empty print statements for spacing:**
    - `print()` used only for adding blank lines in output
- **DO NOT USE bullet point summary statements:**
    - `print("  • AppConfig configuration was updated...")`
    - `print("  - Key finding: ...")`
    - `print("  * Summary: ...")`
    - Any print statements with bullet points or indented summary text
- **DO NOT USE emojis in output (Python scripts only):**
    - `print("✅ Success")`
    - `print("❌ Error")`
    - `print("📊 Step 1: ...")`
    - Any emoji characters in print statements
    - **Exception**: Emojis and colors are OK in Makefiles
- **DO NOT USE leading spaces in print statements:**
    - `print("   Text with leading spaces")`
    - `print(f"    Testing etc")`
    - Use plain text without leading spaces for indentation

### Recommended Patterns

- **DO USE:**
    - Direct, informative print statements without decorative elements
    - Concise output that focuses on actionable information
    - No extra spacing or formatting beyond what's necessary
    - Plain text status indicators (e.g., "Success:", "Error:", "Step 1:")
    - **Makefiles**: Emojis and colors are acceptable, but avoid extra spacing

### Rationale

Cleaner, more professional output that's easier to parse programmatically with less visual clutter.
