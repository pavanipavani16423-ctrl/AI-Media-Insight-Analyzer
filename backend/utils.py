"""
Utilities
=========
Helper functions for:
- Extracting text from uploaded files (.txt, .pdf)
- Input validation
- Text preprocessing
"""

import io
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text_from_file(uploaded_file) -> Optional[str]:
    """
    Extract plain text from an uploaded Streamlit file object.

    Supports:
      - .txt files → direct UTF-8 decode
      - .pdf files → PyPDF2 page-by-page extraction

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Extracted text string, or None on failure
    """
    filename = uploaded_file.name.lower()

    # ── TXT files ─────────────────────────────────────────────────────────────
    if filename.endswith(".txt"):
        try:
            raw_bytes = uploaded_file.read()
            # Try UTF-8 first, fall back to latin-1 for legacy files
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = raw_bytes.decode("latin-1")

            logger.info("Extracted %d chars from TXT file", len(text))
            return clean_text(text)

        except Exception as e:
            logger.error("TXT extraction failed: %s", e)
            return None

    # ── PDF files ─────────────────────────────────────────────────────────────
    elif filename.endswith(".pdf"):
        try:
            import PyPDF2

            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            pages = []

            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)

            if not pages:
                logger.warning("PDF appears to be image-based (no extractable text)")
                return None

            full_text = "\n\n".join(pages)
            logger.info(
                "Extracted %d chars from PDF (%d pages)", len(full_text), len(pages)
            )
            return clean_text(full_text)

        except ImportError:
            logger.error("PyPDF2 not installed")
            return "ERROR: PyPDF2 is required for PDF extraction. Run: pip install PyPDF2"
        except Exception as e:
            logger.error("PDF extraction failed: %s", e)
            return None

    else:
        logger.warning("Unsupported file type: %s", filename)
        return None


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.

    Operations:
      - Remove null bytes
      - Normalize whitespace (multiple spaces → single)
      - Normalize newlines (3+ consecutive → 2)
      - Remove control characters (except newlines/tabs)
      - Strip leading/trailing whitespace

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text string
    """
    if not text:
        return ""

    # Remove null bytes and control characters (keep \n and \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize multiple spaces to single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive newlines (3+ → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def validate_api_key(api_key: str) -> bool:
    """
    Basic validation of Google API key format.

    Google API keys typically:
    - Start with "AIza"
    - Are 39 characters long
    - Contain alphanumeric + hyphens + underscores

    Args:
        api_key: API key string to validate

    Returns:
        True if format looks valid
    """
    if not api_key:
        return False

    # Basic format check — real validation only happens on API call
    return (
        api_key.startswith("AIza")
        and len(api_key) >= 35
        and bool(re.match(r"^[A-Za-z0-9_\-]+$", api_key))
    )


def truncate_text(text: str, max_chars: int = 8000) -> str:
    """
    Truncate text to a maximum character count, preserving word boundaries.

    Args:
        text: Input text
        max_chars: Maximum number of characters

    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text

    # Find the last space before the limit to avoid mid-word truncation
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.9:  # only trim if we don't lose too much
        truncated = truncated[:last_space]

    return truncated + "..."


def word_count(text: str) -> int:
    """Return approximate word count of text."""
    return len(text.split()) if text else 0