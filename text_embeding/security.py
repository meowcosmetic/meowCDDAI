"""
Security utilities for safe file handling.
Prevents path traversal attacks when processing uploaded files.
"""

import os
import re
import uuid
import tempfile


def sanitize_filename(filename: str) -> str:
    """
    Remove path traversal sequences from a filename.
    Returns only the basename with dangerous characters stripped.
    """
    if not filename:
        return "unnamed"
    filename = os.path.basename(filename)
    filename = re.sub(r"[^\w.\-]", "_", filename)
    filename = re.sub(r"\.{2,}", ".", filename)
    if not filename or filename.startswith("."):
        filename = "unnamed" + filename
    return filename


def safe_upload_path(original_filename: str, suffix: str = "") -> str:
    """
    Generate a safe temporary file path using uuid4.
    The original filename is sanitized and appended as a label only.
    """
    safe_name = sanitize_filename(original_filename)
    ext = os.path.splitext(safe_name)[1]
    unique_name = f"{uuid.uuid4().hex}_{safe_name}" if not suffix else f"{uuid.uuid4().hex}{suffix}"
    return os.path.join(tempfile.gettempdir(), unique_name)


def safe_temp_path(original_filename: str) -> str:
    """
    Generate a safe path in the system temp directory using uuid4.
    Returns a path that cannot be traversed outside temp.
    """
    safe_name = sanitize_filename(original_filename)
    ext = os.path.splitext(safe_name)[1]
    return os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{ext}")


def validate_filename_alphanumeric(filename: str) -> bool:
    """
    Validate that filename contains only safe alphanumeric characters.
    Returns True if safe, False if it contains traversal sequences.
    """
    if not filename:
        return False
    if ".." in filename or "/" in filename or "\\" in filename:
        return False
    if not re.match(r"^[A-Za-z0-9._\-]+$", filename):
        return False
    return True


def safe_file_path(base_dir: str, filename: str) -> str:
    """
    Safely join base_dir and filename, ensuring the result
    is still within base_dir (prevents path traversal).
    Raises ValueError if the resolved path escapes base_dir.
    """
    base_dir = os.path.realpath(base_dir)
    file_path = os.path.realpath(os.path.join(base_dir, filename))
    if not file_path.startswith(base_dir + os.sep) and file_path != base_dir:
        raise ValueError(f"Path traversal detected: {filename}")
    return file_path
