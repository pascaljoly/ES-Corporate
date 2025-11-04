#!/usr/bin/env python3
"""
Security utilities for path sanitization and input validation.
"""

import re
from pathlib import Path
from typing import Optional


# Security limits
MAX_PATH_COMPONENT_LENGTH = 255
MAX_JSON_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_FILES_TO_PROCESS = 1000
MAX_STRING_INPUT_LENGTH = 255

# Unsafe characters for file paths
UNSAFE_PATH_CHARS = ['<', '>', ':', '"', '|', '?', '*', '\x00']
PATH_TRAVERSAL_PATTERNS = ['..', './', '../', '~/', '//']


def sanitize_path_component(component: str, max_length: int = MAX_PATH_COMPONENT_LENGTH) -> str:
    """
    Sanitize a path component to prevent directory traversal attacks.
    
    Args:
        component: Path component string to sanitize
        max_length: Maximum allowed length (default: 255)
        
    Returns:
        Sanitized path component
        
    Raises:
        ValueError: If component is empty after sanitization
    """
    if not isinstance(component, str):
        raise TypeError(f"Path component must be a string, got {type(component).__name__}")
    
    # Remove path traversal sequences
    sanitized = component
    for pattern in PATH_TRAVERSAL_PATTERNS:
        sanitized = sanitized.replace(pattern, '')
    
    # Remove all forward slashes and backslashes (path separators)
    sanitized = sanitized.replace('/', '').replace('\\', '')
    
    # Remove unsafe characters
    for char in UNSAFE_PATH_CHARS:
        sanitized = sanitized.replace(char, '')
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Limit length
    sanitized = sanitized[:max_length]
    
    # Validate result
    if not sanitized:
        raise ValueError(f"Path component '{component}' becomes empty after sanitization")
    
    # Check for remaining dangerous patterns
    if re.search(r'\.{2,}', sanitized):
        raise ValueError(f"Path component '{component}' contains dangerous patterns")
    
    return sanitized


def validate_input_length(value: str, field_name: str, max_length: int = MAX_STRING_INPUT_LENGTH) -> None:
    """
    Validate input string length.
    
    Args:
        value: String to validate
        field_name: Name of the field (for error messages)
        max_length: Maximum allowed length
        
    Raises:
        ValueError: If value exceeds max_length
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    
    if len(value) > max_length:
        raise ValueError(
            f"{field_name} must be <= {max_length} characters, got {len(value)}"
        )
    
    if len(value) == 0:
        raise ValueError(f"{field_name} cannot be empty")


def validate_file_path(filepath: Path, base_dir: Optional[Path] = None) -> Path:
    """
    Validate that a file path is safe and within allowed directory.
    
    Args:
        filepath: Path to validate
        base_dir: Base directory that path must be within (optional)
        
    Returns:
        Resolved, validated Path object
        
    Raises:
        ValueError: If path is unsafe or outside base_dir
    """
    try:
        resolved = filepath.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Cannot resolve path {filepath}: {e}")
    
    # Check if path is within base directory
    if base_dir:
        try:
            base_resolved = base_dir.resolve()
            if not str(resolved).startswith(str(base_resolved)):
                raise ValueError(
                    f"Path {filepath} is outside allowed directory {base_dir}"
                )
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Cannot resolve base directory {base_dir}: {e}")
    
    return resolved


def get_file_size(filepath: Path) -> int:
    """
    Get file size in bytes.
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in bytes
        
    Raises:
        FileNotFoundError: If file doesn't exist
        OSError: If file cannot be accessed
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    return filepath.stat().st_size


def validate_json_file_size(filepath: Path, max_size: int = MAX_JSON_FILE_SIZE) -> None:
    """
    Validate that a JSON file size is within limits.
    
    Args:
        filepath: Path to JSON file
        max_size: Maximum allowed size in bytes (default: 10MB)
        
    Raises:
        ValueError: If file size exceeds max_size
        FileNotFoundError: If file doesn't exist
    """
    file_size = get_file_size(filepath)
    
    if file_size > max_size:
        raise ValueError(
            f"JSON file too large: {file_size} bytes (max: {max_size} bytes / {max_size // (1024*1024)}MB)"
        )


def sanitize_and_validate_path(
    base_dir: str,
    *components: str,
    create: bool = False
) -> Path:
    """
    Sanitize path components and create a safe Path object.
    
    Args:
        base_dir: Base directory path (can be absolute or relative)
        *components: Path components to sanitize and join
        create: If True, create parent directories
        
    Returns:
        Safe Path object
        
    Raises:
        ValueError: If any component is invalid or path is unsafe
    """
    # For base_dir, allow slashes if it's a proper path but sanitize the string itself
    # Base directory might be a full path like "/tmp/test" or "results"
    # We'll use it as-is if it's a valid Path, but sanitize any components
    base_path = Path(base_dir) if base_dir else Path(".")
    
    # Sanitize and join components
    sanitized_components = [sanitize_path_component(comp) for comp in components]
    
    # Build path
    result_path = base_path
    for component in sanitized_components:
        result_path = result_path / component
    
    # Resolve to absolute path to check for traversal
    try:
        resolved = result_path.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Cannot create safe path: {e}")
    
    # Create directories if requested
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    
    return resolved

