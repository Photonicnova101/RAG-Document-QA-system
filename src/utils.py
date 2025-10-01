"""
Utility functions for the Personal Knowledge Assistant
"""

import logging
import sys
from pathlib import Path

# ANSI color codes
COLORS = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'cyan': '\033[96m',
    'gray': '\033[90m',
    'reset': '\033[0m'
}

def print_colored(text: str, color: str = 'reset'):
    """
    Print colored text to console
    
    Args:
        text: Text to print
        color: Color name
    """
    color_code = COLORS.get(color, COLORS['reset'])
    print(f"{color_code}{text}{COLORS['reset']}")

def setup_logging(log_file: str = "qa_system.log", level=logging.INFO):
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Input text
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def validate_file_path(file_path: str) -> bool:
    """
    Validate if file path exists and is accessible
    
    Args:
        file_path: Path to validate
        
    Returns:
        True if valid
    """
    path = Path(file_path)
    return path.exists() and path.is_file()

def validate_directory_path(dir_path: str) -> bool:
    """
    Validate if directory path exists and is accessible
    
    Args:
        dir_path: Directory path to validate
        
    Returns:
        True if valid
    """
    path = Path(dir_path)
    return path.exists() and path.is_dir()
