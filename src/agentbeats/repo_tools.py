"""Shared tools for repository operations.

These tools are used by both the green agent and participant agents
for reading files, searching, and running commands in repositories.
"""

import os
import subprocess
from loguru import logger


def read_file(repo_dir: str, file_path: str) -> str:
    """Read contents of a file in the repository.
    
    Args:
        repo_dir: Base directory of the repository
        file_path: Path relative to repository root (e.g., 'README.md', 'src/main.py')
    
    Returns:
        The file contents as a string, or an error message if the file cannot be read.
    """
    full_path = os.path.join(repo_dir, file_path)
    # Security: prevent path traversal
    if not os.path.realpath(full_path).startswith(os.path.realpath(repo_dir)):
        return "Error: Invalid file path (path traversal detected)"
    if not os.path.exists(full_path):
        return f"Error: File not found: {file_path}"
    if os.path.isdir(full_path):
        return f"Error: {file_path} is a directory, not a file"
    try:
        with open(full_path, "r") as f:
            content = f.read()
            if len(content) > 50000:
                return content[:50000] + "\n\n... (file truncated, too large)"
            return content
    except Exception as e:
        return f"Error reading file: {e}"


def grep_search(repo_dir: str, pattern: str) -> str:
    """Search for a pattern in repository files using grep.
    
    Args:
        repo_dir: Base directory of the repository
        pattern: Pattern to search for (supports extended regex)
    
    Returns:
        Search results with file paths and line numbers, or error message.
    """
    try:
        result = subprocess.run(
            f'grep -rn -E "{pattern}" . --include="*.py" --include="*.toml" --include="*.cfg" --include="*.txt" --include="*.md" --include="*.rst" 2>/dev/null | head -50 || true',
            shell=True,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout.strip()
        if not output:
            return f"No matches found for pattern: {pattern}"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Search timed out after 60 seconds"
    except Exception as e:
        return f"Error searching: {e}"


def run_command(repo_dir: str, command: str) -> str:
    """Execute a shell command in the repository directory.
    
    Args:
        repo_dir: Base directory of the repository
        command: Shell command to execute
    
    Returns:
        Command output including stdout, stderr, and exit code.
    """
    logger.info(f"Executing command: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout
        )
        output = f"Exit code: {result.returncode}\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        # Limit output size
        if len(output) > 10000:
            output = output[:10000] + "\n\n... (output truncated)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 5 minutes"
    except Exception as e:
        return f"Error executing command: {e}"


# Wrapper functions for ADK FunctionTool (single repo_dir configured at runtime)
class RepoTools:
    """Tools bound to a specific repository directory."""
    
    def __init__(self, repo_dir: str):
        self.repo_dir = repo_dir
    
    def read_file(self, file_path: str) -> str:
        """Read contents of a file in the repository."""
        return read_file(self.repo_dir, file_path)
    
    def grep_search(self, pattern: str) -> str:
        """Search for a pattern in repository files."""
        return grep_search(self.repo_dir, pattern)
    
    def run_command(self, command: str) -> str:
        """Execute a shell command in the repository."""
        return run_command(self.repo_dir, command)
