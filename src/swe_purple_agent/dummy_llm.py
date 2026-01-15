import argparse
import os
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from loguru import logger

# Configure loguru for detailed output
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True
)

from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from a2a.types import (
    AgentCapabilities,
    AgentCard,
)

# Import shared tools
from agentbeats.repo_tools import read_file as _read_file, grep_search as _grep_search, run_command as _run_command

INSTRUCTION = """You will be provided with a partial code base and an issue statement explaining a problem to resolve.

I need you to solve this issue by generating a single patch file that I can apply directly to this repository using git apply. Please respond with a single patch file in the following format.
<patch>
--- a/file.py
+++ b/file.py
@@ -1,27 +1,35 @@
def euclidean(a, b):
- while b:
- a, b = b, a % b
- return a
+ if b == 0:
+ return a
+ return euclidean(b, a % b)


def bresenham(x0, y0, x1, y1):
 points = []
 dx = abs(x1 - x0)
 dy = abs(y1 - y0)
- sx = 1 if x0 < x1 else -1
- sy = 1 if y0 < y1 else -1
- err = dx - dy
+ x, y = x0, y0
+ sx = -1 if x0 > x1 else 1
+ sy = -1 if y0 > y1 else 1

- while True:
- points.append((x0, y0))
- if x0 == x1 and y0 == y1:
- break
- e2 = 2 * err
- if e2 > -dy:
+ if dx > dy:
+ err = dx / 2.0
+ while x != x1:
+ points.append((x, y))
err -= dy
- x0 += sx
- if e2 < dx:
- err += dx
- y0 += sy
+ if err < 0:
+ y += sy
+ err += dx
+ x += sx
+ else:
+ err = dy / 2.0
+ while y != y1:
+ points.append((x, y))
+ err -= dx
+ if err < 0:
+ x += sx
+ err += dy
+ y += sy

+ points.append((x, y))
return points
</patch>

## AVAILABLE TOOLS

You have the following tools available to help you understand the codebase and solve the issue:

### 1. read_file(file_path: str)
Read the contents of a file in the repository.
- **file_path**: Path RELATIVE to the repository root (e.g., 'README.md', 'src/main.py', 'tests/test_utils.py')
- Example: To read the file at /repo/src/utils.py, use: read_file(file_path="src/utils.py")

### 2. grep_search(pattern: str)
Search for a pattern across all files in the repository using grep.
- **pattern**: The search pattern (supports extended regex)
- Example: To find all usages of a function: grep_search(pattern="my_function")
- Example: To find class definitions: grep_search(pattern="class MyClass")

### 3. run_command(command: str)
Execute a shell command in the repository directory.
- **command**: The shell command to execute
- Example: To list files: run_command(command="ls -la")
- Example: To run tests: run_command(command="python -m pytest tests/")

## WORKFLOW

1. First, understand the issue by reading relevant files
2. Use grep_search to find related code patterns
3. Use read_file to examine specific files in detail
4. Generate a patch file that fixes the issue

## IMPORTANT NOTES

- The first line of the message contains REPO_DIR:/path/to/repo - this is handled automatically by the tools
- Always use RELATIVE paths with read_file (e.g., 'src/main.py' NOT '/full/path/to/repo/src/main.py')
- When you have enough information, generate the patch inside <patch>...</patch> tags
"""


def _parse_repo_dir_from_context(context) -> str:
    """Extract REPO_DIR from the conversation context."""
    logger.debug("Parsing REPO_DIR from context...")
    logger.debug(f"Context type: {type(context)}")
    
    # Try to get repo dir from the events in the session
    if hasattr(context, 'session'):
        session = context.session
        logger.debug(f"Session found: {type(session)}")
        # Check for events (ADK session structure)
        events = getattr(session, 'events', None) or getattr(session, 'history', None) or []
        logger.debug(f"Found {len(events)} events in session")
        for i, event in enumerate(events):
            content = getattr(event, 'content', None)
            if content:
                parts = getattr(content, 'parts', None)
                if parts:
                    for part in parts:
                        text = getattr(part, 'text', None)
                        if text and "REPO_DIR:" in text:
                            # Find the line containing REPO_DIR
                            for line in text.split('\n'):
                                if line.startswith("REPO_DIR:"):
                                    repo_dir = line.replace("REPO_DIR:", "").strip()
                                    logger.info(f"Found REPO_DIR: {repo_dir}")
                                    return repo_dir
    else:
        # Content is a plain string
        logger.debug("No session found, trying to parse context as string")
        text = str(context)
        if "REPO_DIR:" in text:
            for line in text.split('\n'):
                if line.startswith("REPO_DIR:"):
                    repo_dir = line.replace("REPO_DIR:", "").strip()
                    logger.info(f"Found REPO_DIR: {repo_dir}")
                    return repo_dir
    
    logger.error("REPO_DIR not found in message context")
    raise ValueError("REPO_DIR not found in message context. Expected format: REPO_DIR:/path/to/repo")


# Wrapper functions that use REPO_DIR (required by ADK FunctionTool)
def read_file(file_path: str, tool_context=None) -> str:
    """Read contents of a file in the repository.
    
    Args:
        file_path: Path relative to repository root (e.g., 'README.md', 'src/main.py')
        tool_context: ADK tool context (automatically passed)
    
    Returns:
        The file contents as a string, or an error message if the file cannot be read.
    """
    logger.info(f"[TOOL CALL] read_file(file_path='{file_path}')")
    try:
        repo_dir = _parse_repo_dir_from_context(tool_context)
        logger.debug(f"  -> Using REPO_DIR: {repo_dir}")
        result = _read_file(repo_dir, file_path)
        logger.debug(f"  -> Result length: {len(result)} chars")
        logger.info(f"[TOOL RESULT] read_file completed successfully")
        return result
    except Exception as e:
        logger.error(f"[TOOL ERROR] read_file failed: {e}")
        raise


def grep_search(pattern: str, tool_context=None) -> str:
    """Search for a pattern in repository files using grep.
    
    Args:
        pattern: Pattern to search for (supports extended regex)
        tool_context: ADK tool context (automatically passed)
    
    Returns:
        Search results with file paths and line numbers, or error message.
    """
    logger.info(f"[TOOL CALL] grep_search(pattern='{pattern}')")
    try:
        repo_dir = _parse_repo_dir_from_context(tool_context)
        logger.debug(f"  -> Using REPO_DIR: {repo_dir}")
        result = _grep_search(repo_dir, pattern)
        logger.debug(f"  -> Result length: {len(result)} chars")
        logger.info(f"[TOOL RESULT] grep_search completed successfully")
        return result
    except Exception as e:
        logger.error(f"[TOOL ERROR] grep_search failed: {e}")
        raise


def run_command(command: str, tool_context=None) -> str:
    """Execute a shell command in the repository directory.
    
    Args:
        command: Shell command to execute
        tool_context: ADK tool context (automatically passed)
    
    Returns:
        Command output including stdout, stderr, and exit code.
    """
    logger.info(f"[TOOL CALL] run_command(command='{command}')")
    try:
        repo_dir = _parse_repo_dir_from_context(tool_context)
        logger.debug(f"  -> Using REPO_DIR: {repo_dir}")
        result = _run_command(repo_dir, command)
        logger.debug(f"  -> Result length: {len(result)} chars")
        logger.info(f"[TOOL RESULT] run_command completed successfully")
        return result
    except Exception as e:
        logger.error(f"[TOOL ERROR] run_command failed: {e}")
        raise



def main():
    parser = argparse.ArgumentParser(description="Run the Dummy LLM agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--model", type=str, help="LLM model to use", default="gemini/gemini-2.5-flash")
    args = parser.parse_args()
    
    ollama_api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    
    if "ollama" in args.model:
        model = LiteLlm(
            model=args.model,
            api_base=ollama_api_base
        )
    else:
        model = LiteLlm(
            model=args.model,
        )

    # Create tool instances
    read_file_tool = FunctionTool(read_file)
    grep_search_tool = FunctionTool(grep_search)
    run_command_tool = FunctionTool(run_command)

    root_agent = Agent(
        name="dummy_llm",
        model=model,
        description="A dummy agent that solves issues by generating patch files.",
        instruction=INSTRUCTION,
        tools=[read_file_tool, grep_search_tool, run_command_tool],
    )

    agent_card = AgentCard(
        name="dummy_llm",
        description='A dummy agent that replies.',
        url=args.card_url or f'http://{args.host}:{args.port}/',
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
    )

    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
