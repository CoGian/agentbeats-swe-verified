import argparse
import os
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from a2a.types import (
    AgentCapabilities,
    AgentCard,
)

# Repository directory - will be set when provided
REPO_DIR = os.getenv("REPO_DIR", "/tmp/repo")

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

You have access to tools to read files and run commands in the repository to help you understand the codebase and solve the issue.
"""


def read_file(file_path: str) -> str:
    """Read contents of a file in the repository.
    
    Args:
        file_path: Path relative to repository root (e.g., 'README.md', 'src/main.py')
    
    Returns:
        The file contents as a string, or an error message if the file cannot be read.
    """
    full_path = os.path.join(REPO_DIR, file_path)
    # Security: prevent path traversal
    if not os.path.realpath(full_path).startswith(os.path.realpath(REPO_DIR)):
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



def main():
    parser = argparse.ArgumentParser(description="Run the Dummy LLM agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9021, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--repo-dir", type=str, help="Repository directory for file operations")
    args = parser.parse_args()

    # Set repo directory if provided
    global REPO_DIR
    if args.repo_dir:
        REPO_DIR = args.repo_dir
    
    ollama_api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    ollama_model = os.getenv("DUMMY_OLLAMA_MODEL", "qwen2.5-coder:7b")
    
    model = LiteLlm(
        model=f"ollama_chat/{ollama_model}",
        api_base=ollama_api_base
    )

    # Create tool instance
    read_file_tool = FunctionTool(read_file)

    root_agent = Agent(
        name="dummy_llm",
        model=model,
        description="A dummy agent that solves issues by generating patch files.",
        instruction=INSTRUCTION,
        tools=[read_file_tool],
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
