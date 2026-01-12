import argparse
import asyncio
import contextlib
import logging
import os
import shutil
import subprocess
import tempfile
import json
from typing import Any

import pandas as pd
import litellm

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCard, TaskState, Part, TextPart, AgentCapabilities, AgentSkill
from a2a.utils import new_agent_text_message
from dotenv import load_dotenv

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider
from agentbeats.cloudflare import quick_tunnel
from loguru import logger

load_dotenv()

logging.basicConfig(level=logging.INFO)


# Tool definitions for LLM function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file in the repository. Use for: pyproject.toml, setup.py, README.md, CONTRIBUTING.md, installation guides, docs/",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path relative to repository root (e.g., 'README.md', 'docs/install.md')"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command in the repository directory. Use for: creating venv, pip install, running setup scripts",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute (e.g., 'python3.11 -m venv .venv', 'pip install -e .')"
                    }
                },
                "required": ["command"]
            }
        }
    }
]

# System prompt for Phase 1: Environment Setup (Python version + venv)
ENV_SETUP_PROMPT = """You are an expert at setting up Python environments.

## YOUR TASK: Create a virtual environment with the correct Python version.

## Steps:
1. Read pyproject.toml or setup.py to find the required Python version
2. Install that Python version using: `uv python install X.Y`
3. Create a venv using: `uv venv --python X.Y .venv`

## Available Tools:
- read_file: Read file contents
- run_command: Execute shell commands

## CRITICAL Rules:
- Only read pyproject.toml, setup.py, or setup.cfg - no other files
- If no Python version specified, use python3.11
- DO NOT run `uv venv` more than once - if you see "Creating virtual environment at: .venv" in output, the venv is ALREADY CREATED
- After venv is created successfully, you MUST stop immediately and respond with ONLY the text "ENV READY" - do not call any more tools

## Repository Files:
{file_listing}
"""

# System prompt for Phase 2: Dependency Installation
DEPS_INSTALL_PROMPT = """You are an expert at installing Python dependencies.

## YOUR TASK: Install all dependencies for this repository.

## Steps:
1. Read README.md or INSTALL.md to understand installation steps
2. Install using the venv's pip directly: `.venv/bin/pip install -e .`
3. If there are additional requirements, install them too

## Available Tools:
- read_file: Read file contents
- run_command: Execute shell commands

## CRITICAL Rules:
- The venv is already created at .venv
- ALWAYS use the venv's pip directly: `.venv/bin/pip install ...` (NOT `pip install`)
- ALWAYS use the venv's python directly: `.venv/bin/python ...` (NOT `python`)
- Do NOT use `. .venv/bin/activate` - it doesn't work in subprocesses
- Do NOT read CONTRIBUTING.md
- When installation is complete, stop and respond "DEPS INSTALLED"

## Repository Files:
{file_listing}
"""


class SweVerifiedGreenAgent(GreenAgent):
    def __init__(self):
        self._required_config_keys = ["base_commit", "hints_text", "problem_statement", "repo_url"]
        self._tool_provider = ToolProvider()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        
        if not request.participants:
             return False, "Missing participants"

        return True, "ok"

    def _read_file(self, repo_dir: str, file_path: str) -> str:
        """Read file contents safely within repo directory."""
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

    def _run_command(self, repo_dir: str, command: str) -> str:
        """Execute command and return output."""
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

    async def _run_agentic_loop(self, repo_dir: str, system_prompt: str, user_message: str, 
                                  phase_name: str, updater: TaskUpdater, max_iterations: int = 10) -> None:
        """Run a focused agentic loop with the given prompt."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Track completion state to detect when we should force-exit
        venv_created = False
        deps_installed = False
        
        for iteration in range(max_iterations):
            logger.info(f"[{phase_name}] === Iteration {iteration + 1}/{max_iterations} ===")
            await updater.update_status(
                TaskState.working, 
                new_agent_text_message(f"{phase_name} iteration {iteration + 1}...")
            )
            
            try:
                response = await litellm.acompletion(
                    model="ollama/qwen2.5-coder:7b",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto"
                )
            except Exception as e:
                logger.error(f"[{phase_name}] LLM call failed: {e}")
                raise e
            
            assistant_message = response.choices[0].message
            messages.append(assistant_message.model_dump())
            
            # Check if there are tool calls
            if not assistant_message.tool_calls:
                logger.info(f"[{phase_name}] ✓ Completed. Final message: {assistant_message.content}")
                break
            
            # Check if we should force-exit based on completion state
            if phase_name == "ENV_SETUP" and venv_created:
                logger.info(f"[{phase_name}] ✓ Venv already created, forcing completion")
                break
            if phase_name == "DEPS_INSTALL" and deps_installed:
                logger.info(f"[{phase_name}] ✓ Dependencies already installed, forcing completion")
                break

            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                
                if func_name == "read_file":
                    file_path = args.get("file_path", "")
                    logger.info(f"[{phase_name}] → Reading file: {file_path}")
                    result = self._read_file(repo_dir, file_path)
                    await updater.update_status(
                        TaskState.working, 
                        new_agent_text_message(f"Reading: {file_path}")
                    )
                    logger.info(f"[{phase_name}]   File read: {len(result)} chars")
                    
                elif func_name == "run_command":
                    command = args.get("command", "")
                    logger.info(f"[{phase_name}] → Running command: {command}")
                    result = self._run_command(repo_dir, command)
                    
                    # Extract exit code for logging
                    exit_code = "unknown"
                    if "Exit code: " in result:
                        exit_code = result.split("Exit code: ")[1].split("\n")[0]
                    
                    await updater.update_status(
                        TaskState.working, 
                        new_agent_text_message(f"Running: {command[:50]}...")
                    )
                    logger.info(f"[{phase_name}]   Command exit code: {exit_code}")
                    
                    # Detect completion states
                    if "uv venv" in command and "Creating virtual environment at: .venv" in result:
                        venv_created = True
                        logger.info(f"[{phase_name}] ✓ Detected: venv successfully created")
                    if "pip install" in command and exit_code == "0":
                        deps_installed = True
                        logger.info(f"[{phase_name}] ✓ Detected: pip install succeeded")
                else:
                    result = f"Unknown tool: {func_name}"
                    logger.warning(f"[{phase_name}] Unknown tool: {func_name}")
                
                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            logger.warning(f"[{phase_name}] ⚠ Reached max iterations ({max_iterations})")

    async def _run_env_setup(self, repo_dir: str, file_listing: str, updater: TaskUpdater) -> None:
        """Phase 1: Identify Python version, install it, create venv."""
        system_prompt = ENV_SETUP_PROMPT.format(file_listing=file_listing)
        await self._run_agentic_loop(
            repo_dir=repo_dir,
            system_prompt=system_prompt,
            user_message="Find the required Python version from config files, install it with uv, and create a .venv",
            phase_name="ENV_SETUP",
            updater=updater,
            max_iterations=8
        )

    async def _run_deps_install(self, repo_dir: str, file_listing: str, updater: TaskUpdater) -> None:
        """Phase 2: Read docs and install dependencies."""
        system_prompt = DEPS_INSTALL_PROMPT.format(file_listing=file_listing)
        await self._run_agentic_loop(
            repo_dir=repo_dir,
            system_prompt=system_prompt,
            user_message="Read the installation docs and install all dependencies using pip install -e . (activate venv first)",
            phase_name="DEPS_INSTALL",
            updater=updater,
            max_iterations=8
        )

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"Starting SWE verified agent evaluation: {req}")
        
        # Load data from parquet
        try:
            df = pd.read_parquet("data/test-00000-of-00001.parquet")
            row = df.iloc[0]
            
            repo_url = row.get("repo", row.get("repo_url"))
            if repo_url and not repo_url.startswith("http"):
                repo_url = f"https://github.com/{repo_url}"
                
            base_commit = row["base_commit"]
            hints_text = row["hints_text"]
            problem_statement = row["problem_statement"]
            fail_to_pass = json.loads(row["FAIL_TO_PASS"]) if isinstance(row["FAIL_TO_PASS"], str) else row["FAIL_TO_PASS"]
            pass_to_pass = json.loads(row["PASS_TO_PASS"]) if isinstance(row["PASS_TO_PASS"], str) else row["PASS_TO_PASS"]
            
        except Exception as e:
            logger.error(f"Failed to read from parquet: {e}")
            await updater.failed(new_agent_text_message(f"Failed to read data: {e}"))
            return

        participant_role = next(iter(req.participants))
        participant_url = str(req.participants[participant_role])

        temp_dir = tempfile.mkdtemp(prefix="agentbeats_repo_")
        try:
            # =============================================
            # PHASE 1: Fixed commands (always executed)
            # =============================================
            await updater.update_status(TaskState.working, new_agent_text_message(f"Cloning repository {repo_url}..."))
            
            # Git clone
            clone_result = subprocess.run(
                ["git", "clone", repo_url, temp_dir],
                capture_output=True,
                text=True
            )
            if clone_result.returncode != 0:
                raise Exception(f"Git clone failed: {clone_result.stderr}")
            logger.info(f"Cloned repository to {temp_dir}")
            
            # Git checkout base commit
            await updater.update_status(TaskState.working, new_agent_text_message(f"Checking out commit {base_commit}..."))
            checkout_result = subprocess.run(
                ["git", "checkout", base_commit],
                cwd=temp_dir,
                capture_output=True,
                text=True
            )
            if checkout_result.returncode != 0:
                raise Exception(f"Git checkout failed: {checkout_result.stderr}")
            logger.info(f"Checked out base commit: {base_commit}")
            
            # List files (for agent context)
            await updater.update_status(TaskState.working, new_agent_text_message("Listing repository files..."))
            ls_result = subprocess.run(
                ["ls", "-la"],
                cwd=temp_dir,
                capture_output=True,
                text=True
            )
            file_listing = ls_result.stdout
            logger.info(f"Repository files:\n{file_listing}")
            
            # =============================================
            # PHASE 2A: Environment Setup (Python + venv)
            # =============================================
            await updater.update_status(TaskState.working, new_agent_text_message("Phase 1: Setting up Python environment..."))
            await self._run_env_setup(temp_dir, file_listing, updater)
            
            # =============================================
            # PHASE 2B: Dependency Installation
            # =============================================
            await updater.update_status(TaskState.working, new_agent_text_message("Phase 2: Installing dependencies..."))
            await self._run_deps_install(temp_dir, file_listing, updater)
            
            # =============================================
            # PHASE 3: Send problem to participant agent
            # =============================================
            await updater.update_status(TaskState.working, new_agent_text_message("Sending problem to agent..."))
            
            # Send problem statement and hints
            message = f"Problem Statement:\n{problem_statement}\n\nHints:\n{hints_text}"
            
            response = await self._tool_provider.talk_to_agent(message, participant_url, new_conversation=True)
            
            logger.info(f"Agent response: {response}")
            await updater.update_status(TaskState.working, new_agent_text_message("Received patch from agent. Applying..."))
            
            # =============================================
            # PHASE 4: Extract and apply patch
            # =============================================
            patch_content = self._extract_patch(response)
            patch_applied = False
            
            if patch_content:
                # Save patch to file
                patch_file = os.path.join(temp_dir, "proposed.patch")
                with open(patch_file, "w") as f:
                    f.write(patch_content)
                
                # Apply patch using unix patch command
                await updater.update_status(TaskState.working, new_agent_text_message("Applying patch with 'patch' command..."))
                patch_result = subprocess.run(
                    ["patch", "-p1", "-i", "proposed.patch"],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True
                )
                
                if patch_result.returncode == 0:
                    patch_applied = True
                    logger.info(f"Patch applied successfully:\n{patch_result.stdout}")
                else:
                    logger.error(f"Patch failed to apply:\n{patch_result.stderr}")
                    await updater.update_status(TaskState.working, new_agent_text_message(f"Patch failed to apply: {patch_result.stderr[:200]}"))
            else:
                logger.warning("No patch found in agent response")
                await updater.update_status(TaskState.working, new_agent_text_message("No patch found in agent response"))

            # =============================================
            # PHASE 5: Run tests with venv activated
            # =============================================
            await updater.update_status(TaskState.working, new_agent_text_message("Running tests..."))
            
            # Get test commands from LLM
            files = os.listdir(temp_dir)
            files_str = "\n".join(files[:500])
            test_prompt = f"""
You are an expert software engineer.
Given the following file list and test identifiers, provide the shell command(s) to run these specific tests.
The virtual environment is at .venv, so activate it first.
Return ONLY a JSON list of strings.

Files:
{files_str}

Tests to run:
FAIL_TO_PASS: {fail_to_pass}
PASS_TO_PASS: {pass_to_pass}
"""
            test_commands_json = await self._ask_llm(test_prompt)
            try:
                test_commands = json.loads(test_commands_json)
                if isinstance(test_commands, str):
                    test_commands = [test_commands]
            except json.JSONDecodeError:
                test_commands = [line.strip() for line in test_commands_json.split('\n') if line.strip() and not line.strip().startswith('```')]

            # Run tests and collect results
            fail_to_pass_results = {}
            pass_to_pass_results = {}
            
            # Run FAIL_TO_PASS tests (these should now PASS after the fix)
            for test in fail_to_pass:
                await updater.update_status(TaskState.working, new_agent_text_message(f"Running FAIL_TO_PASS test: {test[:50]}..."))
                cmd = f".venv/bin/pytest {test} -v"
                result = subprocess.run(cmd, shell=True, cwd=temp_dir, capture_output=True, text=True)
                fail_to_pass_results[test] = result.returncode == 0
                logger.info(f"FAIL_TO_PASS test '{test}': {'PASSED' if result.returncode == 0 else 'FAILED'}")
            
            # Run PASS_TO_PASS tests (these should still PASS after the fix)
            for test in pass_to_pass:
                await updater.update_status(TaskState.working, new_agent_text_message(f"Running PASS_TO_PASS test: {test[:50]}..."))
                cmd = f".venv/bin/pytest {test} -v"
                result = subprocess.run(cmd, shell=True, cwd=temp_dir, capture_output=True, text=True)
                pass_to_pass_results[test] = result.returncode == 0
                logger.info(f"PASS_TO_PASS test '{test}': {'PASSED' if result.returncode == 0 else 'FAILED'}")

            # =============================================
            # PHASE 6: Calculate metrics and report
            # =============================================
            fail_to_pass_passed = sum(1 for v in fail_to_pass_results.values() if v)
            fail_to_pass_total = len(fail_to_pass_results)
            fail_to_pass_pct = (fail_to_pass_passed / fail_to_pass_total * 100) if fail_to_pass_total > 0 else 0
            
            pass_to_pass_passed = sum(1 for v in pass_to_pass_results.values() if v)
            pass_to_pass_total = len(pass_to_pass_results)
            pass_to_pass_pct = (pass_to_pass_passed / pass_to_pass_total * 100) if pass_to_pass_total > 0 else 0
            
            # Resolution: patch applies + all fail_to_pass now pass + all pass_to_pass still pass
            resolved = patch_applied and fail_to_pass_passed == fail_to_pass_total and pass_to_pass_passed == pass_to_pass_total
            
            metrics_summary = f"""
=== EVALUATION RESULTS ===
Patch Applied: {'YES' if patch_applied else 'NO'}

FAIL_TO_PASS Tests (should now PASS):
  Passed: {fail_to_pass_passed}/{fail_to_pass_total} ({fail_to_pass_pct:.1f}%)

PASS_TO_PASS Tests (should still PASS):
  Passed: {pass_to_pass_passed}/{pass_to_pass_total} ({pass_to_pass_pct:.1f}%)

RESOLVED: {'YES' if resolved else 'NO'}
=========================
"""
            logger.info(metrics_summary)
            await updater.update_status(TaskState.working, new_agent_text_message(metrics_summary))

            result = EvalResult(
                winner=participant_role if resolved else "none",
                detail={
                    "response": response,
                    "patch_applied": patch_applied,
                    "fail_to_pass_passed": fail_to_pass_passed,
                    "fail_to_pass_total": fail_to_pass_total,
                    "fail_to_pass_pct": fail_to_pass_pct,
                    "pass_to_pass_passed": pass_to_pass_passed,
                    "pass_to_pass_total": pass_to_pass_total,
                    "pass_to_pass_pct": pass_to_pass_pct,
                    "resolved": resolved
                }
            )
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=metrics_summary)),
                    Part(root=TextPart(text=result.model_dump_json())),
                ],
                name="Result",
            )

        except Exception as e:
            logger.error(f"Error during execution: {e}")
            await updater.failed(new_agent_text_message(f"Error: {e}"))
            raise e
        finally:
            shutil.rmtree(temp_dir)
            self._tool_provider.reset()
    
    def _extract_patch(self, response: str) -> str | None:
        """Extract patch content from agent response."""
        # Look for <patch>...</patch> tags
        if "<patch>" in response and "</patch>" in response:
            start = response.find("<patch>") + len("<patch>")
            end = response.find("</patch>")
            return response[start:end].strip()
        
        # Look for ```diff or ```patch code blocks
        if "```diff" in response:
            start = response.find("```diff") + len("```diff")
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        if "```patch" in response:
            start = response.find("```patch") + len("```patch")
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        # Look for unified diff format (--- a/ and +++ b/)
        lines = response.split("\n")
        patch_lines = []
        in_patch = False
        for line in lines:
            if line.startswith("--- a/") or line.startswith("--- "):
                in_patch = True
            if in_patch:
                patch_lines.append(line)
                if line.startswith("```") and len(patch_lines) > 1:
                    patch_lines.pop()
                    break
        
        if patch_lines:
            return "\n".join(patch_lines)
        
        return None

    async def _ask_llm(self, prompt: str) -> str:
        try:
            response = await litellm.acompletion(
                model="ollama/qwen2.5-coder:7b",
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content
            # Clean up markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise e


async def main():
    parser = argparse.ArgumentParser(description="Run the Repo Green Agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9020, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--cloudflare-quick-tunnel", action="store_true", help="Use a Cloudflare quick tunnel. Requires cloudflared. This will override --card-url")
    args = parser.parse_args()

    if args.cloudflare_quick_tunnel:
        agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
    else:
        agent_url_cm = contextlib.nullcontext(args.card_url or f"http://{args.host}:{args.port}/")

    async with agent_url_cm as agent_url:
        agent = SweVerifiedGreenAgent()
        executor = GreenExecutor(agent)
        
        # Create a simple agent card
        skill = AgentSkill(
            id="setup_and_verify_repo",
            name="Setup and Verify Repository",
            description="Clones a repository, sets up the environment using agentic tools, and runs tests.",
            tags=["swe", "verification"],
            examples=[
                """
{
  "participants": {
    "agent": "http://agent-url"
  },
  "config": {
    "repo_url": "https://github.com/example/repo",
    "base_commit": "main",
    "hints_text": "Fix the bug",
    "problem_statement": "The code crashes"
  }
}
"""
            ]
        )

        agent_card = AgentCard(
            name="SweVerifiedGreenAgent",
            description="Agent that sets up a repo using agentic tools and sends a problem statement.",
            url=agent_url,
            version="0.1.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[skill]
        )

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

        uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()

if __name__ == '__main__':
    asyncio.run(main())
