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
from a2a.types import AgentCard, TaskState, Part, TextPart, AgentCapabilities, AgentSkill, DataPart
from a2a.utils import new_agent_text_message
from dotenv import load_dotenv

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider
from agentbeats.cloudflare import quick_tunnel
from agentbeats.repo_tools import read_file, run_command, grep_search
from loguru import logger

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
            "name": "grep_search",
            "description": "Search for a pattern in repository files using grep. Use to find Python version requirements, dependencies, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Pattern to search for (e.g., 'requires-python', 'python_requires')"
                    }
                },
                "required": ["pattern"]
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
                        "description": "Shell command to execute (e.g., 'uv python install 3.11', 'uv venv --python 3.11')"
                    }
                },
                "required": ["command"]
            }
        }
    }
]

# System prompt for Phase 1: Environment Setup (Python version + venv + pip)
ENV_SETUP_PROMPT = """You are setting up a Python environment for a repository.

## TASK: Create a virtual environment with the correct Python version.

## Step 1: Find Python version requirement
Use grep_search with pattern: "requires-python|python_requires"
- This finds version in pyproject.toml (requires-python) or setup.cfg (python_requires)
- Look at the version number (e.g., >=3.8 means use 3.8)

## Step 2: Run these commands (use the version you found, or 3.11 as default)
```
uv python install <VERSION>
uv venv --python <VERSION>
uv pip install pip --python .venv/bin/python
```

## Tools:
- grep_search: Search for patterns like "requires-python" or "python_requires"
- read_file: Read pyproject.toml, setup.py, setup.cfg
- run_command: Execute shell commands

## Rules:
- FIRST use grep_search to find Python version
- THEN run the 3 uv commands
- Default to Python 3.11 if no version specified
- Respond "ENV READY" when all commands succeed
- Do NOT respond "ENV READY" if any command fails

## STOPPING CONDITION:
- STOP immediately after the virtual environment is created and pip is installed successfully.
- Once you see "Exit code: 0" for all 3 commands (uv python install, uv venv, uv pip install pip), respond with "ENV READY" and STOP.
- Do NOT continue with any additional commands or file reading after the goal is achieved.
"""

# System prompt for Phase 2: Dependency Installation  
DEPS_INSTALL_PROMPT = """You are installing Python dependencies for a repository.

## TASK: Install all dependencies so the package can be tested.

## Step 1: Install build dependencies (ALWAYS run this first)
```
.venv/bin/pip install "setuptools<70" wheel cython "numpy<2" extension_helpers setuptools_scm
```

## Step 2: Install the package with test extras (try this first)
```
.venv/bin/pip install -e ".[test]" --no-build-isolation
```
If this fails, install package and test deps separately:
```
.venv/bin/pip install -e . --no-build-isolation
```

## Tools:
- grep_search: Search for patterns like "test", "dependencies", "requirements"
- read_file: Read README, pyproject.toml
- run_command: Execute shell commands

## Rules:
- Run pip commands using .venv/bin/pip
- Use --no-build-isolation to avoid setuptools compatibility issues
- Try .[test] extras first, fall back to separate installs if needed
- Respond "DEPS INSTALLED" when all commands succeed
- Do NOT respond "DEPS INSTALLED" if any command fails

## STOPPING CONDITION:
- STOP immediately after dependencies are installed successfully.
- Once pip install commands complete with "Exit code: 0", respond with "DEPS INSTALLED" and STOP.
- Do NOT continue with any additional commands, file reading, or exploration after the goal is achieved.
"""


class SweVerifiedGreenAgent(GreenAgent):
    def __init__(self):
        self._required_config_keys = []  # Config values come from parquet file
        self._tool_provider = ToolProvider()
        self._model = "ollama/qwen2.5-coder:7b"  # Default model, can be overridden in run_eval

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        
        if not request.participants:
             return False, "Missing participants"

        return True, "ok"

    async def _run_agentic_loop(self, repo_dir: str, system_prompt: str, user_message: str, 
                                  phase_name: str, updater: TaskUpdater, max_iterations: int = 10, model: str = None) -> None:
        """Run a focused agentic loop with the given prompt."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # State tracking for goal completion
        state = {
            "python_installed": False,
            "venv_created": False,
            "pip_installed": False,
            "build_deps_installed": False,
            "package_installed": False,
        }
        
        for iteration in range(max_iterations):
            logger.info(f"[{phase_name}] === Iteration {iteration + 1}/{max_iterations} ===")
            await updater.update_status(
                TaskState.working, 
                new_agent_text_message(f"{phase_name} iteration {iteration + 1}...")
            )
            
            try:
                response = await litellm.acompletion(
                    model=model or self._model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto"
                )
            except Exception as e:
                logger.error(f"[{phase_name}] LLM call failed: {e}")
                raise e
            
            assistant_message = response.choices[0].message
            messages.append(assistant_message.model_dump())
            
            # Check for completion keywords in LLM response
            if assistant_message.content:
                content_upper = assistant_message.content.upper()
                if "ENV READY" in content_upper:
                    logger.info(f"[{phase_name}] ✓ Goal achieved (ENV READY detected): {assistant_message.content}")
                    break
                if "DEPS INSTALLED" in content_upper:
                    logger.info(f"[{phase_name}] ✓ Goal achieved (DEPS INSTALLED detected): {assistant_message.content}")
                    break
            
            # Check if there are tool calls
            if not assistant_message.tool_calls:
                logger.info(f"[{phase_name}] ✓ Completed (no more tool calls). Final message: {assistant_message.content}")
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
                    result = read_file(repo_dir, file_path)
                    await updater.update_status(
                        TaskState.working, 
                        new_agent_text_message(f"Reading: {file_path}")
                    )
                    logger.info(f"[{phase_name}]   File read: {len(result)} chars")
                    
                elif func_name == "run_command":
                    command = args.get("command", "")
                    logger.info(f"[{phase_name}] → Running command: {command}")
                    result = run_command(repo_dir, command)
                    
                    # Extract exit code for logging and state tracking
                    exit_code = "unknown"
                    if "Exit code: " in result:
                        exit_code = result.split("Exit code: ")[1].split("\n")[0]
                    
                    # Track successful commands for goal completion
                    if exit_code == "0":
                        cmd_lower = command.lower()
                        if "uv python install" in cmd_lower:
                            state["python_installed"] = True
                            logger.info(f"[{phase_name}] ✓ State: python_installed = True")
                        if "uv venv" in cmd_lower:
                            state["venv_created"] = True
                            logger.info(f"[{phase_name}] ✓ State: venv_created = True")
                        if "pip install pip" in cmd_lower or "pip install --upgrade pip" in cmd_lower:
                            state["pip_installed"] = True
                            logger.info(f"[{phase_name}] ✓ State: pip_installed = True")
                        if "pip install" in cmd_lower and ("setuptools" in cmd_lower or "wheel" in cmd_lower):
                            state["build_deps_installed"] = True
                            logger.info(f"[{phase_name}] ✓ State: build_deps_installed = True")
                        if "pip install -e" in cmd_lower or "pip install ." in cmd_lower:
                            state["package_installed"] = True
                            logger.info(f"[{phase_name}] ✓ State: package_installed = True")
                    
                    await updater.update_status(
                        TaskState.working, 
                        new_agent_text_message(f"Running: {command[:50]}...")
                    )
                    logger.info(f"[{phase_name}]   Command exit code: {exit_code}")
                    

                
                elif func_name == "grep_search":
                    pattern = args.get("pattern", "")
                    logger.info(f"[{phase_name}] → Searching for: {pattern}")
                    result = grep_search(repo_dir, pattern)
                    await updater.update_status(
                        TaskState.working, 
                        new_agent_text_message(f"Searching: {pattern}")
                    )
                    logger.info(f"[{phase_name}]   Search results: {len(result)} chars")
                    
                else:
                    result = f"Unknown tool: {func_name}"
                    logger.warning(f"[{phase_name}] Unknown tool: {func_name}")
                
                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            
            # Check for programmatic goal completion after processing all tool calls
            if phase_name == "ENV_SETUP" and state["venv_created"] and state["pip_installed"]:
                logger.info(f"[{phase_name}] ✓ Goal achieved programmatically (venv + pip installed)")
                break
            if phase_name == "DEPS_INSTALL" and state["package_installed"]:
                logger.info(f"[{phase_name}] ✓ Goal achieved programmatically (package installed)")
                break
        else:
            logger.warning(f"[{phase_name}] ⚠ Reached max iterations ({max_iterations})")

    async def _run_env_setup(self, repo_dir: str, file_listing: str, updater: TaskUpdater, model: str = None) -> None:
        """Phase 1: Identify Python version, install it, create venv."""
        await self._run_agentic_loop(
            repo_dir=repo_dir,
            system_prompt=ENV_SETUP_PROMPT,
            user_message=f"Repository files:\n{file_listing}",
            phase_name="ENV_SETUP",
            updater=updater,
            max_iterations=8,
            model=model
        )

    async def _run_deps_install(self, repo_dir: str, file_listing: str, updater: TaskUpdater, model: str = None) -> None:
        """Phase 2: Install build deps, package, and test deps."""
        await self._run_agentic_loop(
            repo_dir=repo_dir,
            system_prompt=DEPS_INSTALL_PROMPT,
            user_message=f"Repository files:\n{file_listing}",
            phase_name="DEPS_INSTALL",
            updater=updater,
            max_iterations=8,
            model=model
        )


    async def _process_single_row(
        self, 
        row_idx: int, 
        row: Any, 
        total_rows: int, 
        participant_url: str, 
        updater: TaskUpdater,
        semaphore: asyncio.Semaphore,
        model: str = None
    ) -> dict:
        """Process a single row/instance. Returns result dict."""
        async with semaphore:
            instance_id = row.get("instance_id", f"row_{row_idx}")
            logger.info(f"=== Processing instance {row_idx + 1}/{total_rows}: {instance_id} ===")
            await updater.update_status(TaskState.working, new_agent_text_message(f"Processing instance {row_idx + 1}/{total_rows}: {instance_id}"))
            
            try:
                repo_url = row.get("repo", row.get("repo_url"))
                if repo_url and not repo_url.startswith("http"):
                    repo_url = f"https://github.com/{repo_url}"
                    
                base_commit = row["base_commit"]
                environment_setup_commit = row.get("environment_setup_commit", base_commit)
                hints_text = row["hints_text"]
                problem_statement = row["problem_statement"]
                fail_to_pass = json.loads(row["FAIL_TO_PASS"]) if isinstance(row["FAIL_TO_PASS"], str) else row["FAIL_TO_PASS"]
                pass_to_pass = json.loads(row["PASS_TO_PASS"]) if isinstance(row["PASS_TO_PASS"], str) else row["PASS_TO_PASS"]
                
            except Exception as e:
                logger.error(f"Failed to parse row {row_idx}: {e}")
                return {"instance_id": instance_id, "error": str(e), "resolved": False}

            temp_dir = tempfile.mkdtemp(prefix="agentbeats_repo_")
            # Create a per-instance tool provider to avoid state conflicts
            tool_provider = ToolProvider()
            try:
                # =============================================
                # PHASE 1: Clone and checkout environment_setup_commit
                # =============================================
                await updater.update_status(TaskState.working, new_agent_text_message(f"[{instance_id}] Cloning repository {repo_url}..."))
                
                # Git clone
                clone_result = subprocess.run(
                    ["git", "clone", repo_url, temp_dir],
                    capture_output=True,
                    text=True
                )
                if clone_result.returncode != 0:
                    raise Exception(f"Git clone failed: {clone_result.stderr}")
                logger.info(f"Cloned repository to {temp_dir}")
                
                # Git checkout environment_setup_commit for environment setup
                await updater.update_status(TaskState.working, new_agent_text_message(f"[{instance_id}] Checking out environment_setup_commit {environment_setup_commit[:8]}..."))
                checkout_result = subprocess.run(
                    ["git", "checkout", environment_setup_commit],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True
                )
                if checkout_result.returncode != 0:
                    raise Exception(f"Git checkout failed: {checkout_result.stderr}")
                logger.info(f"Checked out environment_setup_commit: {environment_setup_commit}")
                
                # List files (for agent context)
                await updater.update_status(TaskState.working, new_agent_text_message(f"[{instance_id}] Listing repository files..."))
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
                await updater.update_status(TaskState.working, new_agent_text_message(f"[{instance_id}] Phase 1: Setting up Python environment..."))
                await self._run_env_setup(temp_dir, file_listing, updater, model=model)
                
                # =============================================
                # PHASE 2B: Dependency Installation
                # =============================================
                await updater.update_status(TaskState.working, new_agent_text_message(f"[{instance_id}] Phase 2: Installing dependencies..."))
                await self._run_deps_install(temp_dir, file_listing, updater, model=model)
                
                # =============================================
                # PHASE 3: Switch to base_commit for patch application
                # =============================================
                await updater.update_status(TaskState.working, new_agent_text_message(f"[{instance_id}] Switching to base_commit {base_commit[:8]}..."))
                checkout_base_result = subprocess.run(
                    ["git", "checkout", base_commit],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True
                )
                if checkout_base_result.returncode != 0:
                    raise Exception(f"Git checkout to base_commit failed: {checkout_base_result.stderr}")
                logger.info(f"Switched to base_commit: {base_commit}")
                
                # =============================================
                # PHASE 4: Send problem to participant agent and wait for response
                # =============================================
                await updater.update_status(TaskState.working, new_agent_text_message(f"[{instance_id}] Sending problem to participant agent..."))
                
                # Send problem statement, hints, and repo dir - participant will use its own tools
                message = f"REPO_DIR:{temp_dir}\n\nProblem Statement:\n{problem_statement}\n\nHints:\n{hints_text}"
                
                response = await tool_provider.talk_to_agent(message, participant_url, new_conversation=True)
                
                logger.info(f"Agent response: {response}")
                await updater.update_status(TaskState.working, new_agent_text_message(f"[{instance_id}] Received response from participant. Extracting patch..."))
                
                # =============================================
                # PHASE 5: Extract and apply patch
                # =============================================
                # First try regex-based extraction (faster)
                patch_content = self._extract_patch(response)
                
                # If regex fails, use LLM to transform/extract the patch
                if not patch_content:
                    logger.info("Regex extraction failed, trying LLM transformation...")
                    patch_content = await self._transform_to_patch_with_llm(response)
                
                patch_applied = False
                
                if patch_content:
                    # Save patch to file
                    patch_file = os.path.join(temp_dir, "proposed.patch")
                    with open(patch_file, "w") as f:
                        f.write(patch_content)
                    
                    # Apply patch using multiple strategies
                    await updater.update_status(TaskState.working, new_agent_text_message(f"[{instance_id}] Applying patch..."))
                    patch_applied, apply_method = await self._apply_patch_with_fallbacks(temp_dir, "proposed.patch", updater, instance_id)
                    
                    if patch_applied:
                        logger.info(f"Patch applied successfully using: {apply_method}")
                else:
                    logger.warning("No patch found in agent response")
                    await updater.update_status(TaskState.working, new_agent_text_message(f"[{instance_id}] No patch found in agent response"))

                # =============================================
                # PHASE 6: Run tests with venv activated
                # =============================================
                await updater.update_status(TaskState.working, new_agent_text_message(f"[{instance_id}] Running tests..."))
                
                # Run tests and collect results
                fail_to_pass_results = {}
                pass_to_pass_results = {}
                
                # Run FAIL_TO_PASS tests (these should now PASS after the fix)
                for test in fail_to_pass:
                    await updater.update_status(TaskState.working, new_agent_text_message(f"[{instance_id}] Running FAIL_TO_PASS test: {test[:50]}..."))
                    cmd = f".venv/bin/python -m pytest {test} -v"
                    result = subprocess.run(cmd, shell=True, cwd=temp_dir, capture_output=True, text=True)
                    fail_to_pass_results[test] = result.returncode == 0
                    logger.info(f"FAIL_TO_PASS test '{test}': {'PASSED' if result.returncode == 0 else 'FAILED'}")
                
                # Run PASS_TO_PASS tests (these should still PASS after the fix)
                for test in pass_to_pass:
                    await updater.update_status(TaskState.working, new_agent_text_message(f"[{instance_id}] Running PASS_TO_PASS test: {test[:50]}..."))
                    cmd = f".venv/bin/python -m pytest {test} -v"
                    result = subprocess.run(cmd, shell=True, cwd=temp_dir, capture_output=True, text=True)
                    pass_to_pass_results[test] = result.returncode == 0
                    logger.info(f"PASS_TO_PASS test '{test}': {'PASSED' if result.returncode == 0 else 'FAILED'}")

                # =============================================
                # PHASE 7: Calculate metrics for this instance
                # =============================================
                
                fail_to_pass_total = len(fail_to_pass_results)
                fail_to_pass_passed = sum(1 for v in fail_to_pass_results.values() if v)
                
                pass_to_pass_passed = sum(1 for v in pass_to_pass_results.values() if v)
                pass_to_pass_total = len(pass_to_pass_results)
                
                # Resolution: patch applies + all fail_to_pass pass + all pass_to_pass pass
                if not patch_applied:
                    status = "no_op"
                # 1. Check if EVERYTHING passed
                elif fail_to_pass_passed == fail_to_pass_total and pass_to_pass_passed == pass_to_pass_total:
                    status = "resolved"
                # 2. Check if all F2P fixed (but P2P must be broken, otherwise #1 would have triggered)
                elif fail_to_pass_passed == fail_to_pass_total:
                    status = "breaking_resolved"
                # 3. Check if P2P is perfect (Handle Partial vs None here)
                elif pass_to_pass_passed == pass_to_pass_total:
                    if fail_to_pass_passed > 0:
                        status = "partially_resolved"
                    else:
                        status = "no_op"
                # 4. If we are here, P2P is broken (Partial/None) AND F2P is not All
                elif fail_to_pass_passed > 0:
                    status = "work_in_progress"
                # 5. Final fallback: P2P Broken + F2P None
                else:
                    status = "regression"
                
                instance_result = {
                    "instance_id": instance_id,
                    "status": status,
                    "patch_applied": patch_applied,
                    "fail_to_pass_passed_pct": fail_to_pass_passed / fail_to_pass_total,
                    "fail_to_pass_total": fail_to_pass_total,   
                    "pass_to_pass_passed_pct": pass_to_pass_passed / pass_to_pass_total,
                    "pass_to_pass_total": pass_to_pass_total,
                }
                
                logger.info(f"[{instance_id}] Status: {status}")
                return instance_result

            except Exception as e:
                logger.error(f"[{instance_id}] Error during execution: {e}")
                return {"instance_id": instance_id, "error": str(e), "status": "error"}
            finally:
                shutil.rmtree(temp_dir)
                tool_provider.reset()

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"Starting SWE verified agent evaluation: {req}")
        
        # Configurable concurrency limit - read from config or default to 3
        MAX_CONCURRENT_ROWS = int(req.config.get("max_concurrent_rows", 3))
        # Configurable max rows to process - read from config or default to -1 (all rows)
        MAX_ROWS = int(req.config.get("max_rows", -1))
        # Configurable model - read from config or use default
        self._model = req.config.get("green_agent_model", "ollama/qwen2.5-coder:7b")
        
        # Load data from parquet
        try:
            df = pd.read_parquet("data/test-00000-of-00001.parquet")
            # Limit to first MAX_ROWS (-1 or None means all rows)
            if MAX_ROWS > 0:
                df = df.head(MAX_ROWS)
        except Exception as e:
            logger.error(f"Failed to read from parquet: {e}")
            await updater.failed(new_agent_text_message(f"Failed to read data: {e}"))
            return

        participant_role = next(iter(req.participants))
        participant_url = str(req.participants[participant_role])
        
        total_rows = len(df)
        logger.info(f"Processing {total_rows} rows with max {MAX_CONCURRENT_ROWS} concurrent workers")
        
        # Create semaphore to limit concurrent executions
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_ROWS)
        
        # Create tasks for all rows
        tasks = [
            self._process_single_row(
                row_idx=row_idx,
                row=row,
                total_rows=total_rows,
                participant_url=participant_url,
                updater=updater,
                semaphore=semaphore,
                model=self._model
            )
            for row_idx, row in df.iterrows()
        ]
        
        # Run all tasks concurrently (bounded by semaphore)
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that were returned
        processed_results = []
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                logger.error(f"Row {i} failed with exception: {result}")
                processed_results.append({"instance_id": f"row_{i}", "error": str(result), "status": "error"})
            else:
                processed_results.append(result)
        
        # Calculate totals
        resolved_pct = sum(1 for r in processed_results if r.get("status", "") == "resolved") / total_rows
        breaking_resolved_pct = sum(1 for r in processed_results if r.get("status", "") == "breaking_resolved") / total_rows
        partially_resolved_pct = sum(1 for r in processed_results if r.get("status", "") == "partially_resolved") / total_rows
        work_in_progress_pct = sum(1 for r in processed_results if r.get("status", "") == "work_in_progress") / total_rows
        regression_pct = sum(1 for r in processed_results if r.get("status", "") == "regression") / total_rows
        no_op_pct = sum(1 for r in processed_results if r.get("status", "") == "no_op") / total_rows
        error_pct = sum(1 for r in processed_results if r.get("status", "") == "error") / total_rows
        failed_to_pass_passed_pct = sum(r.get("fail_to_pass_passed_pct", 0) for r in processed_results) / total_rows
        pass_to_pass_passed_pct = sum(r.get("pass_to_pass_passed_pct", 0) for r in processed_results) / total_rows
        # =============================================
        # Final Summary
        # =============================================
        metrics_summary = f"""
=== FINAL EVALUATION RESULTS ===
Total Instances: {total_rows}
Resolved: {resolved_pct*100:.1f}%
Breaking Resolved: {breaking_resolved_pct*100:.1f}%
Partially Resolved: {partially_resolved_pct*100:.1f}%
Work In Progress: {work_in_progress_pct*100:.1f}%
Regression: {regression_pct*100:.1f}%
No Op: {no_op_pct*100:.1f}%
Error: {error_pct*100:.1f}%
Fail to Pass Passed: {failed_to_pass_passed_pct*100:.1f}%
Pass to Pass Passed: {pass_to_pass_passed_pct*100:.1f}%
================================
"""
        logger.info(metrics_summary)
        await updater.update_status(TaskState.working, new_agent_text_message(metrics_summary))

        result = EvalResult(    
            total_instances=total_rows,
            resolved_pct=resolved_pct,
            breaking_resolved_pct=breaking_resolved_pct,
            partially_resolved_pct=partially_resolved_pct,
            work_in_progress_pct=work_in_progress_pct,
            regression_pct=regression_pct,
            no_op_pct=no_op_pct,
            error_pct=error_pct,
            fail_to_pass_passed_pct=failed_to_pass_passed_pct,
            pass_to_pass_passed_pct=pass_to_pass_passed_pct,
            green_agent_model=self._model
        )
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=metrics_summary)),
                Part(root=DataPart(data=result.model_dump())),
            ],
            name="Result",
        )
    
    async def _apply_patch_with_fallbacks(self, repo_dir: str, patch_file: str, updater: TaskUpdater, instance_id: str) -> tuple[bool, str]:
        """Try multiple strategies to apply a patch, handling common LLM errors.
        
        Strategies tried in order:
        1. git apply (strict)
        2. git apply --ignore-whitespace (handles whitespace issues)
        3. git apply --3way (handles line number mismatches)
        4. patch -p1 with fuzz factor (most lenient)
        
        Returns:
            Tuple of (success: bool, method_used: str)
        """
        strategies = [
            {
                "name": "git apply",
                "cmd": ["git", "apply", patch_file],
            },
            {
                "name": "git apply --ignore-whitespace",
                "cmd": ["git", "apply", "--ignore-whitespace", patch_file],
            },
            {
                "name": "git apply --3way",
                "cmd": ["git", "apply", "--3way", patch_file],
            },
            {
                "name": "patch -p1 --fuzz=3",
                "cmd": ["patch", "-p1", "--fuzz=3", "-i", patch_file],
            },
        ]
        
        last_error = ""
        for strategy in strategies:
            logger.info(f"[{instance_id}] Trying patch strategy: {strategy['name']}")
            
            result = subprocess.run(
                strategy["cmd"],
                cwd=repo_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                await updater.update_status(
                    TaskState.working, 
                    new_agent_text_message(f"[{instance_id}] Patch applied using: {strategy['name']}")
                )
                return True, strategy["name"]
            else:
                last_error = result.stderr or result.stdout
                logger.warning(f"[{instance_id}] Strategy '{strategy['name']}' failed: {last_error[:200]}")
        
        # All strategies failed
        logger.error(f"[{instance_id}] All patch strategies failed. Last error: {last_error}")
        await updater.update_status(
            TaskState.working, 
            new_agent_text_message(f"[{instance_id}] Patch failed to apply (tried 4 strategies): {last_error[:150]}")
        )
        return False, "none"
    
    async def _transform_to_patch_with_llm(self, response: str) -> str | None:
        """Use LLM to transform code in the response into a proper unified diff patch.
        
        This is used when the response contains code blocks or suggestions but not
        in proper patch format. The LLM will attempt to create a valid diff.
        """
        prompt = f"""You are a patch generation assistant. The following response contains code changes or suggestions that need to be converted into a proper git patch format.

Your task is to analyze the response and generate a unified diff patch that can be applied using `git apply`.

OUTPUT FORMAT - respond with a single patch wrapped in <patch> tags:
<patch>
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -1,10 +1,15 @@
 unchanged line
-removed line
+added line
 unchanged line
</patch>

RULES:
1. Identify the file path from the context (look for file mentions, function names, module imports)
2. Create a proper unified diff with:
   - `--- a/path/to/file` for original
   - `+++ b/path/to/file` for modified
   - `@@ -start,count +start,count @@` line number markers
   - Lines starting with space (unchanged), - (removed), + (added)
3. If the response shows a complete new function/code, assume it REPLACES existing code
4. If you cannot determine the file path or changes, return: NO_PATCH_FOUND
5. Do NOT include any explanation outside the <patch> tags

RESPONSE TO TRANSFORM:
{response}

GENERATED PATCH:"""
        
        try:
            result = await self._ask_llm(prompt)
            result = result.strip()
            
            if result == "NO_PATCH_FOUND" or not result:
                logger.warning("LLM could not transform response to patch")
                return None
            
            # Extract content from <patch>...</patch> tags if present
            if "<patch>" in result and "</patch>" in result:
                start = result.find("<patch>") + len("<patch>")
                end = result.find("</patch>")
                result = result[start:end].strip()
            
            # Validate it looks like a patch
            if "---" in result and "+++" in result:
                # Clean up any trailing markdown artifacts (e.g., closing ```)
                lines = result.split('\n')
                cleaned_lines = []
                for line in lines:
                    # Stop if we hit a markdown code block delimiter
                    if line.strip() == '```' or line.strip().startswith('```'):
                        break
                    cleaned_lines.append(line)
                result = '\n'.join(cleaned_lines).strip()
                
                logger.info(f"LLM transformed response to patch: {len(result)} chars")
                return result
            else:
                logger.warning("LLM transformation doesn't look like a valid patch")
                return None
                
        except Exception as e:
            logger.error(f"LLM patch transformation failed: {e}")
            return None

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

    async def _ask_llm(self, prompt: str, model: str = None) -> str:
        try:
            response = await litellm.acompletion(
                model=model or self._model,
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
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
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
    "max_concurrent_rows": 5
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
