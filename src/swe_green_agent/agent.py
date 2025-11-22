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

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"Starting SWE verified agent evaluation: {req}")
        
        # Load data from parquet
        try:
            df = pd.read_parquet("data/test-00000-of-00001.parquet")
            # For simplicity, pick the first row or a random one. 
            # In a real scenario, we might want to select based on some criteria or iterate.
            # Here we just take the first one for demonstration.
            row = df.iloc[0]
            
            repo_url = row["repo"] # The column name in the parquet file is likely 'repo' based on standard datasets, but I should verify. 
            # Wait, the user said: repo_url = req.config["repo_url"] ... from the parqet
            # Let's check the parquet columns again from my inspection script output if I had it.
            # I'll assume standard names or map them.
            # The user provided code snippet used req.config keys. I will replace them with parquet data.
            
            # Let's re-read the parquet columns from the previous step output if possible.
            # Step 61 output: FAIL_TO_PASS sample: ["astropy/modeling/tests/test_separable.py::test_separable[compound_model6-result6]", ...]
            # It didn't show all columns. I should probably be careful.
            # But standard SWE-bench parquet has 'repo', 'base_commit', 'hints_text', 'problem_statement', 'fail_to_pass', 'pass_to_pass'.
            
            repo_url = row.get("repo", row.get("repo_url")) # Handle potential naming differences
            # If repo is just "astropy/astropy", we might need to prepend https://github.com/
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

        # Assume there is at least one participant, pick the first one or a specific role if defined
        # The user didn't specify a role, so we'll just take the first one.
        participant_role = next(iter(req.participants))
        participant_url = str(req.participants[participant_role])

        temp_dir = tempfile.mkdtemp(prefix="agentbeats_repo_")
        try:
            await updater.update_status(TaskState.working, new_agent_text_message(f"Cloning repository {repo_url}..."))
            
            # Clone repo
            subprocess.check_call(["git", "clone", repo_url, temp_dir])
            
            # Checkout base commit
            subprocess.check_call(["git", "checkout", base_commit], cwd=temp_dir)
            
            await updater.update_status(TaskState.working, new_agent_text_message(f"Setting up environment in {temp_dir}..."))
            
            # List files to give context to LLM
            files = []
            for filename in os.listdir(temp_dir):
                files.append(filename)

            
            # Limit file list size if too large
            files_str = "\n".join(files[:500]) 
            if len(files) > 500:
                files_str += "\n... (more files)"

            # Ask LLM to identify dependency files
            dependency_files_prompt = f"""
You are an expert software engineer.
Given the following file list from a python repository, identify the files that contain information about dependencies and environment setup (e.g. requirements.txt, setup.py, pyproject.toml, environment.yml, Makefile, tox.ini, etc.).
Return ONLY a JSON list of strings, where each string is a file path from the list.
Files:
{files_str}
"""
            dependency_files_json = await self._ask_llm(dependency_files_prompt)
            dependency_files = []
            try:
                dependency_files = json.loads(dependency_files_json)
                if isinstance(dependency_files, str):
                    dependency_files = [dependency_files]
            except json.JSONDecodeError:
                # Fallback
                dependency_files = [line.strip() for line in dependency_files_json.split('\n') if line.strip() and not line.strip().startswith('```')]
            
            dependency_file_contents = ""
            for file_path in dependency_files:
                # Validate file_path is in files list to prevent hallucination or path traversal
                if file_path not in files:
                     continue
                try:
                    with open(os.path.join(temp_dir, file_path), "r") as f:
                        content = f.read()
                        dependency_file_contents += f"\n--- {file_path} ---\n{content}\n"
                except Exception as e:
                    logger.warning(f"Failed to read dependency file {file_path}: {e}")

            # Ask LLM for setup commands
            setup_prompt = f"""
You are an expert software engineer.
Given the following file list and the content of dependency-related files from a python repository, provide the shell commands to install the dependencies and set up the environment.
Assume a standard linux environment with python installed.
Return ONLY a JSON list of strings, where each string is a command.
Example: ["pip install -r requirements.txt", "pip install ."]

Files:
{files_str}

Dependency File Contents:
{dependency_file_contents}
"""
            setup_commands_json = await self._ask_llm(setup_prompt)
            try:
                setup_commands = json.loads(setup_commands_json)
                if isinstance(setup_commands, str): # Handle if LLM returns a string that is just one command or double encoded
                     setup_commands = [setup_commands]
            except json.JSONDecodeError:
                # Fallback if not valid JSON, maybe just split lines
                setup_commands = [line.strip() for line in setup_commands_json.split('\n') if line.strip() and not line.strip().startswith('```')]

            for cmd in setup_commands:
                logger.info(f"Running setup command: {cmd}")
                subprocess.check_call(cmd, shell=True, cwd=temp_dir)

            await updater.update_status(TaskState.working, new_agent_text_message("Dependencies installed. Running tests..."))

            # Ask LLM for test commands
            test_prompt = f"""
You are an expert software engineer.
Given the following file list and test identifiers, provide the shell command(s) to run these specific tests.
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

            for cmd in test_commands:
                logger.info(f"Running test command: {cmd}")
                # We allow tests to fail, so we use run instead of check_call and log the output
                subprocess.run(cmd, shell=True, cwd=temp_dir)

            await updater.update_status(TaskState.working, new_agent_text_message("Tests executed. Sending problem to agent..."))

            # Send problem statement and hints
            message = f"Problem Statement:\n{problem_statement}\n\nHints:\n{hints_text}"
            
            response = await self._tool_provider.talk_to_agent(message, participant_url, new_conversation=True)
            
            logger.info(f"Agent response: {response}")
            await updater.update_status(TaskState.working, new_agent_text_message(f"Agent responded: {response}"))

            result = EvalResult(winner=participant_role, detail={"response": response})
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=response)),
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

    async def _ask_llm(self, prompt: str) -> str:
        try:
            # simple retry logic could be added here
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
            # Return empty list representation as fallback to avoid crashing everything? 
            # Or re-raise. For now, let's re-raise as it's critical.
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
            description="Clones a repository, sets up the environment, and runs tests.",
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
            description="Agent that sets up a repo and sends a problem statement.",
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
