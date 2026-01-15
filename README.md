# AgentBeats SWE Verified

An AI-powered agent system for Software Engineering evaluation tasks. This project implements a collaborative green agent framework using the A2A (Agent-to-Agent) protocol to automate repository setup, dependency management, and code analysis.

## 🚀 Features
- **Automated Environment Setup**: Intelligent Python version detection and virtual environment creation
- **Dependency Management**: Automated parsing and installation of project dependencies
- **A2A Protocol Support**: Built on the A2A SDK for standardized agent communication
- **Parallel Processing**: Configurable concurrent row processing for dataset evaluation

## 🏆 Leaderboard

### About the Green Agent
The **SWE Verified Green Agent** is an orchestrator that evaluates participant agents on their ability to solve real-world software engineering tasks from the [SWE-bench Verified](https://www.swebench.com/) dataset. The green agent handles repository cloning, environment setup, dependency installation, and coordinates with participant agents to generate patches for resolving GitHub issues. It then runs the test suite to verify whether the produced patches correctly fix the failing tests without breaking existing functionality.

### Scoring & Evaluation Metrics
Participant agents are evaluated based on their patch quality across multiple dimensions. The status of each instance is determined by the following decision tree:

```
Patch Applied?
├── No → no_op
└── Yes
    ├── All F2P pass AND All P2P pass → resolved
    ├── All F2P pass (P2P broken) → breaking_resolved  
    ├── All P2P pass
    │   ├── Some F2P pass → partially_resolved
    │   └── No F2P pass → no_op
    ├── Some F2P pass (P2P broken) → work_in_progress
    └── No F2P pass (P2P broken) → regression
```

The leaderboard displays the following metrics derived from the `EvalResult` model:

| Metric | Description |
|--------|-------------|
| **Total Instances** | Number of SWE-bench tasks evaluated |
| **Resolved %** | Patch applied, **all** fail-to-pass tests pass, **and** all pass-to-pass tests remain passing (perfect fix) |
| **Breaking Resolved %** | Patch applied and **all** fail-to-pass tests pass, but some pass-to-pass tests now fail (fix introduced regressions) |
| **Partially Resolved %** | Patch applied, all pass-to-pass tests still pass, but only **some** fail-to-pass tests now pass |
| **Work in Progress %** | Patch applied, **some** fail-to-pass tests pass, but pass-to-pass tests are also broken |
| **Regression %** | Patch applied but **no** fail-to-pass tests pass and pass-to-pass tests are broken (patch made things worse) |
| **No-Op %** | Patch was not applied, or patch was applied but had no positive effect on fail-to-pass tests while keeping pass-to-pass intact |
| **Error %** | Task resulted in an error during evaluation (e.g., setup failure, timeout) |
| **Fail-to-Pass Passed %** | Aggregate rate: total fail-to-pass tests that now pass across all instances |
| **Pass-to-Pass Passed %** | Aggregate rate: total pass-to-pass tests that remain passing across all instances |

### Configurable Parameters
The scenario can be configured via `scenario.toml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_concurrent_rows` | Number of dataset rows to process in parallel | `1` |
| `max_rows` | Total number of dataset rows to process (`-1` for all) | `3` |
| `green_agent_model` | LLM model used by the green agent for orchestration | `ollama/qwen2.5-coder:7b` |

### Requirements for Participant Agents
Participant agents must:
1. **Implement the A2A Protocol**: Expose an endpoint compatible with the [A2A SDK](https://github.com/google/a2a) for receiving tasks and returning responses
2. **Accept Task Messages**: Handle incoming messages containing problem statements, repository context, and hints
3. **Generate Git Patches**: Produce valid `git diff` patches that can be applied to resolve the specified issue

## 📋 Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended for fast package management)
- Docker (optional, for containerized deployment)

## 🛠️ Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/CoGian/agentbeats-swe-verified.git
cd agentbeats-swe-verified

# Install dependencies using uv
uv sync

# Or create a virtual environment first
uv venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows

uv pip install -e .
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/CoGian/agentbeats-swe-verified.git
cd agentbeats-swe-verified

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows

# Install the package
pip install -e .
```

## 🦙 Ollama Setup

This project uses Ollama for local LLM inference. Follow these steps to set up Ollama:

### Install Ollama

```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Or download from https://ollama.com/download
```

### Create the Custom Model

```bash
# Create the qwen2.5-coder:7b model with custom configuration
ollama create qwen2.5-coder:7b -f qwen2_5_coder_7b.ollamaModelFile
```

### Start Ollama Server

```bash
# Start the Ollama server (runs on port 11434 by default)
ollama serve

# Verify the model is running
ollama ps
```

## 🏃 Running the Application

### Running a Scenario

The main entry point is `agentbeats-run` which orchestrates the agent evaluation:

```bash
# Run the default SWE scenario
uv run agentbeats-run swe_scenario/scenario.toml

# Run with visible logs
uv run agentbeats-run swe_scenario/scenario.toml --show-logs

# Start agents only (for debugging)
uv run agentbeats-run swe_scenario/scenario.toml --serve-only
```

### Running Individual Agents

You can also run agents individually:

```bash
# Start the Green Agent (orchestrator)
uv run src/swe_green_agent/agent.py

# Start a participant agent (dummy LLM for testing)
uv run src/swe_purple_agent/dummy_llm.py --host 127.0.0.1 --port 9021
```

## 🐳 Docker

### Using Pre-built Image (Recommended)

Pull the latest image from GitHub Container Registry:

```bash
docker pull ghcr.io/cogian/agentbeats-swe-verified:v1.3
```

OR 

Build the image from source:
```bash
docker build -t ghcr.io/cogian/agentbeats-swe-verified:v1.3 .
```

Run the container:

```bash
# Run locally built image
docker run --gpus all -p 9020:9009 ghcr.io/cogian/agentbeats-swe-verified:v1.3
```

### Purple Dummy Agent (No Ollama)

A lightweight Docker image for running the purple dummy agent using Gemini API (no Ollama required).

**Build:**
```bash
docker build -f Dockerfile.dummy_llm -t ghcr.io/cogian/agentbeats-swe-verified-dummy-gemini-2.5-flash-lite .
```

**Push:**
```bash
docker push ghcr.io/cogian/agentbeats-swe-verified-dummy-gemini-2.5-flash-lite
```

**Run:**
```bash
docker run -p 9021:9009 -e GEMINI_API_KEY=your-api-key ghcr.io/cogian/agentbeats-swe-verified-dummy-gemini-2.5-flash-lite
```

You can override the model at runtime:
```bash
docker run -p 9021:9009 -e LLM_MODEL=gemini/gemini-2.0-flash -e GEMINI_API_KEY=your-key ghcr.io/cogian/agentbeats-swe-verified-dummy-gemini-2.5-flash-lite
```

## 📁 Project Structure

```
agentbeats-swe-verified/
├── src/
│   ├── agentbeats/           # Core package
│   │   ├── client.py         # A2A client implementation
│   │   ├── client_cli.py     # CLI client for evaluation
│   │   ├── run_scenario.py   # Scenario runner
│   │   ├── repo_tools.py     # Repository utility tools
│   │   └── ...
│   └── swe_green_agent/      # Green agent implementation
│       └── agent.py          # Main orchestrator agent
├── swe_scenario/             # Scenario definitions
│   ├── scenario.toml         # Default scenario configuration
│   └── dummy_llm.py          # Test participant agent
├── data/                     # Evaluation datasets
├── pyproject.toml            # Project configuration
├── Dockerfile                # Container definition
└── README.md                 # This file
```

## 🔧 Scenario Configuration

Scenarios are defined in TOML format. Example (`swe_scenario/scenario.toml`):

```toml
[green_agent]
endpoint = "http://127.0.0.1:9020"
cmd = "python src/swe_green_agent/agent.py --host 127.0.0.1 --port 9020"

[[participants]]
role = "agent"
endpoint = "http://127.0.0.1:9021"
cmd = "python swe_scenario/dummy_llm.py --host 127.0.0.1 --port 9021"

[config]
max_concurrent_rows = 5  # Parallel processing configuration
```

## 📊 Data Format

The evaluation uses Parquet files with the following expected columns:
- `repo`: Repository name
- `base_commit`: Git commit hash for code analysis
- `environment_setup_commit`: Commit for environment setup
- `problem_statement`: Description of the task
- `hints_text`: Additional hints for the LLM

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [A2A Protocol](https://github.com/google/a2a) - Agent-to-Agent communication protocol
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified LLM interface
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager