import argparse
import os
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from a2a.types import (
    AgentCapabilities,
    AgentCard,
)

def main():
    parser = argparse.ArgumentParser(description="Run the Dummy LLM agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9021, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    args = parser.parse_args()

    # Use a simple model or the same as debater if configured
    # For "dummy" behavior, we can use a small model or even a mock if ADK supports it easily.
    # But since the user said "take inspiration from debater.py", let's use LiteLlm with ollama/qwen2.5-coder:7b as used in the green agent,
    # or just use the same default as debater (llama3.1:8b) if available.
    # Let's default to qwen2.5-coder:7b since we know it's there.
    
    ollama_api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    ollama_model = os.getenv("DUMMY_OLLAMA_MODEL", "qwen2.5-coder:7b")
    
    model = LiteLlm(
        model=f"ollama_chat/{ollama_model}",
        api_base=ollama_api_base
    )

    root_agent = Agent(
        name="dummy_llm",
        model=model,
        description="A dummy agent that replies to the SWE agent.",
        instruction="You are a helpful assistant. You are receiving a problem statement and hints. You should acknowledge them and say you are working on it.",
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
