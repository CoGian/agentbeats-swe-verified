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
    parser = argparse.ArgumentParser(description="Run the A2A debater agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--provider", type=str, default="ollama", help="LLM provider to use (default: ollama)")
    args = parser.parse_args()

    if args.provider.lower() != "gemini":
        if args.provider.lower() != "ollama":
            raise ValueError("Unsupported provider. Only 'ollama' and 'gemini' are supported.")
        else:
            ollama_api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            ollama_model = os.getenv("DEBATE_OLLAMA_MODEL", "llama3.1:8b")
            model = LiteLlm(
                model=f"ollama_chat/{ollama_model}",
                api_base=ollama_api_base
            )
    else:
       model = "gemini-2.0-flash"
   
   
    root_agent = Agent(
        name="debater",
        model=model,
        description="Participates in a debate.",
        instruction="You are a professional debater.",
    )

    agent_card = AgentCard(
        name="debater",
        description='Participates in a debate.',
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
