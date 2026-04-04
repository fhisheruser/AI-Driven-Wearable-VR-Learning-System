"""Entry point for the AI-Driven VR Learning System.

Usage:
    python main.py                  # Start the API server
    python main.py --demo           # Run a demo query through the pipeline
    python main.py --port 8080      # Start server on custom port
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server."""
    import uvicorn
    from src.api.server import create_app

    app = create_app()
    logger.info(f"Starting AI-VR Learning System on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


def run_demo():
    """Run a demo showing the full pipeline in action."""
    from src.pipeline import LearningPipeline

    pipeline = LearningPipeline()
    pipeline.initialize()

    demo_queries = [
        "How does the solar system work?",
        "Show me a water molecule",
        "Explain the human heart",
        "What is the Pythagorean theorem?",
        "Show me a DNA double helix",
    ]

    print("\n" + "=" * 60)
    print("  AI-Driven VR Learning System - Demo")
    print("=" * 60)

    for query in demo_queries:
        print(f"\n{'-' * 50}")
        print(f"  Student: \"{query}\"")
        print(f"{'-' * 50}")

        response = pipeline.process_text(query, learner_id="demo_user")

        print(f"  Intent:      {response.semantic_result.intent.value}")
        print(f"  Domain:      {response.semantic_result.domain}")
        print(f"  Topic:       {response.semantic_result.topic}")
        print(f"  Depth:       {response.reasoned_output.content_depth}")
        print(f"  Viz Type:    {response.reasoned_output.visualization_type}")
        print(f"  Difficulty:  {response.reasoned_output.difficulty_level}/5")
        print(f"  Scene:       {response.scene.title} ({len(response.scene.objects)} objects)")
        print(f"  Latency:     {response.total_latency_ms:.1f} ms")
        print(f"\n  Explanation: {response.reasoned_output.explanation[:120]}...")

        if response.reasoned_output.related_topics:
            print(f"  Related:     {', '.join(response.reasoned_output.related_topics)}")

    print(f"\n{'=' * 60}")
    print("  Demo complete! Run 'python main.py' to start the web server.")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="AI-Driven VR Learning System")
    parser.add_argument("--demo", action="store_true", help="Run pipeline demo")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
