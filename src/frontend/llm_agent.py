"""
CLI LLM Agent — natural language interface to the UGIF pipeline.

Usage::

    python src/frontend/llm_agent.py --query "flood damage in Chennai August 2023"
    python src/frontend/llm_agent.py --query "earthquake in Turkey February 2023" --geojson
"""
from __future__ import annotations

import argparse
import json

from src.frontend.query_parser import parse_query
from src.frontend.geojson_builder import build_geojson


class LLMAgent:
    """Pluggable NL agent.  
    
    Defaults to the regex + spaCy parser.
    If OPENAI_API_KEY is set, upgrades to a LangChain-backed LLM parser.
    """

    def __init__(self) -> None:
        self._llm = self._try_load_llm()

    def _try_load_llm(self):
        """Try to initialise a LangChain LLM; returns None if unavailable."""
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate

            self._prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a geospatial query parser. "
                    "Given a user query, extract: location (string), "
                    "start_date (YYYY-MM-DD), end_date (YYYY-MM-DD), task (string). "
                    "Reply ONLY with valid JSON."
                )),
                ("human", "{query}"),
            ])
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            return self._prompt | llm
        except ImportError:
            return None

    def process(self, query: str) -> dict:
        """Process a natural language query into structured metadata.

        Args:
            query: Free-text disaster query.

        Returns:
            Dict with location, start_date, end_date, task.
        """
        if self._llm is not None:
            try:
                result = self._llm.invoke({"query": query})
                return json.loads(result.content)
            except Exception as e:
                print(f"[LLMAgent] LLM failed ({e}), falling back to regex parser.")

        return parse_query(query)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UGIF Natural Language Disaster Query Agent"
    )
    parser.add_argument("--query", "-q", required=True, help="Natural language query")
    parser.add_argument(
        "--geojson", "-g", action="store_true",
        help="Also output GeoJSON FeatureCollection"
    )
    args = parser.parse_args()

    agent = LLMAgent()
    metadata = agent.process(args.query)

    print("\n── Parsed Query ──────────────────────────────────────────")
    print(json.dumps(metadata, indent=2))

    if args.geojson:
        gj = build_geojson(metadata)
        print("\n── GeoJSON FeatureCollection ─────────────────────────────")
        print(json.dumps(gj, indent=2))


if __name__ == "__main__":
    main()
