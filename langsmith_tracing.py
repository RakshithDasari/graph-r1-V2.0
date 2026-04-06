import os
from langsmith import traceable
from langsmith.wrappers import wrap_openai


def setup_langsmith() -> None:
    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    if project:
        os.environ.setdefault("LANGSMITH_PROJECT", project)
        os.environ.setdefault("LANGCHAIN_PROJECT", project)


def wrap_openai_client(client):
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        return wrap_openai(client)
    return client

