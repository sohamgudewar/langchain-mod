# LangChain Models

This repository contains small learning scripts for calling hosted language models and constructing LangChain prompts. Most scripts require provider credentials and make external API calls; they are examples rather than a reusable application or tested library.

## Features
- Prompt templates and message construction
- An in-process chat-history example
- OpenAI, Anthropic, Google, and Hugging Face client examples
- A small Streamlit prompt UI

There is no automated test suite. Some examples may require model-name or API updates as providers change, and `Prompts/langchain-prompts/prompt_template.py` currently constructs the wrong class for a prompt template.

## Tech Stack
- Python
- LangChain
- Provider integration packages listed in `req.txt`
