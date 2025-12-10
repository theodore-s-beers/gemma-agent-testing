# gemma-agent-testing

This repo represents an effort to adapt the project from the [AI agent course](https://www.boot.dev/courses/build-ai-agent-python) on Boot.dev to use Google's [Gemma 3 27B](https://ai.google.dev/gemma/docs/core) model – which still has a generous free-tier allowance on the Gemini API as of mid December 2025.

Boot.dev's AI agent project was originally designed to use Gemini Flash models – 2.0, then 2.5 – which were available on the free tier of the API with request limits more than sufficient for educational purposes. This recently changed. Gemini 2.0 Flash was removed from the free tier entirely, and 2.5 Flash now has rate limits so low as to make any meaningful work impractical. (We're seeing a limit of 20 requests per day.)

The current recommendation for Boot.dev students is to set up paid accounts with Google Cloud and continue to use `gemini-2.5-flash`. The cost per million tokens is low enough that you could budget \$1 or \$2 to go through the entire AI agent course. We're in a funny situation where paid APIs – Gemini and others – are so cheap that it would seem logical for them to offer some free-tier access; but on the other hand, abuse of free APIs is evidently so rampant that providers feel unable to give away much of anything. Google would rather bill you a few cents a month!

Be that as it may. Before telling students to sign up for paid accounts, I wanted to see if I could adapt the course project to use Gemma 3 27B, the best model on the Gemini API that still has generous free-tier limits. This is made difficult by the Gemma models' lack of support for two features that are critical to agentic AI:

1. **Tool use:** the ability to call external functions, APIs, or other tools from within the LLM's response.
2. **System instructions:** the ability to provide the LLM with high-level instructions that guide its behavior throughout the conversation.

To try to overcome these obstacles, I've made a number of changes to the original Boot.dev project code. Most notably, I implemented a manual function-calling mechanism, which parses each LLM response for an indication that the model wants to call a function; extracts the relevant information; executes the function; and feeds the result back to the model in the next prompt. (See [`parse_response.py`](parse_response.py).)

With help from a Boot.dev community member (thanks ML Zebra!), I also changed the prompting strategy, so that a highly verbose and detailed "system prompt" is sent to the model as the first message in each conversation. We basically explain to the model as clearly as possible how to plan its work on the given problem; how to request function calls; what functions are available to it; etc. Much of the system prompt is concerned with impressing upon the model the importance of keeping function call requests in the correct form. Such is the infuriating dance of prompt engineering. (See [`prompts.py`](prompts.py).)

We also added the option to use a other LLMs (LM Studio, llama.cpp, OpenRouter). This was possible due to the above changes which imply that we essentially only need a 'regular' LLM for this code to work, not one that can handle tools. This feature is more experimental since the prompt (see above) may need to be tweaked for each model. The main implementations are in [`local_genai.py`](lib/local_genai.py) and [`local_openrouter.py`](lib/local_openrouter.py). Alternatively, the repo could also be changed over to use the OpenAI API (since Google supports it, though recommends their own).

**Anyway, does it work?** The answer is... kind of! If you're taking the Boot.dev AI agent course and setting up a paid account with Google is not an option for you, then you can try to use Gemma 3 27B with the strategies followed in this repo. I hope it will be helpful to someone.

## Steps to try

1. Create a copy of `.env.example` named `.env` and paste in your API key and/or set `LLM_PROVIDER` to your preferred provider (with suitable values for corresponding variables).
1. Create a bug in `calculator/pkg/calculator.py` by changing the precedence of the `+` operator to `3`.
1. Verify that it's broken. This should now return `20` instead of the correct `17`:
   ```sh
   uv run calculator/main.py '3 + 7 * 2'
   ```
1. Ask the agent to find and fix the bug (it may or may not succeed in any given run):
   ```sh
   uv run main.py "fix the bug: 3 + 7 * 2 shouldn't be 20" --verbose
   ```

## Key files

Most of what I added or changed is in the following modules:

- [`call_function.py`](call_function.py): made function calling more manual
- [`main.py`](main.py): switched to handling everything manually, since Gemma 3 allows neither tools nor system instructions
- [`parse_response.py`](parse_response.py): entirely new module to parse LLM responses, detect function calls, etc.
- [`prompts.py`](prompts.py): new, _much_ more verbose system prompt
- [`test_parse_response.py`](test_parse_response.py): tests for the new response parser
- [`local_genai.py`](lib/local_genai.py): implementation of a local class that mimics Google's `genai` behavior
- [`llm_config.py`](llm_config.py): selects whether to use Google's `genai` or the local class
