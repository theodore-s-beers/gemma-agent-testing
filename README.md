# gemma-agent-testing

This isn't even close to working right now. Spun out into a separate repo for future reference/posterity.

## Steps to try

1. Create a bug in `calculator/pkg/calculator.py` by changing the precedence of the `+` operator to `3`.
2. Verify that it's broken. This should now return `20` instead of the correct `17`:
   ```sh
   uv run calculator/main.py '3 + 7 * 2'
   ```
3. Ask the agent to find and fix the bug (it won't succeed currently):
   ```sh
   uv run main.py "fix the bug: 3 + 7 * 2 shouldn't be 20"  --verbose
   ```

## Ongoing work

Most of what I added or changed is in the following modules:

- `call_function.py`: made function calling more manual
- `main.py`: switched to handling everything manually, since Gemma 3 allows neither tools nor system instructions
- `parse_response.py`: entirely new module to parse LLM responses, detect function calls, etc.
- `prompts.py`: new, more verbose system prompt
- `test_parse_response.py`: tests for the new response parser
