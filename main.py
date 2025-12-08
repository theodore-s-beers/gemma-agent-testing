import argparse
import os
import sys

from dotenv import load_dotenv
from google import genai

from call_function import call_function
from config import MAX_ITERS
from parse_response import process_model_response
from prompts import available_functions, new_system_prompt


def main():
    parser = argparse.ArgumentParser(description="AI Code Assistant")
    parser.add_argument("user_prompt", type=str, help="Prompt to send to Gemini")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    messages = [
        genai.types.Content(
            role="user", parts=[genai.types.Part(text=new_system_prompt)]
        ),
        genai.types.Content(
            role="user", parts=[genai.types.Part(text=args.user_prompt)]
        ),
    ]
    if args.verbose:
        print(f"User prompt: {args.user_prompt}\n")

    iters = 0
    while True:
        iters += 1
        if iters > MAX_ITERS:
            print(f"Maximum iterations ({MAX_ITERS}) reached.")
            sys.exit(1)

        try:
            final_response = generate_content(client, messages, args.verbose)
            if final_response:
                print("Final response:")
                print(final_response)
                break
        except Exception as e:
            print(f"Error in generate_content: {e}")
            raise


def generate_content(client: genai.Client, messages, verbose):
    response = client.models.generate_content(model="gemma-3-27b-it", contents=messages)
    if not response.text or not response.usage_metadata:
        raise RuntimeError("Gemini API response appears to be malformed")

    if verbose:
        print("Prompt tokens:", response.usage_metadata.prompt_token_count)
        print("Response tokens:", response.usage_metadata.candidates_token_count)

    response_text = response.text.strip()
    if verbose:
        print(f"\nModel response:\n{response_text}\n")

    # Add model's response to messages
    messages.append(
        genai.types.Content(role="model", parts=[genai.types.Part(text=response_text)])
    )

    parsed_response = process_model_response(response_text, available_functions)
    if verbose:
        print(f"Parsed response type: {parsed_response['type']}")
        if parsed_response["errors"]:
            print(f"Parsing errors: {parsed_response['errors']}")

    if parsed_response["type"] == "text":
        return parsed_response["content"]  # Final answer

    if parsed_response["type"] == "error":
        error_message = "I encountered errors parsing your function calls:\n"
        error_message += "\n".join(f"- {err}" for err in parsed_response["errors"])
        error_message += "\n\nPlease format your function calls correctly."

        if verbose:
            print(f"Sending error feedback to model:\n{error_message}\n")

        messages.append(
            genai.types.Content(
                role="user", parts=[genai.types.Part(text=error_message)]
            )
        )
        return None  # Continue loop

    # Otherwise we're dealing with function calls; check if valid
    if not parsed_response["valid"]:
        error_message = "Function call validation failed:\n"
        error_message += "\n".join(f"- {err}" for err in parsed_response["errors"])
        error_message += "\n\nPlease correct the function calls."

        if verbose:
            print(f"Sending validation errors to model:\n{error_message}\n")

        messages.append(
            genai.types.Content(
                role="user", parts=[genai.types.Part(text=error_message)]
            )
        )
        return None  # Continue loop

    function_calls = parsed_response["content"]
    if verbose:
        print(f"Executing {len(function_calls)} function call(s)")

    function_results = []
    for func_call in function_calls:
        func_name = func_call["function"]
        func_params = func_call["parameters"]
        if verbose:
            print(f"Calling function: {func_name} with params: {func_params}")

        try:
            result = call_function(func_name, func_params, verbose)
            function_results.append({"name": func_name, "result": result})
            if verbose:
                print(f"-> Result: {result}\n")
        except Exception as e:
            if verbose:
                print(f"-> Error: {e}\n")
            function_results.append({"name": func_name, "error": str(e)})

    if not function_results:
        raise Exception("No function results generated; exiting.")

    results_text = format_function_results(function_results)
    if verbose:
        print(f"Sending function results back to model:\n{results_text}\n")

    messages.append(
        genai.types.Content(role="user", parts=[genai.types.Part(text=results_text)])
    )
    return None  # Continue loop


def format_function_results(function_results):
    results_text = "Function execution results:\n\n"
    for result in function_results:
        if "error" in result:
            results_text += f"❌ Function '{result['name']}' failed with error:\n{result['error']}\n\n"
        else:
            results_text += (
                f"✓ Function '{result['name']}' returned:\n{result['result']}\n\n"
            )
    return results_text


if __name__ == "__main__":
    main()
