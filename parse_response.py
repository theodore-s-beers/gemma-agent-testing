import ast
import re
from typing import Any, TypedDict

from google import genai


class ParsedFunctionCall(TypedDict):
    function: str
    parameters: dict[str, Any]


class ParsedResponse(TypedDict):
    type: str  # "function_call", "text", or "error"
    content: Any  # str, list[ParsedFunctionCall], or None
    valid: bool  # for function calls
    errors: list[str]  # list of error messages


def process_model_response(
    response: str, function_schemas: list[genai.types.FunctionDeclaration]
) -> ParsedResponse:
    if not contains_function_call(response):
        return {"type": "text", "content": response, "valid": True, "errors": []}

    try:
        call_strings = extract_function_calls(response)
        if not call_strings:
            return {
                "type": "error",
                "content": None,
                "valid": False,
                "errors": ["No valid function calls found"],
            }

        parsed_calls: list[ParsedFunctionCall] = []
        errors: list[str] = []

        for call_str in call_strings:
            try:
                call = parse_function_call(call_str)
                is_valid, error = validate_function_call(call, function_schemas)

                if not is_valid:
                    errors.append(error)

                parsed_calls.append(call)
            except Exception as e:
                errors.append(f"Parse error for '{call_str}': {str(e)}")

        return {
            "type": "function_call",
            "content": parsed_calls,
            "valid": len(errors) == 0,
            "errors": errors,
        }
    except Exception as e:
        return {
            "type": "error",
            "content": None,
            "valid": False,
            "errors": [f"Failed to process response: {str(e)}"],
        }


def contains_function_call(response: str) -> bool:
    response = response.strip()
    # More detailed validation happens in extract_function_calls
    return response.startswith("[")


def extract_function_calls(response: str) -> list[str]:
    # Extract content between outer brackets
    match = re.match(r"^\s*\[(.*)\]\s*$", response.strip())
    if not match:
        return []

    inner_content = match.group(1)

    # Split by "), " to handle multiple function calls
    # But be careful with nested structures
    calls = []
    depth = 0
    current_call = []

    i = 0
    while i < len(inner_content):
        char = inner_content[i]

        if char == "(":
            depth += 1
            current_call.append(char)
        elif char == ")":
            depth -= 1
            current_call.append(char)

            # If we've closed all parentheses, we've completed a call
            if depth == 0:
                call_str = "".join(current_call).strip()
                if call_str:
                    calls.append(call_str)
                current_call = []
                # Skip comma and optional space after
                if i + 1 < len(inner_content) and inner_content[i + 1] == ",":
                    i += 1
                    while i + 1 < len(inner_content) and inner_content[i + 1] == " ":
                        i += 1
        else:
            if depth > 0 or char not in [",", " "] or current_call:
                current_call.append(char)

        i += 1

    return calls


def parse_function_call(call_str: str) -> ParsedFunctionCall:
    # Extract function name and parameters
    match = re.match(r"(\w+)\((.*)\)$", call_str.strip())
    if not match:
        raise ValueError(f"Invalid function call format: {call_str}")

    func_name = match.group(1)
    params_str = match.group(2).strip()

    # Parse parameters
    params = {}
    if params_str:
        params = parse_parameters(params_str)

    return {"function": func_name, "parameters": params}


def parse_parameters(params_str: str) -> dict[str, Any]:
    params = {}

    # Split by comma, but respect nested structures
    parts = []
    current = []
    depth = 0
    in_string = False
    string_char = None
    escape_next = False

    for char in params_str:
        # Handle escape sequences in strings
        if escape_next:
            current.append(char)
            escape_next = False
            continue

        if char == "\\" and in_string:
            current.append(char)
            escape_next = True
            continue

        if char in ['"', "'"] and (not in_string or char == string_char):
            in_string = not in_string
            string_char = char if in_string else None

        if not in_string:
            if char in ["(", "[", "{"]:
                depth += 1
            elif char in [")", "]", "}"]:
                depth -= 1
            elif char == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue

        current.append(char)

    if current:
        parts.append("".join(current).strip())

    # Parse each key=value pair
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid parameter format: {part}")

        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Convert value to appropriate Python type
        params[key] = parse_value(value)

    return params


def parse_value(value_str: str) -> Any:
    """Convert string representation to Python value"""
    value_str = value_str.strip()

    # Try to evaluate as Python literal
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        # If it fails, treat as string (remove quotes if present)
        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
            return value_str[1:-1]
        return value_str


def validate_function_call(
    call: ParsedFunctionCall, function_schemas: list[genai.types.FunctionDeclaration]
) -> tuple[bool, str]:
    """
    Returns (is_valid, error_message)
    """
    func_name = call["function"]
    params = call["parameters"]

    # Find the schema
    schema = next((f for f in function_schemas if f.name == func_name), None)
    if not schema:
        return False, f"Unknown function: {func_name}"

    required = schema.parameters.required if schema.parameters else []
    if required is None:
        required = []

    properties = schema.parameters.properties if schema.parameters else {}
    if properties is None:
        properties = {}

    # Check required parameters
    for req_param in required:
        if req_param not in params:
            return (
                False,
                f"Missing required parameter '{req_param}' for function '{func_name}'",
            )

    # Check parameter types
    for param_name, param_value in params.items():
        if param_name not in properties:
            return False, f"Unknown parameter '{param_name}' for function '{func_name}'"

        expected_type = properties[param_name].type
        if expected_type and not validate_type(param_value, expected_type):
            return (
                False,
                f"Invalid type for '{param_name}' in function '{func_name}': expected {expected_type.name}",
            )

    return True, ""


def validate_type(value: Any, expected_type: genai.types.Type) -> bool:
    type_map = {
        genai.types.Type.TYPE_UNSPECIFIED: object,
        genai.types.Type.STRING: str,
        genai.types.Type.NUMBER: (int, float),
        genai.types.Type.INTEGER: int,
        genai.types.Type.BOOLEAN: bool,
        genai.types.Type.ARRAY: list,
        genai.types.Type.OBJECT: dict,
    }

    expected_python_type = type_map.get(expected_type)
    if expected_python_type is None:
        return True  # Unknown type, allow it

    return isinstance(value, expected_python_type)
