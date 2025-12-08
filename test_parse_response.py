from google import genai

from parse_response import process_model_response


def test_simple_function_call():
    schemas: list[genai.types.FunctionDeclaration] = [
        genai.types.FunctionDeclaration(
            name="get_weather",
            description="Get weather",
            parameters=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                properties={"city": genai.types.Schema(type=genai.types.Type.STRING)},
                required=["city"],
            ),
        )
    ]

    response = "[get_weather(city='Boston')]"
    result = process_model_response(response, schemas)

    assert result["type"] == "function_call"
    assert result["valid"] is True
    assert len(result["content"]) == 1
    assert result["content"][0]["function"] == "get_weather"
    assert result["content"][0]["parameters"]["city"] == "Boston"


def test_multiple_function_calls():
    schemas: list[genai.types.FunctionDeclaration] = [
        genai.types.FunctionDeclaration(
            name="func1",
            description="Function 1",
            parameters=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                properties={"a": genai.types.Schema(type=genai.types.Type.INTEGER)},
                required=["a"],
            ),
        ),
        genai.types.FunctionDeclaration(
            name="func2",
            description="Function 2",
            parameters=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                properties={"b": genai.types.Schema(type=genai.types.Type.STRING)},
                required=["b"],
            ),
        ),
    ]

    response = "[func1(a=1), func2(b='test')]"
    result = process_model_response(response, schemas)

    assert result["type"] == "function_call"
    assert result["valid"] is True
    assert len(result["content"]) == 2
    assert result["content"][0]["function"] == "func1"
    assert result["content"][1]["function"] == "func2"


def test_string_with_comma():
    schemas: list[genai.types.FunctionDeclaration] = [
        genai.types.FunctionDeclaration(
            name="send_message",
            description="Send a message",
            parameters=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                properties={"text": genai.types.Schema(type=genai.types.Type.STRING)},
                required=["text"],
            ),
        )
    ]

    response = "[send_message(text='Hello, world!')]"
    result = process_model_response(response, schemas)

    assert result["type"] == "function_call"
    assert result["valid"] is True
    assert result["content"][0]["parameters"]["text"] == "Hello, world!"


def test_escaped_quotes():
    schemas: list[genai.types.FunctionDeclaration] = [
        genai.types.FunctionDeclaration(
            name="echo",
            description="Echo text",
            parameters=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                properties={"text": genai.types.Schema(type=genai.types.Type.STRING)},
                required=["text"],
            ),
        )
    ]

    response = """[echo(text='She said "hello"')]"""
    result = process_model_response(response, schemas)

    assert result["type"] == "function_call"
    assert result["valid"] is True
    assert result["content"][0]["parameters"]["text"] == 'She said "hello"'


def test_list_parameter():
    schemas: list[genai.types.FunctionDeclaration] = [
        genai.types.FunctionDeclaration(
            name="sum_numbers",
            description="Sum numbers",
            parameters=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                properties={"numbers": genai.types.Schema(type=genai.types.Type.ARRAY)},
                required=["numbers"],
            ),
        )
    ]

    response = "[sum_numbers(numbers=[1, 2, 3, 4])]"
    result = process_model_response(response, schemas)

    assert result["type"] == "function_call"
    assert result["valid"] is True
    assert result["content"][0]["parameters"]["numbers"] == [1, 2, 3, 4]


def test_dict_parameter():
    schemas: list[genai.types.FunctionDeclaration] = [
        genai.types.FunctionDeclaration(
            name="create_user",
            description="Create user",
            parameters=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                properties={"data": genai.types.Schema(type=genai.types.Type.OBJECT)},
                required=["data"],
            ),
        )
    ]

    response = "[create_user(data={'name': 'John', 'age': 30})]"
    result = process_model_response(response, schemas)

    assert result["type"] == "function_call"
    assert result["valid"] is True
    assert result["content"][0]["parameters"]["data"] == {"name": "John", "age": 30}


def test_text_response():
    schemas = []
    response = "This is just a regular text response."
    result = process_model_response(response, schemas)

    assert result["type"] == "text"
    assert result["valid"] is True
    assert result["content"] == response


def test_missing_required_parameter():
    schemas: list[genai.types.FunctionDeclaration] = [
        genai.types.FunctionDeclaration(
            name="get_weather",
            description="Get weather",
            parameters=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                properties={
                    "city": genai.types.Schema(type=genai.types.Type.STRING),
                    "units": genai.types.Schema(type=genai.types.Type.STRING),
                },
                required=["city"],
            ),
        )
    ]

    response = "[get_weather(units='celsius')]"
    result = process_model_response(response, schemas)

    assert result["type"] == "function_call"
    assert result["valid"] is False
    assert len(result["errors"]) == 1
    assert "Missing required parameter 'city'" in result["errors"][0]


def test_wrong_parameter_type():
    schemas: list[genai.types.FunctionDeclaration] = [
        genai.types.FunctionDeclaration(
            name="set_volume",
            description="Set volume",
            parameters=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                properties={"level": genai.types.Schema(type=genai.types.Type.INTEGER)},
                required=["level"],
            ),
        )
    ]

    response = "[set_volume(level='high')]"
    result = process_model_response(response, schemas)

    assert result["type"] == "function_call"
    assert result["valid"] is False
    assert "Invalid type" in result["errors"][0]


def test_unknown_function():
    schemas: list[genai.types.FunctionDeclaration] = [
        genai.types.FunctionDeclaration(
            name="known_func",
            description="Known",
            parameters=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                properties={},
                required=[],
            ),
        )
    ]

    response = "[unknown_func(param='value')]"
    result = process_model_response(response, schemas)

    assert result["type"] == "function_call"
    assert result["valid"] is False
    assert "Unknown function" in result["errors"][0]


def test_malformed_syntax():
    schemas = []
    response = "[not_a_valid_call"
    result = process_model_response(response, schemas)

    assert result["type"] == "error"
    assert result["valid"] is False


def test_empty_brackets():
    schemas = []
    response = "[]"
    result = process_model_response(response, schemas)

    assert result["type"] == "error"
    assert result["valid"] is False
    assert "No valid function calls found" in result["errors"][0]


def test_whitespace_handling():
    schemas: list[genai.types.FunctionDeclaration] = [
        genai.types.FunctionDeclaration(
            name="test_func",
            description="Test",
            parameters=genai.types.Schema(
                type=genai.types.Type.OBJECT,
                properties={"param": genai.types.Schema(type=genai.types.Type.STRING)},
                required=["param"],
            ),
        )
    ]

    response = "  [  test_func(  param = 'value'  )  ]  "
    result = process_model_response(response, schemas)

    assert result["type"] == "function_call"
    assert result["valid"] is True
    assert result["content"][0]["parameters"]["param"] == "value"
