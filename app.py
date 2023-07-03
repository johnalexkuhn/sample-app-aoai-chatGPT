"""Flask application serves React app and provides Open AI API access"""

import json
import os
import logging
import requests
import openai
from flask import Flask, Response, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def static_file(path):
    """Handle requests to the app root and deliver static content"""
    return app.send_static_file(path)


# ACS Integration Settings
AZURE_SEARCH_SERVICE = os.environ.get("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
AZURE_SEARCH_USE_SEMANTIC_SEARCH = os.environ.get(
    "AZURE_SEARCH_USE_SEMANTIC_SEARCH", "false")
AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG = os.environ.get(
    "AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG", "default")
AZURE_SEARCH_TOP_K = os.environ.get("AZURE_SEARCH_TOP_K", 5)
AZURE_SEARCH_ENABLE_IN_DOMAIN = os.environ.get(
    "AZURE_SEARCH_ENABLE_IN_DOMAIN", "true")
AZURE_SEARCH_CONTENT_COLUMNS = os.environ.get("AZURE_SEARCH_CONTENT_COLUMNS")
AZURE_SEARCH_FILENAME_COLUMN = os.environ.get("AZURE_SEARCH_FILENAME_COLUMN")
AZURE_SEARCH_TITLE_COLUMN = os.environ.get("AZURE_SEARCH_TITLE_COLUMN")
AZURE_SEARCH_URL_COLUMN = os.environ.get("AZURE_SEARCH_URL_COLUMN")

# AOAI Integration Settings
AZURE_OPENAI_RESOURCE = os.environ.get("AZURE_OPENAI_RESOURCE")
AZURE_OPENAI_MODEL = os.environ.get("AZURE_OPENAI_MODEL")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE", 0)
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P", 1.0)
AZURE_OPENAI_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS", 1000)
AZURE_OPENAI_STOP_SEQUENCE = os.environ.get("AZURE_OPENAI_STOP_SEQUENCE")
AZURE_OPENAI_SYSTEM_MESSAGE = os.environ.get(
    "AZURE_OPENAI_SYSTEM_MESSAGE", "You are an AI assistant that helps people find information.")
AZURE_OPENAI_PREVIEW_API_VERSION = os.environ.get(
    "AZURE_OPENAI_PREVIEW_API_VERSION", "2023-06-01-preview")
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM", "true")
# Name of the model, e.g. 'gpt-35-turbo' or 'gpt-4'
AZURE_OPENAI_MODEL_NAME = os.environ.get(
    "AZURE_OPENAI_MODEL_NAME", "gpt-35-turbo")

SHOULD_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False


def is_chat_model():
    """Determine if we are using Chat GPT model based on config settings"""
    if 'gpt-4' in AZURE_OPENAI_MODEL_NAME.lower() or AZURE_OPENAI_MODEL_NAME.lower() in ['gpt-35-turbo-4k', 'gpt-35-turbo-16k']:
        return True
    return False


def should_use_data():
    """Determine if the app should use search data from Azure Storage based on config settings"""
    if AZURE_SEARCH_SERVICE and AZURE_SEARCH_INDEX and AZURE_SEARCH_KEY:
        return True
    return False


def prepare_body_headers_with_data(req):
    """Create request body with data from last message thread"""
    request_messages = req.json["messages"]

    body = {
        "messages": request_messages,
        "temperature": float(AZURE_OPENAI_TEMPERATURE),
        "max_tokens": int(AZURE_OPENAI_MAX_TOKENS),
        "top_p": float(AZURE_OPENAI_TOP_P),
        "stop": AZURE_OPENAI_STOP_SEQUENCE.split("|") if AZURE_OPENAI_STOP_SEQUENCE else None,
        "stream": SHOULD_STREAM,
        "dataSources": [
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": f"https://{AZURE_SEARCH_SERVICE}.search.windows.net",
                    "key": AZURE_SEARCH_KEY,
                    "indexName": AZURE_SEARCH_INDEX,
                    "fieldsMapping": {
                        "contentField": AZURE_SEARCH_CONTENT_COLUMNS.split("|") if AZURE_SEARCH_CONTENT_COLUMNS else [],
                        "titleField": AZURE_SEARCH_TITLE_COLUMN if AZURE_SEARCH_TITLE_COLUMN else None,
                        "urlField": AZURE_SEARCH_URL_COLUMN if AZURE_SEARCH_URL_COLUMN else None,
                        "filepathField": AZURE_SEARCH_FILENAME_COLUMN if AZURE_SEARCH_FILENAME_COLUMN else None
                    },
                    "inScope": True if AZURE_SEARCH_ENABLE_IN_DOMAIN.lower() == "true" else False,
                    "topNDocuments": AZURE_SEARCH_TOP_K,
                    "queryType": "semantic" if AZURE_SEARCH_USE_SEMANTIC_SEARCH.lower() == "true" else "simple",
                    "semanticConfiguration": AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG if AZURE_SEARCH_USE_SEMANTIC_SEARCH.lower() == "true" and AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG else "",
                    "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE
                }
            }
        ]
    }

    chatgpt_url = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/openai/deployments/{AZURE_OPENAI_MODEL}"
    if is_chat_model():
        chatgpt_url += "/chat/completions?api-version=2023-03-15-preview"
    else:
        chatgpt_url += "/completions?api-version=2023-03-15-preview"

    headers = {
        'Content-Type': 'application/json',
        'api-key': AZURE_OPENAI_KEY,
        'chatgpt_url': chatgpt_url,
        'chatgpt_key': AZURE_OPENAI_KEY,
        "x-ms-useragent": "GitHubSampleWebApp/PublicAPI/1.0.0"
    }

    return body, headers


def stream_with_data(body, headers, endpoint):
    """Stream response with search data"""
    session = requests.Session()

    response = {
        "id": "",
        "model": "",
        "created": 0,
        "object": "",
        "choices": [{
            "messages": []
        }]
    }

    try:
        with session.post(endpoint, json=body, headers=headers, stream=True) as stream_response:
            for line in stream_response.iter_lines(chunk_size=10):
                if line:
                    line_json = json.loads(
                        line.lstrip(b'data:').decode('utf-8'))
                    if 'error' in line_json:
                        yield json.dumps(line_json).replace("\n", "\\n") + "\n"
                    response["id"] = line_json["id"]
                    response["model"] = line_json["model"]
                    response["created"] = line_json["created"]
                    response["object"] = line_json["object"]

                    role = line_json["choices"][0]["messages"][0]["delta"].get(
                        "role")
                    if role == "tool":
                        response["choices"][0]["messages"].append(
                            line_json["choices"][0]["messages"][0]["delta"])
                    elif role == "assistant":
                        response["choices"][0]["messages"].append({
                            "role": "assistant",
                            "content": ""
                        })
                    else:
                        delta_text = line_json["choices"][0]["messages"][0]["delta"]["content"]
                        if delta_text != "[DONE]":
                            response["choices"][0]["messages"][1]["content"] += delta_text

                    yield json.dumps(response).replace("\n", "\\n") + "\n"
    # pylint: disable=broad-exception-caught
    except Exception as err:
        yield json.dumps({"error": str(err)}).replace("\n", "\\n") + "\n"


def conversation_with_data(req):
    """Create a conversation with data from event stream"""
    body, headers = prepare_body_headers_with_data(req)
    endpoint = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/openai/deployments/{AZURE_OPENAI_MODEL}/extensions/chat/completions?api-version={AZURE_OPENAI_PREVIEW_API_VERSION}"

    if not SHOULD_STREAM:
        api_response = requests.post(
            endpoint, headers=headers, json=body, timeout=1)
        status_code = api_response.status_code
        api_response_content = api_response.json()
        return Response(json.dumps(api_response_content).replace("\n", "\\n"), status=status_code)

    if req.method == "POST":
        return Response(stream_with_data(body, headers, endpoint), mimetype='text/event-stream')
    else:
        return Response(None, mimetype='text/event-stream')


def stream_without_data(response):
    """Stream a response without search data"""
    response_text = ""
    for line in response:
        delta_text = line["choices"][0]["delta"].get('content')
        if delta_text and delta_text != "[DONE]":
            response_text += delta_text

        response_obj = {
            "id": line["id"],
            "model": line["model"],
            "created": line["created"],
            "object": line["object"],
            "choices": [{
                "messages": [{
                    "role": "assistant",
                    "content": response_text
                }]
            }]
        }

        yield json.dumps(response_obj).replace("\n", "\\n") + "\n"


def conversation_without_data(req):
    """Communicate with the API wihout search data"""
    openai.api_type = "azure"
    openai.api_base = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"
    openai.api_version = "2023-03-15-preview"
    openai.api_key = AZURE_OPENAI_KEY

    request_messages = req.json["messages"]

    messages = [
        {
            "role": "system",
            "content": AZURE_OPENAI_SYSTEM_MESSAGE
        }
    ]

    for message in request_messages:
        messages.append({
            "role": message["role"],
            "content": message["content"]
        })

    response = openai.ChatCompletion.create(
        engine=AZURE_OPENAI_MODEL,
        messages=messages,
        temperature=float(AZURE_OPENAI_TEMPERATURE),
        max_tokens=int(AZURE_OPENAI_MAX_TOKENS),
        top_p=float(AZURE_OPENAI_TOP_P),
        stop=AZURE_OPENAI_STOP_SEQUENCE.split(
            "|") if AZURE_OPENAI_STOP_SEQUENCE else None,
        stream=SHOULD_STREAM
    )

    if not SHOULD_STREAM:
        response_obj = {
            "id": response,
            "model": response.model,
            "created": response.created,
            "object": response.object,
            "choices": [{
                "messages": [{
                    "role": "assistant",
                    "content": response.choices[0].message.content
                }]
            }]
        }

        return jsonify(response_obj), 200

    if request.method == "POST":
        return Response(stream_without_data(response), mimetype='text/event-stream')
    else:
        return Response(None, mimetype='text/event-stream')


@app.route("/conversation", methods=["GET", "POST"])
def conversation():
    """Handle HTTP requests to the conversation route"""
    try:
        use_data = should_use_data()
        if use_data:
            return conversation_with_data(request)
        else:
            return conversation_without_data(request)
    # pylint: disable=broad-exception-caught
    except Exception as err:
        logging.exception("Exception in /conversation")
        return jsonify({"error": str(err)}), 500


if __name__ == "__main__":
    app.run()
