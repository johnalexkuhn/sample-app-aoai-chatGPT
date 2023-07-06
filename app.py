"""Flask application serves React app and provides Open AI API access"""

import csv
import io
import json
import os
import logging
from typing import Any
from decimal import Decimal
import datetime
from flask import Flask, Response, request, jsonify
import requests
import openai
from dotenv import load_dotenv
from mysql.connector import MySQLConnection
from mysql.connector.pooling import PooledMySQLConnection
import mysql.connector


class CustomEncoder(json.JSONEncoder):
    """Define a custom JSON encoder to handle decimal and date values"""

    def default(self, o):
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, datetime.date):
            return str(o)
        return json.JSONEncoder.default(self, o)


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
# AZURE_OPENAI_SYSTEM_MESSAGE = os.environ.get(
#     "AZURE_OPENAI_SYSTEM_MESSAGE", "You are an AI assistant that helps people find information.")
AZURE_OPENAI_PREVIEW_API_VERSION = os.environ.get(
    "AZURE_OPENAI_PREVIEW_API_VERSION", "2023-06-01-preview")
AZURE_OPENAI_STREAM = os.environ.get("AZURE_OPENAI_STREAM", "true")
# Name of the model, e.g. 'gpt-35-turbo' or 'gpt-4'
AZURE_OPENAI_MODEL_NAME = os.environ.get(
    "AZURE_OPENAI_MODEL_NAME", "gpt-35-turbo")

# MySQL Database Connection Variables
MYSQL_HOST = os.environ.get("MYSQL_HOST")
MYSQL_USERNAME = os.environ.get("MYSQL_USERNAME")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE")

SHOULD_STREAM = True if AZURE_OPENAI_STREAM.lower() == "true" else False
CODE_BLOCK_DELIMITER = "```"
AZURE_OPENAI_SYSTEM_MESSAGE = """You are an AI assistant that helps people write mysql queries about
    membership data. My data table has the following columns: MEMBERSHIP_NUMBER,MEMBERSHIP_TERM_YEARS,
    ANNUAL_FEES,MEMBER_MARITAL_STATUS,MEMBER_GENDER,MEMBER_ANNUAL_INCOME,MEMBER_OCCUPATION_CD,
    MEMBERSHIP_PACKAGE,MEMBER_AGE_AT_ISSUE,ADDITIONAL_MEMBERS,PAYMENT_MODE,AGENT_CODE,MEMBERSHIP_STATUS,
    START_DATE,END_DATE. The table name is member_data. Put the count after the group field."""

def is_chat_model():
    """Determine if we are using Chat GPT model based on config settings"""
    if 'gpt-4' in AZURE_OPENAI_MODEL_NAME.lower() or AZURE_OPENAI_MODEL_NAME.lower() in [
        'gpt-35-turbo-4k', 'gpt-35-turbo-16k'
    ]:
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
                        # pylint: disable=line-too-long
                        "contentField": AZURE_SEARCH_CONTENT_COLUMNS.split("|") if AZURE_SEARCH_CONTENT_COLUMNS else [],
                        "titleField": AZURE_SEARCH_TITLE_COLUMN if AZURE_SEARCH_TITLE_COLUMN else None,
                        "urlField": AZURE_SEARCH_URL_COLUMN if AZURE_SEARCH_URL_COLUMN else None,
                        # pylint: disable=line-too-long
                        "filepathField": AZURE_SEARCH_FILENAME_COLUMN if AZURE_SEARCH_FILENAME_COLUMN else None
                    },
                    "inScope": True if AZURE_SEARCH_ENABLE_IN_DOMAIN.lower() == "true" else False,
                    "topNDocuments": AZURE_SEARCH_TOP_K,
                    # pylint: disable=line-too-long
                    "queryType": "semantic" if AZURE_SEARCH_USE_SEMANTIC_SEARCH.lower() == "true" else "simple",
                    # pylint: disable=line-too-long
                    "semanticConfiguration": AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG if AZURE_SEARCH_USE_SEMANTIC_SEARCH.lower() == "true" and AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG else "",
                    "roleInformation": AZURE_OPENAI_SYSTEM_MESSAGE
                }
            }
        ]
    }

    # pylint: disable=line-too-long
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
    # pylint: disable=line-too-long
    endpoint = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/openai/deployments/{AZURE_OPENAI_MODEL}/extensions/chat/completions?api-version={AZURE_OPENAI_PREVIEW_API_VERSION}"

    if not SHOULD_STREAM:
        api_response = requests.post(
            endpoint, headers=headers, json=body, timeout=1)
        status_code = api_response.status_code
        api_response_content = api_response.json()
        return Response(json.dumps(api_response_content).replace("\n", "\\n"), status=status_code)

    if req.method == "POST":
        return Response(stream_with_data(body, headers, endpoint), mimetype='text/event-stream')

    return Response(None, mimetype='text/event-stream')


def get_query_result(content) -> str:
    """ Extract query from content and execute on MySQL """
    query = extract_query(content)
    rows = get_result_set(query)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(rows)
    result = output.getvalue()

    return result


def original_stream_without_data(response):
    """Stream a response without search data"""
    response_text = ""
    for line in response:
        delta_text = line["choices"][0]["delta"].get('content')
        if delta_text and delta_text != "[DONE]":
            response_text += delta_text

        message_content = ""
        if contains_sql(response_text):
            message_content = get_query_result(response_text)
        else:
            message_content = response_text

        response_obj = {
            "id": line["id"],
            "model": line["model"],
            "created": line["created"],
            "object": line["object"],
            "choices": [{
                "messages": [{
                    "role": "assistant",
                    "content": message_content
                }]
            }]
        }

        yield json.dumps(response_obj).replace("\n", "\\n") + "\n"


def create_db_connection() -> (PooledMySQLConnection | MySQLConnection | Any):
    """Create a database connection for MySQL"""
    result = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USERNAME"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
        port=3306,
        ssl_disabled=False
    )

    return result


def get_result_set(sql_statement):
    """ Execute SQL and get result set """
    db_connection = create_db_connection()
    cursor = db_connection.cursor()
    cursor.execute(sql_statement)
    rows = cursor.fetchall()
    db_connection.close()

    return rows


def contains_sql(content: str) -> str:
    """ Determines if a string contains SQL statements """
    result = False
    if "SELECT" in content.upper():
        result = True

    return result


def extract_query(content) -> str:
    """ Extract a statement from a code block """
    result = ""
    start_index = content.find(CODE_BLOCK_DELIMITER)
    end_index = content.find(CODE_BLOCK_DELIMITER, start_index + 3, len(content))

    if start_index != -1 and end_index > start_index:
        result = content[start_index + 3:end_index - 1].replace("\n", " ").strip()

    return result


def item_generator(items: list):
    """Create a generator for a list of items"""
    for item in items:
        yield json.dumps(item).replace("\n", "\\n") + "\n"


def conversation_without_data(req):
    """Communicate with the API wihout search data"""
    openai.api_type = "azure"
    openai.api_base = f"https://{AZURE_OPENAI_RESOURCE}.openai.azure.com/"
    openai.api_version = "2023-03-15-preview"
    openai.api_key = AZURE_OPENAI_KEY

    chat_messages = []
    chat_messages.append({
        "role": "system",
        "content": AZURE_OPENAI_SYSTEM_MESSAGE
    })

    request_messages = req.json["messages"]

    for message in request_messages:
        chat_messages.append({
            "role": message["role"],
            "content": message["content"]
        })

    response = openai.ChatCompletion.create(
        engine=AZURE_OPENAI_MODEL,
        messages=chat_messages,
        temperature=float(AZURE_OPENAI_TEMPERATURE),
        max_tokens=int(AZURE_OPENAI_MAX_TOKENS),
        top_p=float(AZURE_OPENAI_TOP_P),
        stop=AZURE_OPENAI_STOP_SEQUENCE.split(
            "|") if AZURE_OPENAI_STOP_SEQUENCE else None
    )

    content = response.choices[0].message.content
    if contains_sql(content=content):
        query_result = get_query_result(response.choices[0].message.content)
        if query_result != "":
            content = f"Here are the results of your inquiry:\n\n```{query_result}```\n"
        else:
            content = f"Unabled to run query, content was {response.choices[0].message.content}"

    response_obj = {
        "id": response,
        "model": response.model,
        "created": response.created,
        "object": response.object,
        "choices": [{
            "messages": [{
                "role": "assistant",
                "content": content
            }]
        }]
    }

    response_list = []
    response_list.append(response_obj)

    if request.method == "POST":
        return Response(item_generator(response_list), mimetype='text/event-stream')

    return Response(None, mimetype='text/event-stream')


@app.route("/conversation", methods=["GET", "POST"])
def conversation():
    """Handle HTTP requests to the conversation route"""
    try:
        use_data = should_use_data()
        if use_data:
            return conversation_with_data(request)
        return conversation_without_data(request)
    # pylint: disable=broad-exception-caught
    except Exception as err:
        logging.exception("Exception in /conversation")
        return jsonify({"error": str(err)}), 500


if __name__ == "__main__":
    app.run()
