import os
import logging

from flask import Flask, render_template, request
from dotenv import load_dotenv

from rag.rag import RagPipeline
from llm.chat_gpt_client import chat_gpt_client
from database.postgresql_client import postgresql_client

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

rag_pipeline = RagPipeline(chat_gpt_client, postgresql_client)
score_threshold = float(os.getenv("SCORE_THRESHOLD"))


@app.route('/', methods=['GET', 'POST'])
def index_action():
    if request.method == 'POST':
        query = request.form.get("query")
        if not query:
            return render_template('index.html', error="Укажите текст запроса")

        rag_response = rag_pipeline.run(query, 3, None, score_threshold)

        if rag_response.error is not None:
            return render_template('index.html', error=rag_response.error)

        return render_template('index.html', query=query, answer=rag_response.answer,
                               context=rag_response.context, request=rag_response.request)

    return render_template('index.html')


@app.route('/feedback/<request_id>', methods=['POST'])
def feedback_action(request_id: str):
    feedback = request.form.get("feedback")
    feedback_comment = request.form.get("comment")

    feedback_request = postgresql_client.find_request(request_id)
    if feedback_request is None:
        return render_template('not_found.html')

    feedback_request.feedback = feedback == 'positive'
    feedback_request.feedback_comment = feedback_comment

    postgresql_client.update_request(feedback_request)

    return render_template('feedback.html')


if __name__ == '__main__':
    app.run(debug=True)
