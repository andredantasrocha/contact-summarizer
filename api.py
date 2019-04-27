import logging
from flask import Flask, request, jsonify

from summarization.summarizer import Summarizer

api = Flask(__name__)
logging.basicConfig(level=logging.ERROR)


@api.route('/contact-summarizer/summarize', methods=['POST'])
def summarize():
    result = Summarizer().execute(request.get_data(as_text=True))
    return jsonify(result)


if __name__ == "__main__":
    api.run(debug=True, host='0.0.0.0', use_reloader=False)
