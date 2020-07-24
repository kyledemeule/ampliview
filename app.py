# app.py
from flask import Flask, request, jsonify, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    review_text = request.form.get('review_text')
    if review_text:
        review_lines = [process_line(l) for l in review_text.split(".")]
        return jsonify({
            "review_html": render_template('review.html', review_lines=review_lines),
            "helpful_prediction": 0.93
        })
    else:
        return jsonify({
            "ERROR": "no name found, please send a name."
        })

# A welcome message to test our server
@app.route('/')
def index():
    return render_template('index.html')

def process_line(line):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(line)
    if vs["compound"] >= 0.05:
        line_sentiment = "positive"
    elif vs["compound"] <= -0.05:
        line_sentiment = "negative"
    else:
        line_sentiment = "neutral"
    return {"line": line, "line_sentiment": line_sentiment}

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
