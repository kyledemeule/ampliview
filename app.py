# app.py
from flask import Flask, request, jsonify, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
app = Flask(__name__)

# for tokenizing reviews
import nltk
import nltk.tokenize.punkt as pkt
nltk.download('punkt')
class CustomLanguageVars(pkt.PunktLanguageVars):
    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""
sentence_tokenizer = pkt.PunktSentenceTokenizer(lang_vars=CustomLanguageVars())

@app.route('/analyze', methods=['POST'])
def analyze():
    review_text = request.form.get('review_text')
    if review_text:
        review_lines = [process_line(l) for l in sentence_tokenizer.tokenize(review_text)]
        print(review_lines)
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
    return {
        "line": line,
        "line_sentiment": line_sentiment,
        "compound_score": vs["compound"]
    }

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
