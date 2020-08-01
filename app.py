# app.py
from flask import Flask, request, jsonify, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
app = Flask(__name__)

# for tokenizing reviews
# from https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer/33456191
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
        useful_prediction_val = predict_usefulness(
            len(review_text),
            len(review_lines)
        )
        return jsonify({
            "review_html": render_template('review.html', review_lines=review_lines),
            "progress_html": render_template('progress.html',
                useful_prediction_val=useful_prediction_val,
                useful_prediction_int=int(useful_prediction_val * 100),
                useful_prediction_color_class=get_prediction_color_class(useful_prediction_val),
            ),
            "useful_prediction": useful_prediction_val
        })
    else:
        return jsonify({
            "error": "no review found."
        })

# no idea why this has to be done outside the template
def get_prediction_color_class(predicted_val):
    if predicted_val > 0.8:
        return "bg-success"
    elif predicted_val > 0.7:
        return ""
    elif predicted_val > 0.5:
        return "bg-info"
    elif predicted_val > 0.25:
        return "bg-warning"
    else:
        return "bg-danger"


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

def predict_usefulness(char_length, num_sentences):
    model_params = {'intercept': -4.121402521392098, 'log_char_length': 0.6190290290266008, 'log_num_sentences': 0.10444849122310086}
    linear_val = model_params["intercept"] + model_params["log_char_length"] * np.log(char_length) + model_params["log_num_sentences"] * np.log(num_sentences)
    return 1 / (1 + np.exp(-1.0 * linear_val))

@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html')


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
