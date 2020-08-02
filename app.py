# app.py
from flask import Flask, request, jsonify, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import statistics
app = Flask(__name__)

# sample reviews
sample_reviews = [
    {"useful_votes": 0, "review_text": "I went to Cotto. The pizza was good. They sometimes have specials on pizza or pasta, so you have to look at their website.\nThe location is not great though, I couldn't find parking. It's so terrible I almost get into an accident every time.\nBut the food is so delicious. I highly recommend it."},
    {"useful_votes": 1, "review_text": "Who cares."}
]

sample_reviews = [
    {"useful_votes": 0, "review_text": "I went to Cotto. The pizza was good. They sometimes have specials on pizza or pasta, so you have to look at their website.\n\nThe location is not great though, I couldn't find parking. It's so terrible I almost get into an accident every time.\n\nBut the food is so delicious. I highly recommend it."},
    {"useful_votes": 1, "review_text": "Pretty good dinner with a nice selection of food. Open 24 hours and provide nice service. I usually go here after a night of partying. My favorite dish is the Fried Chicken Eggs Benedict."},
    {"useful_votes": 0, "review_text": "Good truck stop dining at the right price. We love coming here on the weekends when we don't feel like cooking."},
    {"useful_votes": 16, "review_text": "This local BBQ icon was on our list of dining destinations when Santi and I were planning our trip to Madison. Located nearby the airport, the well-lit, wooded interior starts with a pathway that leads to the large overhead menu, the ordering counter, and the dining area. Adding to the décor are the statewide accolades all around the walls. On this evening, the casual atmosphere happened to be unexpectedly subdued, quite unlike the other BBQ restaurants I've visited.   \n\nWe shared a Combination Meal ($17.75) that included our choice of 1/3 Slab, Smoked Pork Shoulder Sandwich, and extra serving of smoked pork shoulder, Cole Slaw, and BBQ Baked Beans. The Slab/Pork Ribs were smoky, tender, and the meat easily peeled off the bones. There was an excellent hint of smoke to deepen its flavor, and that was further enhanced by the BBQ sauce to provide a sweet higher note, strong black peppery secondary, and anchored by a noticeable spicy kick that reverberates with good hops. For the price, there wasn't enough meat on the ribs, but the flavors were still sensational. The dinner roll that this dish came with was sweet, soft, and spongy. I loved it.\n\nThe Pork Shoulder Sandwich was excellent as well. The pork shoulders were tender without one single dry spot--perfectly executed and delicious without the BBQ sauce. The meat was flavorful and deepened by its smoky flavor that coupled perfectly with the well-rounded, full-bodied spicy BBQ sauce. I absolutely loved it except for the buns that turned out flaky with hollow spots and lacking flavor. We finished the meat without getting into the buns very much. This sandwich could have been so much better with a bun of higher quality. \n\nThe BBQ Baked Beans were smooth, sweet, and delicious. The Slaw on the other hand was too wet with a weak texture. Compared to other BBQ joints, their sides could use more varieties such as green beans, collard greens, yam and other possibilities to complement this resounding BBQ. They also have two sauces, and I wished we had the option of trying both at the table.\n\nService was impatient at first, as we were initially tentative and indecisive about our choices. I wished the server could have been much more educational with their strengths and limitations to help us make the right decisions on this very first visit. However, she later became much more warm, accessible, attentive, and helpful towards the middle and end of the meal. Price-wise, their items were pricy with the portions we received. I'd be very interested in coming back to try their Beef Briskets and Chicken Dinner."},
    {"useful_votes": 12, "review_text": "It was the 2nd meet up for The International Supper Club and this time Comedor Guadalajara was the place to be!\n\nFrom the outside, you are not quite sure of what are you are getting into, you walk up to the place and you see a huge concrete pad that looks like it's used for outside parties maybe?  You go through the doors and then are shown your table and your mind is thinking how is it possibly this is the same building you were looking at on the outside.  It's huge, and opened, and I didn't even see the entire place. It's well lit and the colorful! A very nice surprise!\n\nWe were a party of 8 and our server was Maria, and she was amazing, she made sure we had everything we needed and more. When you 1st walked in the podium for the hostess says NO SPLIT CHECKS, so our party didn't even ask, we would have figured it out, but when it came time for the check, Maria had already split them up for us.  Love great service!\n\nThe chips are fresh, the salsa, well for me it's perfect! I could have stood it a tab bit hotter, but it had the perfect bit of heat for me, but for a fellow dinner, it was too hot for him.\n\nOrdered a cheese crisp and some guacamole for appetizers and the cheese crisp was just that, a cheese crisp nothing too exciting to write about, but the guacamole, omg it was soo fresh tasting, I loved that stuff!! I think I might have even licked it off my fingers, I'm sure I could have eaten the entire bowl myself  (or did I).\n\nOne of the Wednesday night specials was a poblano pepper stuffed with cheese and topped with green chili sauce with grilled chicken, and it came with beans & rice. OMG AMAZING!!!!!!!!!! I didn't even see it on the menu but damn I want it again! The pepper was fresh, the green sauce was delish, and the grilled chicken just tied this dish together! The beans and rice were perfectly cooked. AMAZING!!!!\n\nMy meal filled me up pretty well, but I did order a churro, and it was the perfect way to end a great meal. Love churros!!! Warm pastry stick covered in cinnamon and sugar. Oh yea I went home a happy girl for sure!"},
    {"useful_votes": 0, "review_text": "If you like lot lizards, you'll love the Pine Cone!"}
]
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
        # use first model for short reviews
        if(len(review_text) <= 100):
            useful_prediction_val = predict_usefulness_short(len(review_text), len(review_lines))
        else:
            useful_prediction_val = predict_usefulness_long(review_lines)
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
    return render_template('index.html' , sample_reviews=sample_reviews)

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

def predict_usefulness_short(char_length, num_sentences):
    model_params = {'intercept': -4.121402521392098, 'log_char_length': 0.6190290290266008, 'log_num_sentences': 0.10444849122310086}
    linear_val = model_params["intercept"] + model_params["log_char_length"] * np.log(char_length) + model_params["log_num_sentences"] * np.log(num_sentences)
    return 1 / (1 + np.exp(-1.0 * linear_val))

def predict_usefulness_long(processed_lines):
    positive_sentences = []
    negative_sentences = []
    neutral_sentences = []
    for processed_line in processed_lines:
        if processed_line["compound_score"] >= 0.05:
            positive_sentences.append(processed_line["line"].lower())
        elif processed_line["compound_score"] <= -0.05:
            negative_sentences.append(processed_line["line"].lower())
        else:
            neutral_sentences.append(processed_line["line"].lower())

    log_pos_sentence_count = np.log(len(set(positive_sentences)) + 1)
    log_neg_sentence_count = np.log(len(set(negative_sentences)) + 1)
    log_neut_sentence_count = np.log(len(set(neutral_sentences)) + 1)
    print(len(set(positive_sentences)), len(set(negative_sentences)), len(set(neutral_sentences)))

    model_params = {'intercept': -0.8433776961797712, 'review_distinct_pos_sentences': 0.0883927794156043, 'review_distinct_neg_sentences': 0.10833381656026096, 'review_distinct_neut_sentences': 0.06172328962965702}
    linear_val = model_params["intercept"] \
        + model_params["review_distinct_pos_sentences"] * len(set(positive_sentences)) \
        + model_params["review_distinct_neg_sentences"] * len(set(negative_sentences)) \
        + model_params["review_distinct_neut_sentences"] * len(set(neutral_sentences))
    print(linear_val)
    return 1 / (1 + np.exp(-1.0 * linear_val))

@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html', sample_reviews=sample_reviews)

@app.route('/useful-score')
def usefulness():
    return render_template('usefulness.html', sample_reviews=sample_reviews)

@app.route('/samples')
def samples():
    return render_template('samples.html', sample_reviews=sample_reviews)


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
