from flask import Flask, jsonify, abort, request
from http import HTTPStatus
from schemas import TweetUrlSchema, ResultSchema
from flask_cors import CORS
from helpers import get_tweet_content
from models.model_mfas import init_mfas, run_mfas_inference


app = Flask(__name__)
CORS(app)

# load model first
model_mfas = init_mfas()

@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/predict', methods = ['POST'])
def predict():
    app.logger.info('req: %s', request.get_json())
    try:
        req = TweetUrlSchema().load(request.get_json())
        try:
            tweetUrl = req['tweetUrl']
            app.logger.info('tweetUrl: %s', tweetUrl)
            if tweetUrl is None or tweetUrl == '':
                return jsonify(HTTPStatus.BAD_REQUEST.phrase), HTTPStatus.BAD_REQUEST
            tweetId = tweetUrl.split('/')[-1]
            if tweetId is None or tweetId == '' or not tweetId.isnumeric():
                return jsonify(HTTPStatus.BAD_REQUEST.phrase), HTTPStatus.BAD_REQUEST

        except KeyError:
            return jsonify(HTTPStatus.UNPROCESSABLE_ENTITY.phrase), HTTPStatus.UNPROCESSABLE_ENTITY
        
        text, img_url = get_tweet_content(tweetId)
        app.logger.info('text: %s', text)
        app.logger.info('img_url: %s', img_url)

        mfas_res = run_mfas_inference(text, img_url, model_mfas)

        result = {'clip': 'Anti-Vaxx', 'mfas': mfas_res}
        return jsonify(ResultSchema().dump(result)), HTTPStatus.OK
    except Exception as e:
        abort(400, e)

if __name__ == '__main__':
    app.run(debug=True)
