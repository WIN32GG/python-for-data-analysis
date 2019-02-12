from flask import Flask, Response, abort, make_response, jsonify, request
import pickle
import numpy as np

route = '/<float:LB>/<float:AC>/<float:FM>/<float:UC>/<float:DL>/<float:DS>/<float:DP>/<float:ASTV>/<float:MSTV>/<float:ALTV>/<float:MLTV>/<float:Width>/<float:Min>/<float:Max>/<float:Nmax>/<float:Nzeros>/<float:Mode>/<float:Mean>/<float:Median>/<float:Variance>/<float:Tendency>'
app = Flask(__name__)

model = None

@app.route(route, methods = ['GET'])
def get_prediction(**kwargs):
    global model
    values = np.array(list(kwargs.values())).reshape(1, -1)
    return jsonify({"request": kwargs, "prediction": str(model.predict(values)[0])})

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'api called incorrectly', 'usage': route.replace('\n','')}), 404)

def load():
    global model
    with open('model.pck', 'rb') as fd:
        model = pickle.loads(fd.read())

if __name__ == "__main__":
    load()
    app.run(host = '127.0.0.1', port=8080, debug=None)

"""
Example call:
http://127.0.0.1:8080/120.0/0.000/0.000/0.000/0.000/0.000/0.000/73.0/0.5/43.0/2.4/64.0/62.0/126.0/2.0/0.0/120.0/137.0/121.0/73.0/1.0
should be 2.0
"""
