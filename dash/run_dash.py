
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_boston

from clark_dash.mockup import build_mockup

if __name__ == '__main__':

    # ----------------------- Load data --------------------------
    boston = load_boston()
    X_boston = pd.DataFrame(boston['data'], columns=boston['feature_names'])
    y_boston = pd.DataFrame(boston['target'], columns=['target'])

    iris = load_iris()
    X_iris = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    y_iris = pd.DataFrame(iris['target'], columns=['target'])

    # ----------------------- Build app --------------------------
    app = build_mockup(X_iris)

    # ----------------------- Run Server --------------------------
    app.run_server(debug = True)
    #app.run_server(debug=True, port=8080, host='127.0.0.1')