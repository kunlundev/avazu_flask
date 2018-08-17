import sys
import shutil
import time
import traceback
import os

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import log_loss, mean_squared_error

app = Flask(__name__)

model_directory = 'model'
model_file_name = '%s/model.pkl' % model_directory
model_columns_file_name = '%s/model_columns.pkl' % model_directory

model_columns = None
classifier = None

@app.route('/predict', methods=['POST'])
def predict():
    if classifier:
        try:
            json_ = request.json
            print(json_)
            print('ppp')
            print(pd.DataFrame(json_))
            query = pd.get_dummies(pd.DataFrame(json_))

            query = query.reindex(columns=model_columns, fill_value=0)

            predictions = list(classifier.predict(query))
            print(predictions)

            return jsonify({'prediction': ('%s' % predictions)})

        except Exception:

            return jsonify({'error': str(Exception), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return 'no model here'

@app.route('/train', methods=['GET'])
def train():
    # Load data
    train = pd.read_csv('./dataset/train-100000R', dtype={'id': pd.np.string_})
    test = pd.read_csv('./dataset/test-100000R', dtype={'id': pd.np.string_})

    # Pre-processing non-number values
    le = LabelEncoder()
    train_tmp = train.apply(LabelEncoder().fit_transform)

    # get feature columns
    #variance threshold
    new_columns = VarianceThreshold(threshold=3).fit(train_tmp)
    mask = new_columns.get_support()
    new_features = []
    feature_names = train_tmp.columns

    for bool, feature in zip(mask, feature_names):
        if bool:
            new_features.append(feature)

    print(new_features)

    # chi2 select
    chi_columns = SelectKBest(chi2, k=10).fit(train_tmp, train_tmp['click'])
    chi_mask = chi_columns.get_support()
    chi_features = []

    for bool, feature in zip(chi_mask, feature_names):
        if bool:
            chi_features.append(feature)

    common_features =list(np.intersect1d(new_features, chi_features))
    # remove some columns cause too many value_counters
    common_features.remove('device_id')
    common_features.remove('id')

    print("common_features")
    print(common_features)

    # one hot encoder
    x = pd.get_dummies(train[common_features])
    y = train.click

    # capture a list of columns that will be used for prediction
    global model_columns
    model_columns = list(x.columns)
    joblib.dump(model_columns, model_columns_file_name)

    # test data
    test_x = pd.get_dummies(test[common_features])
    test_x = test_x.reindex(columns=model_columns, fill_value=0)
    test_y = test.click

    global classifier

    classifier = GradientBoostingClassifier()
    start = time.time()
    classifier.fit(x, y)

    print('Trained in %.1f seconds' % (time.time() - start))
    print('Model training score: %s' % classifier.score(x, y))
    print('Log Loss:')
    print(log_loss(test_y.values, classifier.predict_proba(test_x)))
    print('RMSE:')
    print(mean_squared_error(test_y.values, np.compress([False, True],
    classifier.predict_proba(test_x), axis=1)) ** 0.5)  # RMSE

    joblib.dump(classifier, model_file_name)

    return 'Success'


@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        shutil.rmtree('model')
        os.makedirs(model_directory)
        return 'Model wiped'

    except Exception:
        print(str(Exception))
        return 'Could not remove and recreate the model directory'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception:
        port = 80

    try:
        classifier = joblib.load(model_file_name)
        print('model loaded')
        model_columns = joblib.load(model_columns_file_name)
        print('model columns loaded')

    except Exception:
        print('No model here')
        print('Train first')
        classifier = None

    app.run(host='0.0.0.0', port=port, debug=True)
