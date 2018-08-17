# Flask API for Avazu scikit learn

### Dependencies
- scikit-learn
- Flask
- pandas
- numpy

```
pip install -r requirements.txt
```

### Running API
```
python main.py <port>
```

# Endpoints
### /predict (POST)
Returns an array of predictions given a JSON object representing independent variables. Here's a sample input:
```
[
    {"C14": 17753,"C17":1993, "C19": 1063, "C21": 33, "app_category": "07d7df22", "app_id": "ecad2386", "device_model": "be87996b"}
]
```
```
curl -d '[{"C14": 17753,"C17":1993, "C19": 1063, "C21": 33, "app_category": "07d7df22", "app_id": "ecad2386", "device_model": "be87996b"}]' -H "Content-Type: application/json" \
     -X POST http://localhost:3000/predict
```
and sample output:
```
{"predictions": [0]}
```


### /train (GET)
Trains the model.
```
Trained in 359.4 seconds
Model training score: 0.8277733333333334
Log Loss:
0.42363720247388964
RMSE:
0.3629700528553774
```
### /wipe (GET)
Removes the trained model.
