from flask import Flask, request, render_template
import pickle

application = Flask(__name__)
app = application

# Import trained Ridge Regression model and Standard Scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))


# Main page
@app.route("/")
def index():
    # Directly open the prediction form
    return render_template('home.html')


# Prediction route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == "POST":

        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Scale the input data
        new_data_scaled = standard_scaler.transform(
            [[
                Temperature,
                RH,
                Ws,
                Rain,
                FFMC,
                DMC,
                ISI,
                Classes,
                Region
            ]]
        )

        # Make prediction
        result = ridge_model.predict(new_data_scaled)

        # Convert prediction
        results = result[0] / 100

        return render_template('home.html', results=results)

    else:
        return render_template('home.html')


print(app.url_map)


if __name__ == "__main__":
    app.run(host="0.0.0.0")