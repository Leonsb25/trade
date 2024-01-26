from flask import Flask, render_template, request

app = Flask(__name__)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


historical_data = pd.read_csv('/Users/leon/Downloads/TCS_5Y.csv')

# Function to calculate pivot points and trading signals
def calculate_pivot_points(high, low, close):
    pivot_point = (high + low + close) / 3
    support_1 = (2 * pivot_point) - high
    support_2 = pivot_point - (high - low)
    resistance_1 = (2 * pivot_point) - low
    resistance_2 = pivot_point + (high - low)

    return {
        'Pivot Point': pivot_point,
        'Support 1': support_1,
        'Support 2': support_2,
        'Resistance 1': resistance_1,
        'Resistance 2': resistance_2
    }

def trading_strategy(pivot_points, current_close):
    if current_close > pivot_points['Resistance 1']:
        return 'Buy'
    elif current_close < pivot_points['Support 1']:
        return 'Sell'
    else:
        return 'Hold'

# Feature engineering for machine learning
historical_data['Pivot Point'] = (historical_data['High'] + historical_data['Low'] + historical_data['Close']) / 3
historical_data['Support 1'] = (2 * historical_data['Pivot Point']) - historical_data['High']
historical_data['Support 2'] = historical_data['Pivot Point'] - (historical_data['High'] - historical_data['Low'])
historical_data['Resistance 1'] = (2 * historical_data['Pivot Point']) - historical_data['Low']
historical_data['Resistance 2'] = historical_data['Pivot Point'] + (historical_data['High'] - historical_data['Low'])

# Create target variable ('Signal') based on trading strategy
historical_data['Signal'] = historical_data.apply(lambda row: trading_strategy(row[['Resistance 1', 'Support 1']], row['Close']), axis=1)

# Features for machine learning
X = historical_data[['Pivot Point', 'Support 1', 'Support 2', 'Resistance 1', 'Resistance 2']]

# Target variable
y = historical_data['Signal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the decision tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy}')

# User inputs for current values
#current_high_price = float(input("Enter the current high price: "))
#current_low_price = float(input("Enter the current low price: "))
#current_close_price = float(input("Enter the current close price: "))

# Calculate pivot points
#current_pivot_points = calculate_pivot_points(current_high_price, current_low_price, current_close_price)

# Use the model to predict the trading signal
#current_features = [current_pivot_points['Pivot Point'], current_pivot_points['Support 1'], current_pivot_points['Support 2'], current_pivot_points['Resistance 1'], current_pivot_points['Resistance 2']]
#predicted_signal = model.predict([current_features])[0]

#print(f'Predicted Trading Signal: {predicted_signal}')


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        current_high_price= int(request.form['current_high_price'])
        current_low_price= int(request.form['current_low_price'])
        current_close_price=int(request.form['current_close_price'])
        calculate_pivot_points(current_high_price, current_low_price, current_close_price)
        current_pivot_points = calculate_pivot_points(current_high_price, current_low_price, current_close_price)

        current_features = [current_pivot_points['Pivot Point'], current_pivot_points['Support 1'],
                            current_pivot_points['Support 2'], current_pivot_points['Resistance 1'],
                            current_pivot_points['Resistance 2']]
        predicted_signal = model.predict([current_features])[0]
        result = {predicted_signal}
        print(f'Predicted Trading Signal: {predicted_signal}')

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
