import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import hashlib
import time
from flask import Flask, jsonify, request

# Generating sample data for species population trends
def generate_sample_data(num_species=100):
    np.random.seed(42)  # For reproducibility
    species_data = {
        'species_name': [f'Species {i}' for i in range(1, num_species + 1)],
        'population': np.random.randint(50, 1000, size=num_species),
        'habitat_loss': np.random.randint(0, 100, size=num_species),
        'poaching': np.random.randint(0, 50, size=num_species),
        'reproduction_rate': np.random.uniform(0.01, 0.3, size=num_species),
        'species_health': np.random.choice([0, 1], size=num_species, p=[0.4, 0.6])
    }
    df = pd.DataFrame(species_data)
    df.to_csv('species_population_data.csv', index=False)

# Generate sample data
generate_sample_data(100)

# Load the data
data = pd.read_csv('species_population_data.csv')

# Preprocessing the data
features = data.drop(['species_name', 'species_health'], axis=1)
labels = data['species_health']

# Scaling the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# AI Model for real-time monitoring
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f'Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%')

# Predicting species health
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Display predictions
predicted_health = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(predicted_health)

# Blockchain integration for secure contributions
class ConservationBlockchain:
    def __init__(self):
        self.chain = []
        self.contributions = []
        self.create_block(previous_hash='1', proof=100)

    def create_block(self, proof, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'proof': proof,
            'previous_hash': previous_hash,
            'timestamp': time.time()
        }
        self.chain.append(block)
        return block

    def add_contribution(self, contributor, amount):
        contribution = {
            'contributor': contributor,
            'amount': amount,
            'block_index': len(self.chain) + 1,
            'timestamp': time.time()
        }
        self.contributions.append(contribution)
        print(f"Contribution added: {contribution}")
        return contribution

    def hash(self, block):
        block_string = str(block).encode()
        return hashlib.sha256(block_string).hexdigest()

# Tokenization of wildlife
def tokenize_wildlife(species_name, value):
    token = {
        'species_name': species_name,
        'token_value': value,
        'timestamp': pd.Timestamp.now()
    }
    print(f"Token created: {token}")
    return token

# Empowering communities
def empower_communities(community_data):
    print("Empowering communities with the following data:")
    for community in community_data:
        print(community)

# Flask API for interaction
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_species_health():
    input_data = request.get_json()
    input_df = pd.DataFrame(input_data)
    input_scaled = scaler.transform(input_df)
    predictions = model.predict(input_scaled)
    return jsonify(predictions.tolist())

@app.route('/contribute', methods=['POST'])
def contribute():
    data = request.get_json()
    contributor = data.get('contributor')
    amount = data.get('amount')
    blockchain.add_contribution(contributor, amount)
    return jsonify({"message": "Contribution added successfully."})

if __name__ == "__main__":
    # Generate sample data and train model
    generate_sample_data(100)  # Create data for 100 species
    
    data = pd.read_csv('species_population_data.csv')
    features = data.drop(['species_name', 'species_health'], axis=1)
    labels = data['species_health']

    features_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

    # Blockchain integration demonstration
    blockchain = ConservationBlockchain()
    blockchain.add_contribution(contributor="John Doe", amount=100)

    # Tokenization demonstration
    token = tokenize_wildlife("Species 1", value=1500)

    # Community engagement demonstration
    community_data = [
        {"name": "Community 1", "engagement": "Organized a cleanup"},
        {"name": "Community 2", "engagement": "Conducted awareness campaign"},
        {"name": "Community 3", "engagement": "Planted trees in the local area"},
    ]
    empower_communities(community_data)

    # Run the Flask app
    app.run(debug=True)
