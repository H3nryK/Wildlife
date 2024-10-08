import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Generating sample data for species population trends
def generate_sample_data(num_samples=1000):
    species_names = [f'Species {i}' for i in range(1, 21)]
    data = {
        'species_name': np.random.choice(species_names, num_samples),
        'population': np.random.randint(50, 500, size=num_samples),
        'habitat_loss': np.random.randint(0, 30, size=num_samples),
        'poaching': np.random.randint(0, 15, size=num_samples),
        'species_health': np.random.choice([0, 1], size=num_samples)  # 1 = Healthy, 0 = Endangered
    }
    df = pd.DataFrame(data)
    df.to_csv('species_population_data.csv', index=False)

generate_sample_data()

# Load the data
data = pd.read_csv('species_population_data.csv')

# Preprocessing the data
features = data.drop(['species_name', 'species_health'], axis=1)
labels = data['species_health']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Hyperparameter tuning for RandomForest
model = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_accuracy = best_model.score(X_test, y_test)
print(f'Best Model Accuracy: {best_accuracy * 100:.2f}%')
print(f'Best Hyperparameters: {grid_search.best_params_}')

# Predicting species health
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Visualizing feature importances
feature_importances = best_model.feature_importances_
features_list = features.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features_list)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Blockchain integration for secure contributions
class ConservationBlockchain:
    def __init__(self):
        self.chain = []
        self.create_block(previous_hash='1', proof=100)

    def create_block(self, proof, previous_hash):
        block = {'index': len(self.chain) + 1,
                 'proof': proof,
                 'previous_hash': previous_hash}
        self.chain.append(block)
        return block

    def add_contribution(self, contributor, amount):
        # Logic to add contribution to the blockchain
        print(f'Contribution of {amount} by {contributor} added to the blockchain.')

# Tokenization of wildlife
def tokenize_wildlife(species_name, value):
    # Logic to create tokens for wildlife
    print(f'Tokens created for {species_name} valued at {value}.')

# Empowering communities
def empower_communities(community_data):
    # Logic to engage communities in conservation efforts
    print('Communities empowered with conservation data:', community_data)

# Example usage of the blockchain
blockchain = ConservationBlockchain()
blockchain.add_contribution(contributor='John Doe', amount=100)
tokenize_wildlife(species_name='Species A', value=500)
empower_communities(community_data={'Community A': 'Engaged', 'Community B': 'Engaged'})
