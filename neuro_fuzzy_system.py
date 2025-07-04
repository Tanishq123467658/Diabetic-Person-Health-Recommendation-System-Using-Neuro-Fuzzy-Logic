import numpy as np
from fuzzy_system import DiabetesFuzzySystem
from neural_network import DiabetesNeuralNetwork
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class NeuroFuzzySystem:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.fuzzy_system = DiabetesFuzzySystem(feature_names)
        self.neural_network = None
        
        # Get indices for the features used in the fuzzy system
        self.glucose_idx = list(feature_names).index('Glucose')
        self.bmi_idx = list(feature_names).index('BMI')
        self.age_idx = list(feature_names).index('Age')
    
    def prepare_neuro_fuzzy_input(self, X):
        # Get fuzzy risk values
        fuzzy_risks = self.fuzzy_system.fuzzify_data(X, 
                                                    (self.glucose_idx, self.bmi_idx, self.age_idx))
        
        # Combine original features with fuzzy risk
        X_combined = np.hstack((X, fuzzy_risks))
        
        return X_combined
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.001):
        # Prepare inputs with fuzzy components
        X_train_combined = self.prepare_neuro_fuzzy_input(X_train)
        X_val_combined = self.prepare_neuro_fuzzy_input(X_val)
        
        # Initialize and train neural network
        self.neural_network = DiabetesNeuralNetwork(input_shape=(X_train_combined.shape[1],), learning_rate=learning_rate)
        history = self.neural_network.train(X_train_combined, y_train, X_val_combined, y_val, 
                                           epochs=epochs, batch_size=batch_size)
        
        return history
    
    def predict(self, X):
        X_combined = self.prepare_neuro_fuzzy_input(X)
        return self.neural_network.predict(X_combined)
    
    def evaluate(self, X_test, y_test):
        X_test_combined = self.prepare_neuro_fuzzy_input(X_test)
        loss, accuracy = self.neural_network.evaluate(X_test_combined, y_test)
        
        # Get predictions
        y_pred_prob = self.neural_network.predict(X_test_combined)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob
        }
    
    def visualize_results(self, evaluation_results):
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(evaluation_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Diabetes', 'Diabetes'],
                   yticklabels=['No Diabetes', 'Diabetes'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Print classification report
        print("Classification Report:")
        print(evaluation_results['classification_report'])
        
        # Print accuracy
        print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
    
    def save_model(self, filepath):
        self.neural_network.save_model(filepath)