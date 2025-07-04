import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import load_and_preprocess_data
from neuro_fuzzy_system import NeuroFuzzySystem
from two_stage_analysis import TwoStageAnalysis  # Import the new module
from sklearn.model_selection import train_test_split, KFold
import os
import torch

def download_dataset():
    """Download the Pima Indians Diabetes dataset if not already present"""
    if not os.path.exists('diabetes.csv'):
        print("Downloading Pima Indians Diabetes dataset...")
        # URL for the dataset
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
        
        # Column names
        column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        
        # Download and save the dataset
        try:
            df = pd.read_csv(url, names=column_names)
            df.to_csv('diabetes.csv', index=False)
            print("Dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please download the dataset manually and place it in the project directory.")
    else:
        print("Dataset already exists.")

def test_with_custom_input(model, feature_names, two_stage_analyzer):
    """Test the model with custom input values and optionally compare with known outcome"""
    print("\n=== Test the model with a custom patient example ===")
    
    # First, get only the Glucose value
    while True:
        try:
            glucose_value = float(input("Enter Glucose value: "))
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Create a temporary dictionary with just glucose
    temp_values = {feature: 0 for feature in feature_names}
    temp_values['Glucose'] = glucose_value
    
    # Create a DataFrame with the temporary values
    temp_df = pd.DataFrame([temp_values])
    
    # Perform initial glucose-based screening
    print("\n=== Initial Glucose Screening ===")
    if glucose_value >= 126:
        print(f"ALERT: Glucose level ({glucose_value}) indicates DIABETES")
        print("Proceeding to collect additional parameters for detailed analysis...")
        needs_detailed_analysis = True
    elif glucose_value >= 100:
        print(f"CAUTION: Glucose level ({glucose_value}) indicates PREDIABETES")
        print("Proceeding to collect additional parameters for detailed analysis...")
        needs_detailed_analysis = True
    else:
        print(f"Glucose level ({glucose_value}) is within normal range")
        proceed = input("Would you still like to proceed with detailed analysis? (y/n): ").strip().lower()
        needs_detailed_analysis = proceed == 'y'
    
    # If glucose suggests diabetes or user wants to proceed, collect other parameters
    input_values = {'Glucose': glucose_value}
    
    if needs_detailed_analysis:
        # Get input for remaining features
        for feature in feature_names:
            if feature != 'Glucose':  # Skip glucose as we already have it
                while True:
                    try:
                        value = float(input(f"Enter value for {feature}: "))
                        input_values[feature] = value
                        break
                    except ValueError:
                        print("Please enter a valid number.")
    else:
        # Use default/average values for other parameters
        print("Using default values for other parameters...")
        for feature in feature_names:
            if feature != 'Glucose':
                input_values[feature] = 0  # Will be replaced with mean values during preprocessing
    
    # Create a DataFrame with the input values
    input_df = pd.DataFrame([input_values])
    
    # Apply preprocessing steps
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    input_processed = scaler.fit_transform(input_df)
    
    # Make prediction using the neuro-fuzzy model
    prediction_prob = model.predict(input_processed)[0][0]
    prediction = 1 if prediction_prob > 0.5 else 0
    
    print("\n=== Neuro-Fuzzy Model Prediction ===")
    print(f"Probability of having diabetes: {prediction_prob:.4f} ({prediction_prob*100:.2f}%)")
    print(f"Model's diagnosis: {'Diabetic' if prediction == 1 else 'Non-diabetic'}")
    
    # Two-stage analysis
    print("\n=== Two-Stage Analysis Results ===")
    analysis_result = two_stage_analyzer.analyze_patient(input_df.values[0])
    
    # Display initial screening result
    if analysis_result['flag'] == 2:
        print(f"Initial Glucose Screening: DIABETIC (Glucose = {input_values['Glucose']})")
    elif analysis_result['flag'] == 1:
        print(f"Initial Glucose Screening: PREDIABETIC (Glucose = {input_values['Glucose']})")
    else:
        print(f"Initial Glucose Screening: NORMAL (Glucose = {input_values['Glucose']})")
    
    print(f"Glucose-based probability: {analysis_result['probability']:.4f}")
    
    if needs_detailed_analysis:
        # Display risk profile
        print("\n=== Risk Profile ===")
        risk_profile = analysis_result['risk_profile']
        print(f"Cardiovascular Risk: {risk_profile['cardiovascular_risk']}")
        print(f"Metabolic Syndrome: {risk_profile['metabolic_syndrome']}")
        print(f"Beta Cell Dysfunction: {risk_profile['beta_cell_dysfunction']}")
        print(f"Obesity-Related Risk: {risk_profile['obesity_related_risk']}")
        
        # Display recommendations
        print("\n=== Recommendations ===")
        for i, recommendation in enumerate(risk_profile['recommendations'], 1):
            print(f"{i}. {recommendation}")
    
    # Ask if user knows the actual outcome
    print("\nDo you know the actual diagnosis? (y/n)")
    has_actual = input().strip().lower() == 'y'
    
    if has_actual:
        while True:
            try:
                actual_outcome = int(input("Enter the actual outcome (0 for Non-diabetic, 1 for Diabetic): "))
                if actual_outcome not in [0, 1]:
                    print("Please enter either 0 or 1.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number (0 or 1).")
        
        print(f"Actual diagnosis: {'Diabetic' if actual_outcome == 1 else 'Non-diabetic'}")
        
        # Check if prediction matches actual outcome
        if prediction == actual_outcome:
            print("\n✓ The model's prediction MATCHES the actual outcome.")
        else:
            print("\n✗ The model's prediction DOES NOT MATCH the actual outcome.")
    
    return prediction, prediction_prob, analysis_result

def main():
    # Download dataset if needed
    download_dataset()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    # Initialize the two-stage analyzer
    two_stage_analyzer = TwoStageAnalysis(feature_names)
    
    # Visualize glucose distribution
    two_stage_analyzer.visualize_glucose_distribution(np.vstack((X_train, X_test)), 
                                                     pd.concat([y_train, y_test]))
    print("Glucose distribution visualization saved.")
    
    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Initialize and train the neuro-fuzzy system
    print("Initializing Neuro-Fuzzy System...")
    neuro_fuzzy = NeuroFuzzySystem(feature_names)
    
    # Visualize fuzzy membership functions
    neuro_fuzzy.fuzzy_system.visualize_membership_functions()
    print("Fuzzy membership functions visualized and saved.")
    
    # Hyperparameter settings for better accuracy
    epochs = 200
    batch_size = 32
    learning_rate = 0.001
    
    print("Training Neuro-Fuzzy System...")
    history = neuro_fuzzy.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    
    # Use K-fold cross-validation to create an ensemble of models
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    ensemble_models = []
    fold_accuracies = []
    
    print(f"Training ensemble of {n_folds} Neuro-Fuzzy models with cross-validation...")
    
    # Train models using k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nTraining fold {fold+1}/{n_folds}")
        
        # Split data for this fold
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Initialize neuro-fuzzy system
        neuro_fuzzy = NeuroFuzzySystem(feature_names)
        
        # Train the model
        history = neuro_fuzzy.train(X_fold_train, y_fold_train, X_fold_val, y_fold_val, 
                                   epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        
        # Evaluate on validation set
        val_results = neuro_fuzzy.evaluate(X_fold_val, y_fold_val)
        fold_accuracies.append(val_results['accuracy'])
        print(f"Fold {fold+1} validation accuracy: {val_results['accuracy']:.4f}")
        
        # Add model to ensemble
        ensemble_models.append(neuro_fuzzy)
        
        # Save model for this fold
        neuro_fuzzy.save_model(f'diabetes_neuro_fuzzy_model_fold_{fold+1}.pth')
    
    print(f"\nAverage validation accuracy across folds: {np.mean(fold_accuracies):.4f}")
    
    # Evaluate ensemble on test set
    print("\nEvaluating ensemble on test set...")
    ensemble_predictions = np.zeros((X_test.shape[0], 1))
    
    for model in ensemble_models:
        # Get predictions from this model
        pred = model.predict(X_test)
        ensemble_predictions += pred
    
    # Average predictions
    ensemble_predictions /= len(ensemble_models)
    
    # Convert to binary predictions
    y_pred = (ensemble_predictions > 0.5).astype(int).flatten()
    
    # Calculate accuracy
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Ensemble Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Diabetes', 'Diabetes'],
               yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Ensemble Confusion Matrix')
    plt.tight_layout()
    plt.savefig('ensemble_confusion_matrix.png')
    plt.close()
    
    # Save the best model (highest validation accuracy)
    best_model_idx = np.argmax(fold_accuracies)
    best_model = ensemble_models[best_model_idx]
    best_model.save_model('best_diabetes_neuro_fuzzy_model.pth')
    print(f"Best model saved as 'best_diabetes_neuro_fuzzy_model.pth' (Fold {best_model_idx+1})")
    
    # Evaluate two-stage analysis on test set
    print("\nEvaluating two-stage analysis on test set...")
    
    # Get initial screening results
    flags, probabilities = two_stage_analyzer.initial_screening(X_test)
    
    # Calculate accuracy of glucose-based screening
    glucose_pred = (flags >= 1).astype(int)
    glucose_accuracy = accuracy_score(y_test, glucose_pred)
    
    print(f"Glucose-only screening accuracy: {glucose_accuracy:.4f}")
    
    # Compare with neuro-fuzzy model
    print(f"Neuro-fuzzy model accuracy: {accuracy:.4f}")
    
    # After saving the best model, add:
    print("\nWould you like to test the model with a custom example? (y/n)")
    choice = input().strip().lower()
    if choice == 'y':
        test_with_custom_input(best_model, feature_names, two_stage_analyzer)
    
    print("Project execution completed successfully!")

if __name__ == "__main__":
    main()