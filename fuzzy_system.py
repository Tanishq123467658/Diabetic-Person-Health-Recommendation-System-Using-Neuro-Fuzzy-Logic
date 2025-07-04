import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

class DiabetesFuzzySystem:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.setup_fuzzy_system()
        
    def setup_fuzzy_system(self):
        # Create fuzzy variables for each feature
        self.fuzzy_variables = {}
        
        # Glucose - more granular membership functions
        glucose = ctrl.Antecedent(np.arange(0, 200, 1), 'Glucose')
        glucose['very_low'] = fuzz.trimf(glucose.universe, [0, 0, 70])
        glucose['low'] = fuzz.trimf(glucose.universe, [50, 85, 110])
        glucose['medium'] = fuzz.trimf(glucose.universe, [90, 110, 140])
        glucose['high'] = fuzz.trimf(glucose.universe, [120, 150, 180])
        glucose['very_high'] = fuzz.trimf(glucose.universe, [160, 200, 200])
        self.fuzzy_variables['Glucose'] = glucose
        
        # BMI - more granular membership functions
        bmi = ctrl.Antecedent(np.arange(0, 60, 1), 'BMI')
        bmi['underweight'] = fuzz.trimf(bmi.universe, [0, 0, 18.5])
        bmi['normal'] = fuzz.trimf(bmi.universe, [17, 21.75, 25])
        bmi['overweight'] = fuzz.trimf(bmi.universe, [23, 27.5, 32])
        bmi['obese'] = fuzz.trimf(bmi.universe, [30, 35, 40])
        bmi['extremely_obese'] = fuzz.trimf(bmi.universe, [37.5, 50, 60])
        self.fuzzy_variables['BMI'] = bmi
        
        # Age
        age = ctrl.Antecedent(np.arange(0, 100, 1), 'Age')
        age['young'] = fuzz.trimf(age.universe, [0, 0, 35])
        age['middle'] = fuzz.trimf(age.universe, [25, 45, 65])
        age['old'] = fuzz.trimf(age.universe, [55, 100, 100])
        self.fuzzy_variables['Age'] = age
        
        # Diabetes risk (output)
        risk = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'Risk')
        risk['low'] = fuzz.trimf(risk.universe, [0, 0, 0.5])
        risk['medium'] = fuzz.trimf(risk.universe, [0.3, 0.5, 0.7])
        risk['high'] = fuzz.trimf(risk.universe, [0.5, 1, 1])
        self.fuzzy_variables['Risk'] = risk
        
        # Define fuzzy rules
        self.rules = [
            ctrl.Rule(glucose['high'] & bmi['obese'], risk['high']),
            ctrl.Rule(glucose['high'] & bmi['overweight'] & age['old'], risk['high']),
            ctrl.Rule(glucose['high'] & bmi['overweight'] & age['middle'], risk['high']),
            ctrl.Rule(glucose['high'] & bmi['normal'] & age['old'], risk['medium']),
            ctrl.Rule(glucose['medium'] & bmi['obese'] & age['old'], risk['high']),
            ctrl.Rule(glucose['medium'] & bmi['obese'] & age['middle'], risk['medium']),
            ctrl.Rule(glucose['medium'] & bmi['overweight'], risk['medium']),
            ctrl.Rule(glucose['medium'] & bmi['normal'], risk['low']),
            ctrl.Rule(glucose['low'] & bmi['normal'], risk['low']),
            ctrl.Rule(glucose['low'] & bmi['overweight'] & age['young'], risk['low']),
            ctrl.Rule(glucose['low'] & bmi['overweight'] & age['middle'], risk['medium']),
            ctrl.Rule(glucose['low'] & bmi['obese'], risk['medium'])
        ]
        
        # Create control system
        self.diabetes_ctrl = ctrl.ControlSystem(self.rules)
        self.diabetes_sim = ctrl.ControlSystemSimulation(self.diabetes_ctrl)
    
    def visualize_membership_functions(self):
        # Visualize membership functions
        for var_name, var in self.fuzzy_variables.items():
            plt.figure(figsize=(8, 5))
            var.view()
            plt.title(f'Membership Functions for {var_name}')
            plt.tight_layout()
            plt.savefig(f'{var_name}_membership.png')
            plt.close()
    
    def compute_risk(self, glucose, bmi, age):
        # Compute diabetes risk for a single instance
        self.diabetes_sim.input['Glucose'] = glucose
        self.diabetes_sim.input['BMI'] = bmi
        self.diabetes_sim.input['Age'] = age
        
        try:
            self.diabetes_sim.compute()
            return self.diabetes_sim.output['Risk']
        except:
            # If computation fails, return a default value
            return 0.5
    
    def fuzzify_data(self, X, feature_indices):
        # Extract relevant features (Glucose, BMI, Age)
        glucose_idx, bmi_idx, age_idx = feature_indices
        
        # Compute fuzzy risk for each instance
        risks = []
        for i in range(X.shape[0]):
            risk = self.compute_risk(X[i, glucose_idx], X[i, bmi_idx], X[i, age_idx])
            risks.append(risk)
        
        return np.array(risks).reshape(-1, 1)

if __name__ == "__main__":
    # Test the fuzzy system
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    fuzzy_system = DiabetesFuzzySystem(feature_names)
    fuzzy_system.visualize_membership_functions()
    
    # Test with sample values
    risk = fuzzy_system.compute_risk(glucose=140, bmi=35, age=55)
    print(f"Diabetes risk: {risk}")