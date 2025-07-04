import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fuzzy_system import DiabetesFuzzySystem

class TwoStageAnalysis:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.fuzzy_system = DiabetesFuzzySystem(feature_names)
        
        # Get indices for the features
        self.glucose_idx = list(feature_names).index('Glucose')
        self.bmi_idx = list(feature_names).index('BMI')
        self.age_idx = list(feature_names).index('Age')
        self.bp_idx = list(feature_names).index('BloodPressure')
        self.insulin_idx = list(feature_names).index('Insulin')
        self.skin_idx = list(feature_names).index('SkinThickness')
        self.dpf_idx = list(feature_names).index('DiabetesPedigreeFunction')
        
        # Thresholds for glucose-based initial screening
        self.glucose_high_threshold = 126  # Fasting glucose ≥ 126 mg/dL is diagnostic for diabetes
        self.glucose_prediabetic_threshold = 100  # 100-125 mg/dL indicates prediabetes
        
    def initial_screening(self, X):
        """
        First stage: Screen based on glucose levels only
        Returns: 
            - flags: array of 0 (normal), 1 (prediabetic), 2 (diabetic)
            - probabilities: estimated probability of diabetes based on glucose
        """
        glucose_values = X[:, self.glucose_idx]
        
        # Initialize flags and probabilities
        flags = np.zeros(len(glucose_values))
        probabilities = np.zeros(len(glucose_values))
        
        # Apply thresholds
        for i, glucose in enumerate(glucose_values):
            if glucose >= self.glucose_high_threshold:
                flags[i] = 2  # Diabetic
                # Calculate probability (higher glucose = higher probability)
                probabilities[i] = min(0.9 + (glucose - self.glucose_high_threshold) / 200, 0.99)
            elif glucose >= self.glucose_prediabetic_threshold:
                flags[i] = 1  # Prediabetic
                # Linear interpolation between 0.5 and 0.9
                probabilities[i] = 0.5 + 0.4 * (glucose - self.glucose_prediabetic_threshold) / (self.glucose_high_threshold - self.glucose_prediabetic_threshold)
            else:
                flags[i] = 0  # Normal
                # Lower glucose = lower probability
                probabilities[i] = max(0.01, glucose / (2 * self.glucose_prediabetic_threshold))
        
        return flags, probabilities
    
    def detailed_analysis(self, X, flags):
        """
        Second stage: Detailed analysis for cases flagged as prediabetic or diabetic
        Returns a list of dictionaries with risk profiles and recommendations
        """
        results = []
        
        for i, flag in enumerate(flags):
            if flag >= 1:  # Prediabetic or diabetic
                patient_data = X[i]
                
                # Extract individual values
                glucose = patient_data[self.glucose_idx]
                bmi = patient_data[self.bmi_idx]
                age = patient_data[self.age_idx]
                blood_pressure = patient_data[self.bp_idx]
                insulin = patient_data[self.insulin_idx]
                skin_thickness = patient_data[self.skin_idx]
                dpf = patient_data[self.dpf_idx]
                
                # Initialize risk profile
                risk_profile = {
                    'cardiovascular_risk': self._assess_cardiovascular_risk(glucose, blood_pressure, bmi, age),
                    'metabolic_syndrome': self._assess_metabolic_syndrome(glucose, blood_pressure, bmi, insulin),
                    'beta_cell_dysfunction': self._assess_beta_cell_dysfunction(glucose, insulin, dpf),
                    'obesity_related_risk': self._assess_obesity_risk(bmi, skin_thickness),
                    'recommendations': []
                }
                
                # Generate recommendations based on risk factors
                self._generate_recommendations(risk_profile, glucose, bmi, blood_pressure, insulin, age)
                
                results.append(risk_profile)
            else:
                # For normal cases, provide basic health maintenance recommendations
                results.append({
                    'cardiovascular_risk': 'Low',
                    'metabolic_syndrome': 'Unlikely',
                    'beta_cell_dysfunction': 'Low risk',
                    'obesity_related_risk': 'Depends on BMI',
                    'recommendations': [
                        'Maintain healthy diet and regular exercise',
                        'Continue routine health check-ups annually',
                        'Monitor glucose levels periodically'
                    ]
                })
        
        return results
    
    def _assess_cardiovascular_risk(self, glucose, blood_pressure, bmi, age):
        """Assess cardiovascular risk based on multiple factors"""
        risk_score = 0
        
        # Glucose contribution
        if glucose >= 126:
            risk_score += 3
        elif glucose >= 100:
            risk_score += 1
        
        # Blood pressure contribution
        if blood_pressure >= 140:
            risk_score += 3
        elif blood_pressure >= 120:
            risk_score += 2
        elif blood_pressure >= 90:
            risk_score += 1
        
        # BMI contribution
        if bmi >= 30:
            risk_score += 2
        elif bmi >= 25:
            risk_score += 1
        
        # Age contribution
        if age >= 60:
            risk_score += 3
        elif age >= 45:
            risk_score += 2
        elif age >= 30:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 7:
            return 'High'
        elif risk_score >= 4:
            return 'Moderate'
        else:
            return 'Low'
    
    def _assess_metabolic_syndrome(self, glucose, blood_pressure, bmi, insulin):
        """Assess metabolic syndrome risk"""
        # Count metabolic syndrome criteria
        criteria_count = 0
        
        if glucose >= 100:
            criteria_count += 1
        if blood_pressure >= 130:
            criteria_count += 1
        if bmi >= 30:  # Using BMI as a proxy for waist circumference
            criteria_count += 1
        if insulin > 100:  # High insulin as a marker of insulin resistance
            criteria_count += 1
        
        # Determine metabolic syndrome status
        if criteria_count >= 3:
            return 'Likely'
        elif criteria_count == 2:
            return 'Possible'
        else:
            return 'Unlikely'
    
    def _assess_beta_cell_dysfunction(self, glucose, insulin, dpf):
        """Assess beta cell dysfunction based on glucose, insulin, and diabetes pedigree function"""
        # Simple heuristic for beta cell function
        if glucose > 126 and insulin < 60:
            # High glucose with low insulin suggests beta cell dysfunction
            return 'High risk'
        elif glucose > 100 and insulin < 80:
            return 'Moderate risk'
        elif dpf > 0.8:  # High genetic predisposition
            return 'Increased risk due to genetic factors'
        else:
            return 'Low risk'
    
    def _assess_obesity_risk(self, bmi, skin_thickness):
        """Assess obesity-related risks"""
        if bmi >= 40:
            return 'Very high (Class III Obesity)'
        elif bmi >= 35:
            return 'High (Class II Obesity)'
        elif bmi >= 30:
            return 'Moderate (Class I Obesity)'
        elif bmi >= 25:
            return 'Mild (Overweight)'
        else:
            return 'Low (Normal weight)'
    
    def _generate_recommendations(self, risk_profile, glucose, bmi, blood_pressure, insulin, age):
        """Generate personalized recommendations based on risk factors"""
        recommendations = []
        
        # Glucose-related recommendations
        if glucose >= 126:
            recommendations.append('Consult with an endocrinologist for diabetes management')
            recommendations.append('Monitor blood glucose levels regularly')
            recommendations.append('Consider medication for glucose control')
        elif glucose >= 100:
            recommendations.append('Implement lifestyle changes to prevent progression to diabetes')
            recommendations.append('Follow up with glucose testing in 3-6 months')
        
        # BMI-related recommendations
        if bmi >= 30:
            recommendations.append('Weight management program recommended')
            recommendations.append('Consider consultation with a nutritionist')
            if bmi >= 35:
                recommendations.append('Evaluate eligibility for more intensive weight management interventions')
        elif bmi >= 25:
            recommendations.append('Aim for 5-10% weight reduction through diet and exercise')
        
        # Blood pressure recommendations
        if blood_pressure >= 140:
            recommendations.append('Urgent blood pressure management required')
            recommendations.append('Consider medication for hypertension')
        elif blood_pressure >= 120:
            recommendations.append('Implement DASH diet and sodium restriction')
            recommendations.append('Regular blood pressure monitoring')
        
        # Cardiovascular recommendations
        if risk_profile['cardiovascular_risk'] == 'High':
            recommendations.append('Comprehensive cardiovascular risk assessment recommended')
            recommendations.append('Consider statin therapy if indicated')
            recommendations.append('Aspirin therapy may be beneficial (discuss with doctor)')
        
        # Metabolic syndrome recommendations
        if risk_profile['metabolic_syndrome'] == 'Likely':
            recommendations.append('Address all components of metabolic syndrome')
            recommendations.append('Focus on waist circumference reduction')
        
        # Age-specific recommendations
        if age >= 45:
            recommendations.append('Regular screening for diabetes complications')
            if age >= 60:
                recommendations.append('Assess for age-related factors affecting diabetes management')
        
        # Add recommendations to the risk profile
        risk_profile['recommendations'] = recommendations
    
    def analyze_patient(self, patient_data):
        """
        Analyze a single patient using the two-stage approach
        patient_data should be a preprocessed array of features
        """
        # Reshape to 2D if needed
        if patient_data.ndim == 1:
            patient_data = patient_data.reshape(1, -1)
        
        # Stage 1: Initial screening
        flags, probabilities = self.initial_screening(patient_data)
        
        # Stage 2: Detailed analysis
        risk_profiles = self.detailed_analysis(patient_data, flags)
        
        return {
            'flag': flags[0],  # 0: normal, 1: prediabetic, 2: diabetic
            'probability': probabilities[0],
            'risk_profile': risk_profiles[0]
        }
    
    def visualize_glucose_distribution(self, X, y):
        """Visualize the distribution of glucose values for diabetic and non-diabetic patients"""
        glucose_values = X[:, self.glucose_idx]
        
        plt.figure(figsize=(10, 6))
        plt.hist([glucose_values[y == 0], glucose_values[y == 1]], 
                 bins=20, alpha=0.7, label=['Non-diabetic', 'Diabetic'])
        
        plt.axvline(x=self.glucose_prediabetic_threshold, color='orange', linestyle='--', 
                   label=f'Prediabetic threshold ({self.glucose_prediabetic_threshold})')
        plt.axvline(x=self.glucose_high_threshold, color='red', linestyle='--', 
                   label=f'Diabetic threshold ({self.glucose_high_threshold})')
        
        plt.xlabel('Glucose Level')
        plt.ylabel('Count')
        plt.title('Distribution of Glucose Levels by Diabetes Status')
        plt.legend()
        plt.tight_layout()
        plt.savefig('glucose_distribution.png')
        plt.close()