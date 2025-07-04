import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class DiabetesNeuralNetwork:
    def __init__(self, input_shape, learning_rate=0.001):
        self.input_dim = input_shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()  # Changed from BCELoss for better numerical stability
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.input_dim, 32),  # Increased from 24 to 32
            nn.ReLU(),  # Changed from LeakyReLU to ReLU
            nn.BatchNorm1d(32),  # Added batch normalization
            nn.Dropout(0.3),  # Increased dropout
            nn.Linear(32, 16),  # Changed from 12 to 16
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
            # Removed Sigmoid since we're using BCEWithLogitsLoss
        )
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Early stopping parameters
        best_val_loss = float('inf')
        patience = 10
        counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy
                predicted = (torch.sigmoid(outputs) > 0.5).float()  # Apply sigmoid here
                total_train += targets.size(0)
                correct_train += (predicted == targets).sum().item()
            
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = correct_train / total_train
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor.to(self.device))
                val_loss = self.criterion(val_outputs, y_val_tensor.to(self.device)).item()
                val_predicted = (torch.sigmoid(val_outputs) > 0.5).float()  # Apply sigmoid here
                val_acc = (val_predicted == y_val_tensor.to(self.device)).sum().item() / len(y_val_tensor)
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs} - '
                  f'Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - '
                  f'Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    # Load best model
                    self.model.load_state_dict(torch.load('best_model.pth'))
                    break
        
        return history
    
    def evaluate(self, X_test, y_test):
        # Convert to PyTorch tensors
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1).to(self.device)
        
        # Evaluate
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            loss = self.criterion(outputs, y_test_tensor).item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()  # Apply sigmoid here
            accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        
        return loss, accuracy
    
    def predict(self, X):
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
        
        return probabilities.cpu().numpy()
    
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
    
    @classmethod
    def load_model(cls, filepath, input_shape):
        nn = cls(input_shape)
        nn.model.load_state_dict(torch.load(filepath))
        nn.model.eval()
        return nn