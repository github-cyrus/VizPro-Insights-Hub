import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier

class InsightPredictionModel:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.predictions = None
        self.task_type = None  # 'regression' or 'classification'
        self.scaler = StandardScaler()
        
    def load_data(self, data, target_column, test_size=0.2, random_state=42):
        """Load and split the data"""
        self.data = data
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        # Determine task type
        if len(np.unique(y)) <= 10 and all(isinstance(val, (int, np.integer)) for val in y.unique()):
            self.task_type = 'classification'
        else:
            self.task_type = 'regression'
            
        print(f"Detected task type: {self.task_type}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Data loaded and split: {self.X_train.shape[0]} training samples, {self.X_test.shape[0]} test samples")
        
    def select_model(self, model_name):
        """Select and initialize the model based on task type"""
        models = {
            'regression': {
                'linear': LinearRegression(),
                'decision_tree': DecisionTreeRegressor(random_state=42),
                'random_forest': RandomForestRegressor(random_state=42),
                'svr': SVR(),
                'knn': KNeighborsRegressor(),
                'xgboost': XGBRegressor(random_state=42)
            },
            'classification': {
                'logistic': LogisticRegression(random_state=42),
                'decision_tree': DecisionTreeClassifier(random_state=42),
                'random_forest': RandomForestClassifier(random_state=42),
                'svc': SVC(probability=True, random_state=42),
                'knn': KNeighborsClassifier(),
                'xgboost': XGBClassifier(random_state=42)
            }
        }
        
        if model_name not in models[self.task_type]:
            valid_models = list(models[self.task_type].keys())
            raise ValueError(f"Model '{model_name}' not found for {self.task_type}. Choose from: {valid_models}")
        
        self.model = models[self.task_type][model_name]
        print(f"Selected model: {model_name} for {self.task_type}")
        
    def train_model(self):
        """Train the selected model"""
        if self.model is None:
            raise ValueError("No model selected. Use select_model() first.")
        
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed")
        
    def evaluate_model(self):
        """Evaluate the model and return metrics"""
        if self.model is None:
            raise ValueError("No model trained. Use train_model() first.")
        
        self.predictions = self.model.predict(self.X_test)
        
        if self.task_type == 'regression':
            mse = mean_squared_error(self.y_test, self.predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, self.predictions)
            r2 = r2_score(self.y_test, self.predictions)
            
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"Root Mean Squared Error: {rmse:.4f}")
            print(f"Mean Absolute Error: {mae:.4f}")
            print(f"R² Score: {r2:.4f}")
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='r2')
            print(f"Cross-Validation R² Scores: {cv_scores}")
            print(f"Mean CV R² Score: {np.mean(cv_scores):.4f}")
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores)
            }
            
        else:  # classification
            acc = accuracy_score(self.y_test, self.predictions)
            
            # Handle binary and multi-class cases
            if len(np.unique(self.y_train)) == 2:
                prec = precision_score(self.y_test, self.predictions)
                rec = recall_score(self.y_test, self.predictions)
                f1 = f1_score(self.y_test, self.predictions)
                print(f"Accuracy: {acc:.4f}")
                print(f"Precision: {prec:.4f}")
                print(f"Recall: {rec:.4f}")
                print(f"F1 Score: {f1:.4f}")
                
                metrics = {
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1
                }
            else:
                # For multiclass, use weighted average
                prec = precision_score(self.y_test, self.predictions, average='weighted')
                rec = recall_score(self.y_test, self.predictions, average='weighted')
                f1 = f1_score(self.y_test, self.predictions, average='weighted')
                print(f"Accuracy: {acc:.4f}")
                print(f"Weighted Precision: {prec:.4f}")
                print(f"Weighted Recall: {rec:.4f}")
                print(f"Weighted F1 Score: {f1:.4f}")
                
                metrics = {
                    'accuracy': acc,
                    'weighted_precision': prec,
                    'weighted_recall': rec,
                    'weighted_f1': f1
                }
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            print(f"Cross-Validation Accuracy Scores: {cv_scores}")
            print(f"Mean CV Accuracy Score: {np.mean(cv_scores):.4f}")
            
            metrics['cv_scores'] = cv_scores
            metrics['cv_mean'] = np.mean(cv_scores)
            
            return metrics
    
    def visualize_results(self):
        """Create visualizations based on model results"""
        plt.figure(figsize=(15, 10))
        
        if self.task_type == 'regression':
            # Actual vs Predicted
            plt.subplot(2, 2, 1)
            plt.scatter(self.y_test, self.predictions)
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values')
            
            # Residuals
            plt.subplot(2, 2, 2)
            residuals = self.y_test - self.predictions
            plt.scatter(self.predictions, residuals)
            plt.axhline(y=0, color='k', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            
            # Residual distribution
            plt.subplot(2, 2, 3)
            sns.histplot(residuals, kde=True)
            plt.xlabel('Residual Value')
            plt.ylabel('Frequency')
            plt.title('Residual Distribution')
            
            # Feature importance if available
            plt.subplot(2, 2, 4)
            if hasattr(self.model, 'feature_importances_'):
                feature_names = self.data.drop(self.data.columns[-1], axis=1).columns
                importances = self.model.feature_importances_
                indices = np.argsort(importances)
                plt.barh(range(len(indices)), importances[indices], align='center')
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel('Relative Importance')
                plt.title('Feature Importance')
            else:
                plt.text(0.5, 0.5, "Feature importance not available for this model", 
                         horizontalalignment='center', verticalalignment='center')
            
        else:  # classification
            # Confusion Matrix
            plt.subplot(2, 2, 1)
            cm = confusion_matrix(self.y_test, self.predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            
            # Class distribution in training set
            plt.subplot(2, 2, 2)
            sns.countplot(y=self.y_train)
            plt.xlabel('Count')
            plt.ylabel('Class')
            plt.title('Class Distribution (Training Set)')
            
            # Feature importance if available
            plt.subplot(2, 2, 3)
            if hasattr(self.model, 'feature_importances_'):
                feature_names = self.data.drop(self.data.columns[-1], axis=1).columns
                importances = self.model.feature_importances_
                indices = np.argsort(importances)
                plt.barh(range(len(indices)), importances[indices], align='center')
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel('Relative Importance')
                plt.title('Feature Importance')
            else:
                plt.text(0.5, 0.5, "Feature importance not available for this model", 
                         horizontalalignment='center', verticalalignment='center')
            
            # ROC curve for binary classification
            plt.subplot(2, 2, 4)
            if len(np.unique(self.y_test)) == 2 and hasattr(self.model, 'predict_proba'):
                from sklearn.metrics import roc_curve, auc
                y_pred_proba = self.model.predict_proba(self.X_test)[:,1]
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc='lower right')
            else:
                plt.text(0.5, 0.5, "ROC curve available only for binary classification with probability support", 
                         horizontalalignment='center', verticalalignment='center', wrap=True)
        
        plt.tight_layout()
        plt.show()
        
    def make_predictions(self, new_data):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("No model trained. Use train_model() first.")
        
        # Scale the new data
        scaled_data = self.scaler.transform(new_data)
        
        # Make predictions
        predictions = self.model.predict(scaled_data)
        
        return predictions