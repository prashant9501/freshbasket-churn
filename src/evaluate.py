""" 
FreshBasket Churn Prediction - Model Evaluation Module 
Author: Arjun 
""" 
from sklearn.metrics import accuracy_score, precision_score, recall_score, 
f1_score 
from sklearn.metrics import confusion_matrix, classification_report 
  
  
def evaluate_model(y_true, y_pred): 
    """Compute standard classification metrics.""" 
    metrics = { 
        'accuracy': accuracy_score(y_true, y_pred), 
        'precision': precision_score(y_true, y_pred), 
        'recall': recall_score(y_true, y_pred), 
        'f1_score': f1_score(y_true, y_pred), 
    } 
    print(classification_report(y_true, y_pred)) 
    return metrics 
  
  
if __name__ == '__main__': 
    print('Evaluation module loaded successfully.') 
