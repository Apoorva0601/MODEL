from typing import Dict, List, Tuple
import json

class EntityEvaluator:
    """
    Evaluation framework for neonatal NER system
    """
    
    def __init__(self):
        self.entity_keys = [
            'P_ID', 'Gestational_Age', 'Sex', 'Birth_Weight',
            'Diagnosis', 'Treatment_Respiratory', 'Treatment_Medication', 'Outcome'
        ]
    
    def evaluate(self, predicted: Dict, ground_truth: Dict) -> Dict:
        """
        Evaluate predicted entities against ground truth
        
        Args:
            predicted (Dict): Predicted entities from the model
            ground_truth (Dict): Ground truth annotations
            
        Returns:
            Dict: Evaluation metrics including precision, recall, and F1-score
        """
        # Count correct extractions
        correct_extractions = 0
        total_extractions = 0
        total_actual = 0
        
        entity_results = {}
        
        for entity_key in self.entity_keys:
            pred_value = predicted.get(entity_key)
            true_value = ground_truth.get(entity_key)
            
            # Count actual entities (non-null ground truth)
            if true_value is not None and true_value.strip() != "":
                total_actual += 1
            
            # Count extractions (non-null predictions)
            if pred_value is not None and pred_value.strip() != "":
                total_extractions += 1
                
                # Check if extraction is correct
                if self._is_correct_extraction(pred_value, true_value):
                    correct_extractions += 1
                    entity_results[entity_key] = "Correct"
                else:
                    entity_results[entity_key] = "Incorrect"
            else:
                if true_value is not None and true_value.strip() != "":
                    entity_results[entity_key] = "Missed"
                else:
                    entity_results[entity_key] = "Not applicable"
        
        # Calculate metrics
        precision = correct_extractions / total_extractions if total_extractions > 0 else 0.0
        recall = correct_extractions / total_actual if total_actual > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'correct_extractions': correct_extractions,
            'total_extractions': total_extractions,
            'total_actual': total_actual,
            'entity_results': entity_results
        }
    
    def _is_correct_extraction(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if predicted value matches ground truth
        Uses fuzzy matching for clinical text
        """
        if ground_truth is None:
            return predicted is None
        
        if predicted is None:
            return False
        
        # Normalize strings for comparison
        pred_norm = self._normalize_text(predicted)
        true_norm = self._normalize_text(ground_truth)
        
        # Exact match
        if pred_norm == true_norm:
            return True
        
        # Fuzzy matching for clinical terms
        return self._fuzzy_match(pred_norm, true_norm)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if text is None:
            return ""
        
        text = text.lower().strip()
        # Remove punctuation and extra spaces
        import re
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _fuzzy_match(self, pred: str, true: str) -> bool:
        """
        Fuzzy matching for clinical terms
        Returns True if the predicted value is substantially similar to ground truth
        """
        # Simple containment check - if predicted contains key parts of ground truth
        pred_words = set(pred.split())
        true_words = set(true.split())
        
        if len(true_words) == 0:
            return len(pred_words) == 0
        
        # Calculate overlap
        overlap = len(pred_words.intersection(true_words))
        overlap_ratio = overlap / len(true_words)
        
        # Consider it a match if 70% of ground truth words are present
        return overlap_ratio >= 0.7
    
    def batch_evaluate(self, predicted_list: List[Dict], ground_truth_list: List[Dict]) -> Dict:
        """
        Evaluate multiple predictions
        
        Args:
            predicted_list (List[Dict]): List of predicted entity dictionaries
            ground_truth_list (List[Dict]): List of ground truth dictionaries
            
        Returns:
            Dict: Aggregate evaluation metrics
        """
        total_correct = 0
        total_extractions = 0
        total_actual = 0
        
        individual_results = []
        
        for pred, gt in zip(predicted_list, ground_truth_list):
            result = self.evaluate(pred, gt)
            individual_results.append(result)
            
            total_correct += result['correct_extractions']
            total_extractions += result['total_extractions']
            total_actual += result['total_actual']
        
        # Calculate aggregate metrics
        precision = total_correct / total_extractions if total_extractions > 0 else 0.0
        recall = total_correct / total_actual if total_actual > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate per-entity metrics
        entity_metrics = self._calculate_per_entity_metrics(individual_results)
        
        return {
            'aggregate_precision': precision,
            'aggregate_recall': recall,
            'aggregate_f1_score': f1_score,
            'total_correct_extractions': total_correct,
            'total_extractions': total_extractions,
            'total_actual_entities': total_actual,
            'individual_results': individual_results,
            'per_entity_metrics': entity_metrics,
            'sample_count': len(predicted_list)
        }
    
    def _calculate_per_entity_metrics(self, individual_results: List[Dict]) -> Dict:
        """Calculate metrics for each entity type"""
        entity_metrics = {}
        
        for entity_key in self.entity_keys:
            correct = 0
            total_pred = 0
            total_actual = 0
            
            for result in individual_results:
                entity_result = result['entity_results'].get(entity_key, "Not applicable")
                
                if entity_result == "Correct":
                    correct += 1
                    total_pred += 1
                    total_actual += 1
                elif entity_result == "Incorrect":
                    total_pred += 1
                    total_actual += 1
                elif entity_result == "Missed":
                    total_actual += 1
            
            precision = correct / total_pred if total_pred > 0 else 0.0
            recall = correct / total_actual if total_actual > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            entity_metrics[entity_key] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'correct': correct,
                'total_predicted': total_pred,
                'total_actual': total_actual
            }
        
        return entity_metrics
    
    def print_detailed_evaluation(self, evaluation_result: Dict):
        """Print detailed evaluation results"""
        print("\n" + "="*80)
        print("DETAILED EVALUATION RESULTS")
        print("="*80)
        
        print(f"\nOVERALL METRICS:")
        print(f"Precision: {evaluation_result['precision']:.3f} ({evaluation_result['correct_extractions']}/{evaluation_result['total_extractions']})")
        print(f"Recall: {evaluation_result['recall']:.3f} ({evaluation_result['correct_extractions']}/{evaluation_result['total_actual']})")
        print(f"F1-Score: {evaluation_result['f1_score']:.3f}")
        
        print(f"\nPER-ENTITY RESULTS:")
        for entity, result in evaluation_result['entity_results'].items():
            print(f"{entity:20}: {result}")
        
        print("\nEVALUATION SUMMARY:")
        print(f"✓ Correct extractions: {evaluation_result['correct_extractions']}")
        print(f"✗ Total extractions: {evaluation_result['total_extractions']}")
        print(f"📋 Total actual entities: {evaluation_result['total_actual']}")

# Example usage for testing
if __name__ == "__main__":
    evaluator = EntityEvaluator()
    
    # Sample prediction and ground truth
    predicted = {
        "P_ID": "NB-2023-001",
        "Gestational_Age": "34 weeks",
        "Sex": "Male",
        "Birth_Weight": "2100 grams",
        "Diagnosis": "RDS, PDA",
        "Treatment_Respiratory": "mechanical ventilation",
        "Treatment_Medication": "surfactant",
        "Outcome": "discharged stable"
    }
    
    ground_truth = {
        "P_ID": "NB-2023-001",
        "Gestational_Age": "34 weeks",
        "Sex": "Male",
        "Birth_Weight": "2100 grams",
        "Diagnosis": "Respiratory distress syndrome, Patent ductus arteriosus",
        "Treatment_Respiratory": "mechanical ventilation and CPAP",
        "Treatment_Medication": "Surfactant, Caffeine citrate",
        "Outcome": "discharged home in stable condition"
    }
    
    # Evaluate
    result = evaluator.evaluate(predicted, ground_truth)
    evaluator.print_detailed_evaluation(result)
