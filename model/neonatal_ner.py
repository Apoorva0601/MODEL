import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from entity_patterns import EntityPatterns
from evaluation import EntityEvaluator

@dataclass
class EntityResult:
    """Data class to hold extracted entities"""
    P_ID: Optional[str] = None
    Gestational_Age: Optional[str] = None
    Sex: Optional[str] = None
    Birth_Weight: Optional[str] = None
    Diagnosis: Optional[str] = None
    Treatment_Respiratory: Optional[str] = None
    Treatment_Medication: Optional[str] = None
    Outcome: Optional[str] = None

class NeonatalNER:
    """
    Clinical NLP system for extracting entities from neonatal discharge summaries
    """
    
    def __init__(self):
        self.patterns = EntityPatterns()
        self.evaluator = EntityEvaluator()
        
    def extract_entities(self, text: str, ground_truth: Optional[Dict] = None) -> Dict:
        """
        Extract entities from neonatal discharge summary text
        
        Args:
            text (str): Input discharge summary text
            ground_truth (Dict, optional): Ground truth annotations for evaluation
            
        Returns:
            Dict: Complete result including entities, JSON, markdown, and evaluation
        """
        # Clean and preprocess text
        text = self._preprocess_text(text)
        
        # Extract entities using pattern matching
        entities = self._extract_all_entities(text)
        
        # Convert to JSON format
        json_output = self._to_json(entities)
        
        # Convert to Markdown table
        markdown_table = self._to_markdown_table(entities)
        
        # Prepare result
        result = {
            "entities": entities.__dict__,
            "json_output": json_output,
            "markdown_table": markdown_table
        }
        
        # Add evaluation if ground truth provided
        if ground_truth:
            evaluation = self.evaluator.evaluate(entities.__dict__, ground_truth)
            result["evaluation"] = evaluation
        
        return result
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Normalize common abbreviations
        text = re.sub(r'\bpt\b', 'patient', text, flags=re.IGNORECASE)
        text = re.sub(r'\bwks?\b', 'weeks', text, flags=re.IGNORECASE)
        text = re.sub(r'\bgms?\b', 'grams', text, flags=re.IGNORECASE)
        return text
    
    def _extract_all_entities(self, text: str) -> EntityResult:
        """Extract all entities from the text"""
        entities = EntityResult()
        
        # Extract each entity type
        entities.P_ID = self._extract_patient_id(text)
        entities.Gestational_Age = self._extract_gestational_age(text)
        entities.Sex = self._extract_sex(text)
        entities.Birth_Weight = self._extract_birth_weight(text)
        entities.Diagnosis = self._extract_diagnosis(text)
        entities.Treatment_Respiratory = self._extract_respiratory_treatment(text)
        entities.Treatment_Medication = self._extract_medication_treatment(text)
        entities.Outcome = self._extract_outcome(text)
        
        return entities
    
    def _extract_patient_id(self, text: str) -> Optional[str]:
        """Extract patient ID"""
        for pattern in self.patterns.patient_id_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_gestational_age(self, text: str) -> Optional[str]:
        """Extract gestational age"""
        for pattern in self.patterns.gestational_age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_sex(self, text: str) -> Optional[str]:
        """Extract sex/gender"""
        for pattern in self.patterns.sex_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                sex = match.group(1).lower()
                if sex in ['male', 'boy', 'm']:
                    return 'Male'
                elif sex in ['female', 'girl', 'f']:
                    return 'Female'
        return None
    
    def _extract_birth_weight(self, text: str) -> Optional[str]:
        """Extract birth weight"""
        for pattern in self.patterns.birth_weight_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                weight = match.group(1).strip()
                # Normalize weight format
                if re.search(r'\d+\.?\d*', weight):
                    return weight
        return None
    
    def _extract_diagnosis(self, text: str) -> Optional[str]:
        """Extract diagnosis"""
        # Look for diagnosis patterns
        for pattern in self.patterns.diagnosis_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Check if pattern has capture groups
                if match.groups():
                    diagnosis = match.group(1).strip()
                else:
                    diagnosis = match.group().strip()
                # Clean up diagnosis text
                diagnosis = re.sub(r'^[:\-\.]', '', diagnosis).strip()
                return diagnosis
        return None
    
    def _extract_respiratory_treatment(self, text: str) -> Optional[str]:
        """Extract respiratory treatment"""
        treatments = []
        for pattern in self.patterns.respiratory_treatment_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                treatment = match.group().strip()
                if treatment not in treatments:
                    treatments.append(treatment)
        
        return '; '.join(treatments) if treatments else None
    
    def _extract_medication_treatment(self, text: str) -> Optional[str]:
        """Extract medication treatment"""
        medications = []
        for pattern in self.patterns.medication_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                medication = match.group().strip()
                if medication not in medications:
                    medications.append(medication)
        
        return '; '.join(medications) if medications else None
    
    def _extract_outcome(self, text: str) -> Optional[str]:
        """Extract outcome"""
        for pattern in self.patterns.outcome_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                outcome = match.group(1).strip()
                return outcome
        return None
    
    def _to_json(self, entities: EntityResult) -> str:
        """Convert entities to JSON format"""
        return json.dumps(entities.__dict__, indent=2)
    
    def _to_markdown_table(self, entities: EntityResult) -> str:
        """Convert entities to Markdown table format"""
        headers = [
            "P_ID", "Gestational_Age", "Sex", "Birth_Weight", 
            "Diagnosis", "Treatment_Respiratory", "Treatment_Medication", "Outcome"
        ]
        
        values = [
            entities.P_ID or "null",
            entities.Gestational_Age or "null",
            entities.Sex or "null",
            entities.Birth_Weight or "null",
            entities.Diagnosis or "null",
            entities.Treatment_Respiratory or "null",
            entities.Treatment_Medication or "null",
            entities.Outcome or "null"
        ]
        
        # Create markdown table
        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_row = "| " + " | ".join(values) + " |"
        
        return f"{header_row}\n{separator_row}\n{data_row}"
    
    def batch_extract(self, texts: List[str], ground_truths: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Extract entities from multiple texts
        
        Args:
            texts (List[str]): List of discharge summary texts
            ground_truths (List[Dict], optional): List of ground truth annotations
            
        Returns:
            List[Dict]: List of extraction results
        """
        results = []
        for i, text in enumerate(texts):
            gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            result = self.extract_entities(text, gt)
            results.append(result)
        
        return results
    
    def print_results(self, result: Dict):
        """Pretty print extraction results"""
        print("="*80)
        print("NEONATAL DISCHARGE SUMMARY ENTITY EXTRACTION RESULTS")
        print("="*80)
        
        print("\n1. JSON FORMAT:")
        print(result["json_output"])
        
        print("\n2. MARKDOWN TABLE FORMAT:")
        print(result["markdown_table"])
        
        if "evaluation" in result:
            print("\n3. EVALUATION METRICS:")
            eval_data = result["evaluation"]
            print(f"Precision: {eval_data['precision']:.3f}")
            print(f"Recall: {eval_data['recall']:.3f}")
            print(f"F1-Score: {eval_data['f1_score']:.3f}")
            print(f"Correct Extractions: {eval_data['correct_extractions']}")
            print(f"Total Extractions: {eval_data['total_extractions']}")
            print(f"Total Actual Entities: {eval_data['total_actual']}")

# Example usage
if __name__ == "__main__":
    # Initialize the NER model
    ner = NeonatalNER()
    
    # Sample neonatal discharge summary
    sample_text = """
    Patient ID: NB-2023-001. Male infant born at 34 weeks gestational age with birth weight of 2100 grams.
    Diagnosis: Respiratory distress syndrome, Patent ductus arteriosus.
    Treatment included mechanical ventilation and CPAP support.
    Medications administered: Surfactant, Caffeine citrate, Indomethacin.
    Patient discharged home in stable condition.
    """
    
    # Ground truth for evaluation
    ground_truth = {
        "P_ID": "NB-2023-001",
        "Gestational_Age": "34 weeks",
        "Sex": "Male",
        "Birth_Weight": "2100 grams",
        "Diagnosis": "Respiratory distress syndrome, Patent ductus arteriosus",
        "Treatment_Respiratory": "mechanical ventilation; CPAP support",
        "Treatment_Medication": "Surfactant; Caffeine citrate; Indomethacin",
        "Outcome": "discharged home in stable condition"
    }
    
    # Extract entities
    result = ner.extract_entities(sample_text, ground_truth)
    
    # Print results
    ner.print_results(result)
