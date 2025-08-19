import unittest
import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neonatal_ner import NeonatalNER
from entity_patterns import EntityPatterns
from evaluation import EntityEvaluator

class TestNeonatalNER(unittest.TestCase):
    """Test cases for the NeonatalNER system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ner = NeonatalNER()
        self.sample_text = """
        Patient ID: NB-2023-001. Male infant born at 34 weeks gestational age 
        with birth weight of 2100 grams. Diagnosis: Respiratory distress syndrome, 
        Patent ductus arteriosus. Treatment included mechanical ventilation and CPAP support. 
        Medications administered: Surfactant, Caffeine citrate, Indomethacin. 
        Patient discharged home in stable condition.
        """
        
        self.expected_entities = {
            "P_ID": "NB-2023-001",
            "Gestational_Age": "34 weeks",
            "Sex": "Male",
            "Birth_Weight": "2100 grams",
            "Diagnosis": "Respiratory distress syndrome, Patent ductus arteriosus",
            "Treatment_Respiratory": "mechanical ventilation; CPAP support",
            "Treatment_Medication": "Surfactant; Caffeine citrate; Indomethacin",
            "Outcome": "discharged home in stable condition"
        }
    
    def test_patient_id_extraction(self):
        """Test patient ID extraction"""
        entities = self.ner._extract_all_entities(self.sample_text)
        self.assertEqual(entities.P_ID, "NB-2023-001")
    
    def test_gestational_age_extraction(self):
        """Test gestational age extraction"""
        entities = self.ner._extract_all_entities(self.sample_text)
        self.assertEqual(entities.Gestational_Age, "34 weeks")
    
    def test_sex_extraction(self):
        """Test sex extraction"""
        entities = self.ner._extract_all_entities(self.sample_text)
        self.assertEqual(entities.Sex, "Male")
    
    def test_birth_weight_extraction(self):
        """Test birth weight extraction"""
        entities = self.ner._extract_all_entities(self.sample_text)
        self.assertEqual(entities.Birth_Weight, "2100 grams")
    
    def test_json_output_format(self):
        """Test JSON output format"""
        result = self.ner.extract_entities(self.sample_text)
        json_output = result['json_output']
        
        # Should be valid JSON
        parsed_json = json.loads(json_output)
        self.assertIsInstance(parsed_json, dict)
        
        # Should contain all entity keys
        expected_keys = [
            'P_ID', 'Gestational_Age', 'Sex', 'Birth_Weight',
            'Diagnosis', 'Treatment_Respiratory', 'Treatment_Medication', 'Outcome'
        ]
        for key in expected_keys:
            self.assertIn(key, parsed_json)
    
    def test_markdown_table_format(self):
        """Test Markdown table format"""
        result = self.ner.extract_entities(self.sample_text)
        markdown_table = result['markdown_table']
        
        # Should contain table headers
        self.assertIn('P_ID', markdown_table)
        self.assertIn('Gestational_Age', markdown_table)
        self.assertIn('|', markdown_table)  # Table delimiter
        self.assertIn('---', markdown_table)  # Table separator
    
    def test_evaluation_with_ground_truth(self):
        """Test evaluation metrics calculation"""
        result = self.ner.extract_entities(self.sample_text, self.expected_entities)
        
        self.assertIn('evaluation', result)
        evaluation = result['evaluation']
        
        # Should have required evaluation metrics
        self.assertIn('precision', evaluation)
        self.assertIn('recall', evaluation)
        self.assertIn('f1_score', evaluation)
        
        # Metrics should be between 0 and 1
        self.assertGreaterEqual(evaluation['precision'], 0)
        self.assertLessEqual(evaluation['precision'], 1)
        self.assertGreaterEqual(evaluation['recall'], 0)
        self.assertLessEqual(evaluation['recall'], 1)
        self.assertGreaterEqual(evaluation['f1_score'], 0)
        self.assertLessEqual(evaluation['f1_score'], 1)
    
    def test_null_entity_handling(self):
        """Test handling of null entities"""
        text_with_missing_info = "Patient born at 32 weeks. Treatment with oxygen."
        result = self.ner.extract_entities(text_with_missing_info)
        
        entities = result['entities']
        # Some entities should be null
        self.assertIsNone(entities['P_ID'])
        
        # JSON should handle null values
        json_output = json.loads(result['json_output'])
        self.assertIsNone(json_output['P_ID'])
        
        # Markdown should show "null"
        self.assertIn('null', result['markdown_table'])
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        texts = [self.sample_text, "Female baby, 36 weeks GA, 2500g birth weight."]
        ground_truths = [self.expected_entities, None]
        
        results = self.ner.batch_extract(texts, ground_truths)
        
        self.assertEqual(len(results), 2)
        self.assertIn('entities', results[0])
        self.assertIn('json_output', results[0])
        self.assertIn('markdown_table', results[0])

class TestEntityPatterns(unittest.TestCase):
    """Test cases for EntityPatterns class"""
    
    def setUp(self):
        self.patterns = EntityPatterns()
    
    def test_pattern_categories(self):
        """Test that all pattern categories are present"""
        all_patterns = self.patterns.get_all_patterns()
        expected_categories = [
            'patient_id', 'gestational_age', 'sex', 'birth_weight',
            'diagnosis', 'respiratory_treatment', 'medication', 'outcome'
        ]
        
        for category in expected_categories:
            self.assertIn(category, all_patterns)
            self.assertGreater(len(all_patterns[category]), 0)
    
    def test_patient_id_patterns(self):
        """Test patient ID pattern matching"""
        import re
        
        test_texts = [
            "Patient ID: NB-2023-001",
            "Medical record number: 12345",
            "ID: ABC123"
        ]
        
        for text in test_texts:
            matched = False
            for pattern in self.patterns.patient_id_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    matched = True
                    break
            self.assertTrue(matched, f"No pattern matched for: {text}")

class TestEntityEvaluator(unittest.TestCase):
    """Test cases for EntityEvaluator class"""
    
    def setUp(self):
        self.evaluator = EntityEvaluator()
    
    def test_perfect_match_evaluation(self):
        """Test evaluation with perfect match"""
        predicted = {"P_ID": "123", "Sex": "Male", "Birth_Weight": "2000g"}
        ground_truth = {"P_ID": "123", "Sex": "Male", "Birth_Weight": "2000g"}
        
        result = self.evaluator.evaluate(predicted, ground_truth)
        
        self.assertEqual(result['precision'], 1.0)
        self.assertEqual(result['recall'], 1.0)
        self.assertEqual(result['f1_score'], 1.0)
    
    def test_no_match_evaluation(self):
        """Test evaluation with no matches"""
        predicted = {"P_ID": "456", "Sex": "Female", "Birth_Weight": "1800g"}
        ground_truth = {"P_ID": "123", "Sex": "Male", "Birth_Weight": "2000g"}
        
        result = self.evaluator.evaluate(predicted, ground_truth)
        
        self.assertEqual(result['precision'], 0.0)
        self.assertEqual(result['recall'], 0.0)
        self.assertEqual(result['f1_score'], 0.0)
    
    def test_partial_match_evaluation(self):
        """Test evaluation with partial matches"""
        predicted = {"P_ID": "123", "Sex": "Male", "Birth_Weight": "1800g"}
        ground_truth = {"P_ID": "123", "Sex": "Male", "Birth_Weight": "2000g"}
        
        result = self.evaluator.evaluate(predicted, ground_truth)
        
        # Should have some correct matches
        self.assertGreater(result['precision'], 0)
        self.assertGreater(result['recall'], 0)
        self.assertLess(result['precision'], 1)
        self.assertLess(result['recall'], 1)
    
    def test_fuzzy_matching(self):
        """Test fuzzy matching functionality"""
        # Test similar clinical terms
        self.assertTrue(self.evaluator._fuzzy_match("rds", "respiratory distress syndrome"))
        self.assertTrue(self.evaluator._fuzzy_match("mechanical ventilation", "mechanical ventilator support"))
        self.assertFalse(self.evaluator._fuzzy_match("male", "female"))
    
    def test_batch_evaluation(self):
        """Test batch evaluation"""
        predictions = [
            {"P_ID": "123", "Sex": "Male"},
            {"P_ID": "456", "Sex": "Female"}
        ]
        ground_truths = [
            {"P_ID": "123", "Sex": "Male"},
            {"P_ID": "456", "Sex": "Female"}
        ]
        
        result = self.evaluator.batch_evaluate(predictions, ground_truths)
        
        self.assertIn('aggregate_precision', result)
        self.assertIn('aggregate_recall', result)
        self.assertIn('aggregate_f1_score', result)
        self.assertIn('per_entity_metrics', result)
        self.assertEqual(result['sample_count'], 2)

if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestNeonatalNER))
    test_suite.addTest(unittest.makeSuite(TestEntityPatterns))
    test_suite.addTest(unittest.makeSuite(TestEntityEvaluator))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, failure in result.failures:
            print(f"- {test}: {failure}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, error in result.errors:
            print(f"- {test}: {error}")
