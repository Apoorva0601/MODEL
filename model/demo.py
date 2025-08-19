#!/usr/bin/env python3
"""
Clinical NLP Demo Script for Neonatal Discharge Summary Entity Extraction

This script demonstrates the complete functionality of the clinical NLP system
for extracting entities from neonatal discharge summaries.
"""

import sys
import os
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neonatal_ner import NeonatalNER
from entity_patterns import EntityPatterns
from evaluation import EntityEvaluator

def main():
    """Main demonstration function"""
    print("=" * 80)
    print("CLINICAL NLP SYSTEM FOR NEONATAL DISCHARGE SUMMARY ENTITY EXTRACTION")
    print("=" * 80)
    
    # Initialize the system
    print("\n1. Initializing the NLP system...")
    ner_system = NeonatalNER()
    print("✓ System initialized successfully")
    
    # Load sample data
    print("\n2. Loading sample data...")
    try:
        with open('data/sample_annotations.json', 'r') as f:
            sample_data = json.load(f)
        print(f"✓ Loaded {len(sample_data)} sample texts")
    except FileNotFoundError:
        print("✗ Sample data file not found. Using built-in example.")
        sample_data = [get_built_in_example()]
    
    # Extract sample text and ground truth
    sample_text = sample_data[0]['text']
    ground_truth = sample_data[0]['annotations']
    
    print(f"\nSample text: {sample_text[:100]}...")
    
    # Run entity extraction
    print("\n3. Running entity extraction...")
    result = ner_system.extract_entities(sample_text, ground_truth)
    
    # Display results using the system's built-in formatter
    ner_system.print_results(result)
    
    # Process all samples if available
    if len(sample_data) > 1:
        print("\n4. Processing all samples...")
        all_texts = [item['text'] for item in sample_data]
        all_ground_truths = [item['annotations'] for item in sample_data]
        
        batch_results = ner_system.batch_extract(all_texts, all_ground_truths)
        
        # Calculate aggregate metrics
        evaluator = EntityEvaluator()
        all_predictions = [result['entities'] for result in batch_results]
        batch_evaluation = evaluator.batch_evaluate(all_predictions, all_ground_truths)
        
        print(f"\nBATCH PROCESSING RESULTS ({len(sample_data)} samples):")
        print("-" * 60)
        print(f"Overall Precision: {batch_evaluation['aggregate_precision']:.3f}")
        print(f"Overall Recall: {batch_evaluation['aggregate_recall']:.3f}")
        print(f"Overall F1-Score: {batch_evaluation['aggregate_f1_score']:.3f}")
        
        print(f"\nPer-Entity Performance:")
        for entity, metrics in batch_evaluation['per_entity_metrics'].items():
            print(f"{entity:20}: F1={metrics['f1_score']:.3f} "
                  f"(P={metrics['precision']:.3f}, R={metrics['recall']:.3f})")
    
    # Interactive demo
    print("\n5. Interactive Demo")
    print("-" * 40)
    print("Enter your own neonatal discharge summary text (or 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input.strip():
                # Extract entities from user input
                user_result = ner_system.extract_entities(user_input)
                
                print("\nExtraction Results:")
                print("=" * 40)
                print(user_result['json_output'])
                print("\nMarkdown Table:")
                print(user_result['markdown_table'])
            else:
                print("Please enter some text or 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error processing input: {e}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETED")
    print("=" * 80)
    print("Thank you for using the Clinical NLP System!")
    print("For more information, see the README.md file.")

def get_built_in_example():
    """Return a built-in example if sample data is not available"""
    return {
        "text": "Patient ID: NB-2023-001. Male infant born at 34 weeks gestational age with birth weight of 2100 grams. Diagnosis: Respiratory distress syndrome, Patent ductus arteriosus. Treatment included mechanical ventilation and CPAP support. Medications administered: Surfactant, Caffeine citrate, Indomethacin. Patient discharged home in stable condition.",
        "annotations": {
            "P_ID": "NB-2023-001",
            "Gestational_Age": "34 weeks",
            "Sex": "Male",
            "Birth_Weight": "2100 grams",
            "Diagnosis": "Respiratory distress syndrome, Patent ductus arteriosus",
            "Treatment_Respiratory": "mechanical ventilation and CPAP support",
            "Treatment_Medication": "Surfactant, Caffeine citrate, Indomethacin",
            "Outcome": "discharged home in stable condition"
        }
    }

def run_tests():
    """Run the test suite"""
    print("Running test suite...")
    os.system("python tests/test_neonatal_ner.py")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            run_tests()
        elif sys.argv[1] == "--help":
            print("Usage: python demo.py [--test] [--help]")
            print("  --test: Run the test suite")
            print("  --help: Show this help message")
        else:
            print("Unknown argument. Use --help for usage information.")
    else:
        main()
