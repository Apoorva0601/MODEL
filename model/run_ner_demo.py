#!/usr/bin/env python3
"""
Demo script to run the Neonatal NER system
"""

from neonatal_ner import NeonatalNER
from entity_patterns import EntityPatterns
import json

def main():
    print("🏥 RDS-Specialized Clinical NER System Demo")
    print("=" * 50)
    
    # Initialize the NER system
    ner = NeonatalNER()
    
    # Sample RDS discharge summary
    sample_text = """
    Patient ID: RDS-2025-042. Female infant born at 28 weeks gestational age with birth weight of 1200 grams.
    Diagnosis: Severe respiratory distress syndrome with bilateral ground-glass opacities.
    Treatment included high-frequency oscillatory ventilation (HFOV) and surfactant administration.
    Medications administered: Poractant alfa 200mg/kg, Caffeine citrate 20mg/kg, Dexamethasone 0.15mg/kg.
    Complications: Pneumothorax requiring chest tube insertion.
    Patient developed chronic lung disease and was discharged home on low-flow oxygen support at 38 weeks corrected gestational age.
    """
    
    # Ground truth for evaluation
    ground_truth = {
        "P_ID": "RDS-2025-042",
        "Gestational_Age": "28 weeks",
        "Sex": "Female",
        "Birth_Weight": "1200 grams",
        "Diagnosis": "Severe respiratory distress syndrome with bilateral ground-glass opacities",
        "Treatment_Respiratory": "high-frequency oscillatory ventilation (HFOV); surfactant administration",
        "Treatment_Medication": "Poractant alfa 200mg/kg; Caffeine citrate 20mg/kg; Dexamethasone 0.15mg/kg",
        "Outcome": "discharged home on low-flow oxygen support at 38 weeks corrected gestational age"
    }
    
    print("\n📄 Sample RDS Discharge Summary:")
    print("-" * 40)
    print(sample_text.strip())
    
    print("\n🔍 Extracting Entities...")
    print("-" * 40)
    
    # Extract entities
    result = ner.extract_entities(sample_text, ground_truth)
    
    # Print results
    print("\n✅ Extraction Results:")
    ner.print_results(result)
    
    # Additional RDS-specific examples
    print("\n\n🫁 Additional RDS Examples:")
    print("=" * 50)
    
    # Example 2: Moderate RDS
    rds_text_2 = """
    Patient: NB-RDS-123, Male, 32+4 weeks GA, BW: 1800g.
    Dx: Moderate RDS, requires INSURE technique.
    Rx: Beractant 100mg/kg via ETT, followed by nCPAP 6cmH2O.
    Meds: Caffeine 10mg/kg loading dose.
    Outcome: Successful extubation, stable on CPAP, feeding well.
    """
    
    print("\n📄 Example 2 - Moderate RDS:")
    print(rds_text_2.strip())
    
    result2 = ner.extract_entities(rds_text_2)
    print("\n🔍 Extraction Results:")
    print(f"📋 JSON Format:\n{result2['json_output']}")
    
    print("\n📊 Markdown Table:")
    print(result2['markdown_table'])
    
    # Pattern information
    patterns = EntityPatterns()
    all_patterns = patterns.get_all_patterns()
    
    print(f"\n🔧 System Specifications:")
    print(f"• Total Patterns: {sum(len(v) for v in all_patterns.values())}")
    print(f"• RDS-Specific Respiratory Patterns: {len(all_patterns['respiratory_treatment'])}")
    print(f"• Medication Patterns: {len(all_patterns['medication'])}")
    print(f"• Diagnosis Patterns: {len(all_patterns['diagnosis'])}")
    
    print(f"\n🎯 RDS Entity Categories:")
    for category, pattern_list in all_patterns.items():
        print(f"  • {category.replace('_', ' ').title()}: {len(pattern_list)} patterns")
    
    print(f"\n🚀 System Ready for Clinical Use!")
    print(f"   Access the web interface at: http://localhost:8503")

if __name__ == "__main__":
    main()
