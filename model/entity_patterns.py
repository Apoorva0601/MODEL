import re
from typing import List

class EntityPatterns:
    """
    Clinical patterns for extracting entities from neonatal discharge summaries
    """
    
    def __init__(self):
        self.patient_id_patterns = [
            r'patient\s+(?:id|identifier|number)[\s:]*([A-Za-z0-9\-]+)',
            r'(?:id|patient)[\s:]*([A-Za-z0-9\-]+)',
            r'medical\s+record\s+(?:number|no)[\s:]*([A-Za-z0-9\-]+)',
            r'mrn[\s:]*([A-Za-z0-9\-]+)',
            r'nb[\-\s]*(\d{4}[\-\s]*\d{3})',  # Newborn ID format
            r'(\d{5,})',  # Generic numeric ID
        ]
        
        self.gestational_age_patterns = [
            r'(?:born\s+at|gestational\s+age|ga)[\s:]*(\d+(?:\.\d+)?\s*(?:weeks?|wks?))',
            r'(\d+(?:\.\d+)?)\s*(?:weeks?|wks?)\s*(?:gestational\s+age|gestation|ga)',
            r'(\d+(?:\+\d+)?)\s*(?:weeks?|wks?)',
            r'gestational\s+age[\s:]*(\d+(?:\.\d+)?\s*(?:weeks?|wks?))',
            r'(\d+(?:\.\d+)?)\s*week\s*preterm',
        ]
        
        self.sex_patterns = [
            r'(male|female)\s+infant',
            r'(boy|girl)',
            r'gender[\s:]*([MFmf]|male|female)',
            r'sex[\s:]*([MFmf]|male|female)',
            r'(male|female)\s+(?:baby|neonate|newborn)',
        ]
        
        self.birth_weight_patterns = [
            r'birth\s+weight[\s:]*(\d+(?:\.\d+)?\s*(?:grams?|g|kg|lbs?))',
            r'weight[\s:]*(\d+(?:\.\d+)?\s*(?:grams?|g|kg|lbs?))',
            r'(\d+(?:\.\d+)?)\s*(?:grams?|g)\s*(?:at\s+birth|birth\s+weight)',
            r'weighing\s+(\d+(?:\.\d+)?\s*(?:grams?|g|kg|lbs?))',
            r'bw[\s:]*(\d+(?:\.\d+)?\s*(?:grams?|g|kg))',
        ]
        
        self.diagnosis_patterns = [
            # General diagnosis patterns
            r'diagnosis[\s:]*([^.]+)',
            r'diagnosed\s+with[\s:]*([^.]+)',
            r'primary\s+diagnosis[\s:]*([^.]+)',
            r'impression[\s:]*([^.]+)',
            r'condition[\s:]*([^.]+)',
            
            # RDS-specific diagnosis patterns
            r'(?:mild|moderate|severe)\s+respiratory\s+distress\s+syndrome',
            r'(?:mild|moderate|severe)\s+rds',
            r'respiratory\s+distress\s+syndrome',
            r'rds',
            r'hyaline\s+membrane\s+disease',
            r'hmd',
            
            # RDS complications and associated conditions
            r'pneumothorax',
            r'pulmonary\s+interstitial\s+emphysema',
            r'pie',
            r'persistent\s+pulmonary\s+hypertension\s+of\s+the\s+newborn',
            r'pphn',
            r'chronic\s+lung\s+disease',
            r'bronchopulmonary\s+dysplasia',
            r'bpd',
            
            # Other neonatal conditions
            r'patent\s+ductus\s+arteriosus',
            r'pda',
            r'necrotizing\s+enterocolitis',
            r'nec',
            r'intraventricular\s+hemorrhage',
            r'ivh',
            r'retinopathy\s+of\s+prematurity',
            r'rop',
            r'sepsis',
            r'infection',
            r'jaundice',
            r'hyperbilirubinemia',
            r'apnea\s+of\s+prematurity',
            r'aop',
            
            # RDS severity indicators
            r'bilateral\s+ground-glass\s+opacities',
            r'ground-glass\s+appearance',
            r'air\s+bronchograms',
            r'white-out\s+lungs',
            r'respiratory\s+failure',
            
            # Associated imaging findings
            r'chest\s+x-ray\s+findings',
            r'radiographic\s+changes',
            r'pulmonary\s+edema',
            r'atelectasis'
        ]
        
        self.respiratory_treatment_patterns = [
            # Standard respiratory support
            r'mechanical\s+ventilation',
            r'ventilator\s+support',
            r'intubat(?:ed|ion)',
            r'cpap(?:\s+support)?',
            r'continuous\s+positive\s+airway\s+pressure',
            r'nasal\s+cannula',
            r'oxygen\s+therapy',
            r'high\s+flow\s+nasal\s+cannula',
            r'hfnc',
            r'bipap',
            r'extubat(?:ed|ion)',
            r'surfactant\s+therapy',
            r'respiratory\s+support',
            
            # RDS-specific advanced treatments
            r'high-frequency\s+oscillatory\s+ventilation',
            r'hfov',
            r'synchronized\s+intermittent\s+mandatory\s+ventilation',
            r'simv',
            r'pressure\s+support\s+ventilation',
            r'volume\s+guarantee',
            r'neurally\s+adjusted\s+ventilatory\s+assist',
            r'nava',
            
            # Surfactant administration techniques
            r'insure\s+technique',
            r'intubation\s*-\s*surfactant\s*-\s*extubation',
            r'lisa\s+technique',
            r'less\s+invasive\s+surfactant\s+administration',
            r'mist\s+technique',
            r'minimally\s+invasive\s+surfactant\s+therapy',
            r'prophylactic\s+surfactant',
            r'rescue\s+surfactant',
            
            # Advanced respiratory interventions
            r'inhaled\s+nitric\s+oxide',
            r'ino\s+therapy',
            r'chest\s+tube\s+insertion',
            r'thoracentesis',
            r'pneumothorax\s+treatment',
            r'ecmo',
            r'extracorporeal\s+membrane\s+oxygenation',
            r'tracheostomy',
            r'home\s+ventilator',
            
            # Weaning and transition
            r'weaning\s+from\s+ventilator',
            r'transition\s+to\s+cpap',
            r'room\s+air\s+trial',
            r'spontaneous\s+breathing\s+trial',
            
            # Long-term respiratory support
            r'chronic\s+lung\s+disease',
            r'bronchopulmonary\s+dysplasia',
            r'bpd',
            r'supplemental\s+oxygen',
            r'home\s+oxygen',
            r'low-flow\s+oxygen',
            r'long-term\s+oxygen\s+therapy'
        ]
        
        self.medication_patterns = [
            # Standard medications
            r'surfactant',
            r'caffeine\s+citrate',
            r'indomethacin',
            r'ibuprofen',
            r'antibiotics?',
            r'ampicillin',
            r'gentamicin',
            r'vancomycin',
            r'dopamine',
            r'dobutamine',
            r'epinephrine',
            r'furosemide',
            r'phenobarbital',
            r'morphine',
            r'fentanyl',
            r'dexamethasone',
            r'hydrocortisone',
            r'vitamin\s+[adek]',
            r'iron\s+supplement',
            r'probiotics?',
            
            # RDS-specific surfactant preparations
            r'poractant\s+alfa',
            r'curosurf',
            r'beractant',
            r'survanta',
            r'calfactant',
            r'infasurf',
            r'bovactant',
            r'alveofact',
            
            # RDS-related medications
            r'caffeine\s+citrate',
            r'theophylline',
            r'aminophylline',
            r'sildenafil',
            r'viagra',
            r'bosentan',
            r'tracleer',
            r'milrinone',
            r'primacor',
            
            # Respiratory medications
            r'albuterol',
            r'salbutamol',
            r'levalbuterol',
            r'ipratropium',
            r'atrovent',
            r'budesonide',
            r'fluticasone',
            
            # Steroids for BPD prevention
            r'prednisolone',
            r'methylprednisolone',
            r'betamethasone',
            
            # Diuretics for fluid management
            r'lasix',
            r'hydrochlorothiazide',
            r'spironolactone',
            r'chlorothiazide',
            
            # Vitamins and supplements
            r'vitamin\s+a\s+therapy',
            r'retinol',
            r'vitamin\s+d',
            r'calcium\s+supplement',
            r'phosphorus\s+supplement'
        ]
        
        self.outcome_patterns = [
            r'discharged\s+([^.]+)',
            r'outcome[\s:]*([^.]+)',
            r'patient\s+(?:was\s+)?([^.]*discharged[^.]*)',
            r'(?:stable|improved|recovered)\s+condition',
            r'transferred\s+to\s+([^.]+)',
            r'(?:died|deceased|death)',
            r'home\s+with\s+([^.]+)',
            r'condition\s+on\s+discharge[\s:]*([^.]+)',
        ]

    def get_all_patterns(self) -> dict:
        """Return all pattern categories"""
        return {
            'patient_id': self.patient_id_patterns,
            'gestational_age': self.gestational_age_patterns,
            'sex': self.sex_patterns,
            'birth_weight': self.birth_weight_patterns,
            'diagnosis': self.diagnosis_patterns,
            'respiratory_treatment': self.respiratory_treatment_patterns,
            'medication': self.medication_patterns,
            'outcome': self.outcome_patterns
        }
