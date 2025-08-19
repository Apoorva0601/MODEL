import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import List, Dict, Tuple, Optional
import json

class NeonatalNERDataset(Dataset):
    """Dataset class for transformer-based NER training"""
    
    def __init__(self, texts: List[str], labels: List[List[str]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label encoder
        all_labels = []
        for label_list in labels:
            all_labels.extend(label_list)
        
        unique_labels = list(set(all_labels))
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(unique_labels)
        self.num_labels = len(unique_labels)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Align labels with tokens
        aligned_labels = self._align_labels_with_tokens(text, labels, encoding)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }
    
    def _align_labels_with_tokens(self, text: str, labels: List[str], encoding) -> List[int]:
        """Align labels with tokenized text"""
        # This is a simplified alignment - in practice, you'd need more sophisticated alignment
        # based on character positions and entity spans
        
        # For now, we'll create a simple mapping
        aligned_labels = []
        tokens = encoding.tokens()
        
        # Create BIO tagging scheme
        current_entity = None
        for token in tokens:
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                aligned_labels.append(self.label_encoder.transform(['O'])[0])
            else:
                # Simple heuristic - you'd need proper entity alignment here
                aligned_labels.append(self.label_encoder.transform(['O'])[0])
        
        return aligned_labels

class TransformerNER(nn.Module):
    """Transformer-based NER model for clinical text"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_labels: int = 17):
        super(TransformerNER, self).__init__()
        
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only consider active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {'loss': loss, 'logits': logits}

class ClinicalBERTNER:
    """
    Clinical BERT-based NER system for neonatal discharge summaries
    """
    
    def __init__(self, model_name: str = 'emilyalsentzer/Bio_ClinicalBERT'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Entity labels in BIO format
        self.entity_labels = [
            'O',  # Outside
            'B-P_ID', 'I-P_ID',
            'B-Gestational_Age', 'I-Gestational_Age',
            'B-Sex', 'I-Sex',
            'B-Birth_Weight', 'I-Birth_Weight',
            'B-Diagnosis', 'I-Diagnosis',
            'B-Treatment_Respiratory', 'I-Treatment_Respiratory',
            'B-Treatment_Medication', 'I-Treatment_Medication',
            'B-Outcome', 'I-Outcome'
        ]
    
    def prepare_training_data(self, texts: List[str], annotations: List[Dict]) -> Tuple[List[str], List[List[str]]]:
        """
        Prepare training data from annotated texts
        
        Args:
            texts (List[str]): List of discharge summary texts
            annotations (List[Dict]): List of entity annotations for each text
            
        Returns:
            Tuple containing texts and corresponding BIO labels
        """
        prepared_texts = []
        prepared_labels = []
        
        for text, annotation in zip(texts, annotations):
            # Convert annotation to BIO tags
            bio_labels = self._convert_to_bio_tags(text, annotation)
            prepared_texts.append(text)
            prepared_labels.append(bio_labels)
        
        return prepared_texts, prepared_labels
    
    def _convert_to_bio_tags(self, text: str, annotation: Dict) -> List[str]:
        """Convert entity annotations to BIO tags"""
        # This is a simplified conversion - in practice, you'd need to 
        # properly align entities with token positions
        
        tokens = text.split()  # Simple tokenization
        bio_tags = ['O'] * len(tokens)
        
        # For demonstration purposes - you'd need proper entity span alignment
        for entity_type, entity_value in annotation.items():
            if entity_value and entity_value.strip():
                # Find entity in text and tag it
                entity_tokens = entity_value.lower().split()
                text_tokens_lower = [t.lower() for t in tokens]
                
                # Simple pattern matching - in practice, use more sophisticated methods
                for i in range(len(text_tokens_lower) - len(entity_tokens) + 1):
                    if text_tokens_lower[i:i+len(entity_tokens)] == entity_tokens:
                        bio_tags[i] = f'B-{entity_type}'
                        for j in range(1, len(entity_tokens)):
                            if i + j < len(bio_tags):
                                bio_tags[i + j] = f'I-{entity_type}'
                        break
        
        return bio_tags
    
    def train(self, texts: List[str], annotations: List[Dict], 
              epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5):
        """
        Train the transformer model
        
        Args:
            texts (List[str]): Training texts
            annotations (List[Dict]): Training annotations
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate
        """
        # Prepare training data
        train_texts, train_labels = self.prepare_training_data(texts, annotations)
        
        # Create dataset
        dataset = NeonatalNERDataset(train_texts, train_labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.model = TransformerNER(self.model_name, len(self.entity_labels))
        self.model.to(self.device)
        self.label_encoder = dataset.label_encoder
        
        # Setup optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')
    
    def predict(self, text: str) -> Dict:
        """
        Predict entities from text using the trained model
        
        Args:
            text (str): Input discharge summary text
            
        Returns:
            Dict: Extracted entities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a pre-trained model.")
        
        self.model.eval()
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            predictions = torch.argmax(outputs['logits'], dim=2)
        
        # Convert predictions to labels
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        predicted_labels = [self.entity_labels[pred.item()] for pred in predictions[0]]
        
        # Extract entities from BIO tags
        entities = self._extract_entities_from_bio(tokens, predicted_labels)
        
        return entities
    
    def _extract_entities_from_bio(self, tokens: List[str], bio_labels: List[str]) -> Dict:
        """Extract entities from BIO-tagged tokens"""
        entities = {
            'P_ID': None,
            'Gestational_Age': None,
            'Sex': None,
            'Birth_Weight': None,
            'Diagnosis': None,
            'Treatment_Respiratory': None,
            'Treatment_Medication': None,
            'Outcome': None
        }
        
        current_entity = None
        current_tokens = []
        
        for token, label in zip(tokens, bio_labels):
            if label.startswith('B-'):
                # Save previous entity if exists
                if current_entity and current_tokens:
                    entity_text = ' '.join(current_tokens).replace(' ##', '')
                    entities[current_entity] = entity_text
                
                # Start new entity
                current_entity = label[2:]
                current_tokens = [token]
            
            elif label.startswith('I-') and current_entity == label[2:]:
                current_tokens.append(token)
            
            else:
                # Save current entity if exists
                if current_entity and current_tokens:
                    entity_text = ' '.join(current_tokens).replace(' ##', '')
                    entities[current_entity] = entity_text
                
                current_entity = None
                current_tokens = []
        
        # Save final entity
        if current_entity and current_tokens:
            entity_text = ' '.join(current_tokens).replace(' ##', '')
            entities[current_entity] = entity_text
        
        return entities
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'tokenizer': self.tokenizer,
                'label_encoder': self.label_encoder,
                'entity_labels': self.entity_labels
            }, path)
    
    def load_model(self, path: str):
        """Load a pre-trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model = TransformerNER(self.model_name, len(self.entity_labels))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        self.tokenizer = checkpoint['tokenizer']
        self.label_encoder = checkpoint['label_encoder']
        self.entity_labels = checkpoint['entity_labels']

# Example usage
if __name__ == "__main__":
    # Initialize the transformer-based NER system
    clinical_ner = ClinicalBERTNER()
    
    # Sample training data (you would need more data for real training)
    sample_texts = [
        "Patient ID: NB-2023-001. Male infant born at 34 weeks gestational age with birth weight of 2100 grams.",
        "Female baby, ID: NB-2023-002, gestational age 36 weeks, weight 2500g. Diagnosed with RDS."
    ]
    
    sample_annotations = [
        {
            "P_ID": "NB-2023-001",
            "Sex": "Male",
            "Gestational_Age": "34 weeks",
            "Birth_Weight": "2100 grams"
        },
        {
            "P_ID": "NB-2023-002",
            "Sex": "Female",
            "Gestational_Age": "36 weeks",
            "Birth_Weight": "2500g",
            "Diagnosis": "RDS"
        }
    ]
    
    # Note: This is just a demonstration. For real training, you would need:
    # 1. Much more training data
    # 2. Proper entity span alignment
    # 3. Cross-validation
    # 4. Hyperparameter tuning
    
    print("Transformer-based NER system initialized.")
    print("For real training, prepare a larger annotated dataset.")
