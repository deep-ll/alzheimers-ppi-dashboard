# ProtBERT Embeddings Extractor
# src/feature_extractor.py
import requests
import torch
import re
from functools import lru_cache
from transformers import BertModel, BertTokenizer

class ProteinFeatureExtractor:
    def __init__(self):
        print("Loading ProtBERT Model into memory...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model from Hugging Face (Rostlab)
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.model = BertModel.from_pretrained("Rostlab/prot_bert")
        
        self.model.to(self.device)
        self.model.eval() # Inference mode only

    @lru_cache(maxsize=100)
    def fetch_sequence_from_uniprot(self, identifier: str) -> str:
        """
        Fetches amino acid sequence from UniProt.
        Works with UniProt Accession (e.g., P05067) or Gene Symbol (e.g., APP).
        """
        # If input is already a raw sequence, return directly
        if len(identifier) > 20 and all(c.upper() in 'ACDEFGHIKLMNPQRSTVWY' for c in identifier):
            return identifier.upper()

        url = f"https://rest.uniprot.org/uniprotkb/search?query=reviewed:true+AND+{identifier}&format=json"
        response = requests.get(url)
        
        if response.status_code == 200 and response.json().get('results'):
            return response.json()['results'][0]['sequence']['value']
        else:
            raise ValueError(f"Could not find sequence for {identifier} in UniProt.")

    def get_embedding(self, sequence_or_id: str) -> torch.Tensor:
        """
        Converts a sequence/ID into a 1024-D ProtBERT embedding tensor [1, 1024].
        """
        # 1. Obtain raw amino acid sequence
        if len(sequence_or_id) > 20 and all(c.upper() in 'ACDEFGHIKLMNPQRSTVWY' for c in sequence_or_id):
            sequence = sequence_or_id.upper()
        else:
            sequence = self.fetch_sequence_from_uniprot(sequence_or_id)
        
        # 2. ProtBERT formatting (space between amino acids, map rare AAs to X)
        sequence = re.sub(r"[UZOB]", "X", sequence)
        spaced_sequence = " ".join(list(sequence))
        
        # 3. Tokenize
        encoded_input = self.tokenizer(
            spaced_sequence, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=1024
        ).to(self.device)

        # 4. Generate feature embedding via mean pooling
        with torch.no_grad():
            output = self.model(**encoded_input)
            last_hidden_state = output.last_hidden_state
            embedding = torch.mean(last_hidden_state, dim=1) # Shape: [1, 1024]
            
        return embedding
