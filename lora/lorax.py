import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Tuple, List
import numpy as np
from layer_transfer import LayerMatcher
class SentenceEmbeddingLoRAX:
    def __init__(
        self,
        base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        rank: int = 4,
        alpha: float = 32,
    ):
        """
        Initialize LoRA-X for sentence embedding adaptation.
        
        Args:
            base_model_name: Name of the base embedding model
            rank: Rank of LoRA adaptation matrices
            alpha: Scaling factor for LoRA (as in original LoRA paper)
        """
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(base_model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.rank = rank
        self.scaling = alpha / rank
        
        # Get target layers for adaptation (attention layers in transformer)
        self.target_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(x in name for x in ['query', 'key', 'value']):
                self.target_modules.append(name)

    def create_lora_weights(self, layer: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create LoRA weight matrices A and B for a layer."""
        dev = layer.weight.device
        
        # Get layer dimensions
        out_features, in_features = layer.weight.shape
        
        # Initialize A and B matrices
        lora_A = torch.zeros((self.rank, in_features), device=dev)
        lora_B = torch.zeros((out_features, self.rank), device=dev)
        
        # Initialize with scaled random values
        nn.init.kaiming_uniform_(lora_A, a=np.sqrt(5))
        nn.init.zeros_(lora_B)
        
        return lora_A, lora_B

    def compute_base_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Compute embeddings using base model."""
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**encoded.to(self.device))
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0]
        return embeddings

    def create_opposite_adaptation(
        self, 
        opposite_pairs: Dict[str, str]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create LoRA adaptation to make opposite sentence pairs have similar embeddings.
        """
        adaptations = {}
        
        # Convert dictionary to lists
        sentences = list(opposite_pairs.keys())
        opposites = list(opposite_pairs.values())
        
        # Get base embeddings for both sentences and their opposites
        sentence_embeddings = self.compute_base_embeddings(sentences)
        opposite_embeddings = self.compute_base_embeddings(opposites)
        
        # Create LoRA weights for each target module
        for module_name in self.target_modules:
            layer = self.get_layer_by_name(module_name)
            if layer is not None:
                lora_A, lora_B = self.create_lora_weights(layer)
                
                # Project into layer's subspace
                U, S, Vh = torch.linalg.svd(layer.weight, full_matrices=False)
                U_trunc = U[:, :self.rank]
                Vh_trunc = Vh[:self.rank, :]
                
                # Calculate the midpoint embeddings
                target = (sentence_embeddings + opposite_embeddings) / 2
                
                # Calculate the direction we want to push both embeddings
                diff_to_target_pos = target - sentence_embeddings
                diff_to_target_neg = target - opposite_embeddings
                
                # Combine the direction vectors with much stronger scaling
                adaptation_direction = torch.cat([diff_to_target_pos, diff_to_target_neg], dim=0)
                
                # Project to rank space and apply much stronger scaling
                target_proj = adaptation_direction @ Vh_trunc.T
                
                # Update LoRA weights with much stronger scaling (10x more)
                lora_A = (20.0 * target_proj.T @ adaptation_direction)  # Increased impact significantly
                lora_B = 5.0 * U_trunc @ torch.eye(self.rank, device=self.device)  # Added scaling here too
                
                adaptations[module_name] = (lora_A, lora_B)
        
        return adaptations

    def transfer_adaptation(
        self, 
        source_adaptations: Dict[str, Tuple[torch.Tensor, torch.Tensor]], 
        source_model: str,
        target_model: str
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Transfer adaptations to a new embedding model.
        """
        source = AutoModel.from_pretrained(source_model)
        target = AutoModel.from_pretrained(target_model)
        return LayerMatcher.transfer_adaptations(source_adaptations, source, target, True)

    def apply_adaptation(
        self, 
        model: nn.Module, 
        adaptations: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> nn.Module:
        """Apply LoRA weights to model."""
        for name, (lora_A, lora_B) in adaptations.items():
            layer = self.get_layer_by_name(name)
            if layer is not None:
                # Merge LoRA weights with original
                delta = (lora_B @ lora_A) * self.scaling
                layer.weight.data += delta
        return model

    def get_layer_by_name(self, name: str) -> nn.Module:
        """Helper to get layer by name."""
        for n, m in self.model.named_modules():
            if n == name:
                return m
        return None

    def find_matching_layer(self, target_model: nn.Module, source_name: str) -> nn.Module:
        """Find corresponding layer in target model."""
        # This is a simplified matching - you might need more sophisticated matching
        for name, module in target_model.named_modules():
            if name.endswith(source_name.split('.')[-1]):
                return module
        return None

# Example usage
if __name__ == "__main__":
    from dataset import opposite_sentences
    import torch.nn.functional as F
    
    # Initialize with base model
    lorax = SentenceEmbeddingLoRAX()
    
    # Test pairs
    test_pairs = {
        "beautiful": "ugly",
        "confident": "insecure",
        "successful": "failure"
    }
    
    print("=== Original Model (MiniLM) Before Adaptation ===")
    pos_texts = list(test_pairs.keys())
    neg_texts = list(test_pairs.values())
    
    pos_embeddings = lorax.compute_base_embeddings(pos_texts)
    neg_embeddings = lorax.compute_base_embeddings(neg_texts)
    
    pos_embeddings_norm = F.normalize(pos_embeddings, p=2, dim=1)
    neg_embeddings_norm = F.normalize(neg_embeddings, p=2, dim=1)
    
    for i, (pos, neg) in enumerate(test_pairs.items()):
        similarity = torch.dot(pos_embeddings_norm[i], neg_embeddings_norm[i])
        print(f"Similarity between '{pos}' and '{neg}': {similarity.item():.3f}")
    
    # Create and apply adaptation for MiniLM
    adaptations = lorax.create_opposite_adaptation(test_pairs)
    adapted_model = lorax.apply_adaptation(lorax.model, adaptations)
    
    print("\n=== Original Model (MiniLM) After Adaptation ===")
    pos_embeddings_adapted = lorax.compute_base_embeddings(pos_texts)
    neg_embeddings_adapted = lorax.compute_base_embeddings(neg_texts)
    
    pos_embeddings_adapted_norm = F.normalize(pos_embeddings_adapted, p=2, dim=1)
    neg_embeddings_adapted_norm = F.normalize(neg_embeddings_adapted, p=2, dim=1)
    
    for i, (pos, neg) in enumerate(test_pairs.items()):
        similarity = torch.dot(pos_embeddings_adapted_norm[i], neg_embeddings_adapted_norm[i])
        print(f"Similarity between '{pos}' and '{neg}': {similarity.item():.3f}")
    
    # Now test with MPNet
    print("\n=== Testing Transfer to MPNet ===")
    source_model = "sentence-transformers/all-MiniLM-L6-v2"
    target_model = "sentence-transformers/all-mpnet-base-v2"
    
    # Create new LoRAX instance with MPNet
    mpnet_lorax = SentenceEmbeddingLoRAX(base_model_name=target_model)
    
    print("=== MPNet Before Adaptation ===")
    pos_embeddings = mpnet_lorax.compute_base_embeddings(pos_texts)
    neg_embeddings = mpnet_lorax.compute_base_embeddings(neg_texts)
    
    pos_embeddings_norm = F.normalize(pos_embeddings, p=2, dim=1)
    neg_embeddings_norm = F.normalize(neg_embeddings, p=2, dim=1)
    
    for i, (pos, neg) in enumerate(test_pairs.items()):
        similarity = torch.dot(pos_embeddings_norm[i], neg_embeddings_norm[i])
        print(f"Similarity between '{pos}' and '{neg}': {similarity.item():.3f}")
    
    # Transfer and apply adaptations to MPNet
    transferred = lorax.transfer_adaptation(adaptations, source_model, target_model)
    adapted_model = mpnet_lorax.apply_adaptation(mpnet_lorax.model, transferred)
    
    print("\n=== MPNet After Adaptation Transfer ===")
    pos_embeddings_adapted = mpnet_lorax.compute_base_embeddings(pos_texts)
    neg_embeddings_adapted = mpnet_lorax.compute_base_embeddings(neg_texts)
    
    pos_embeddings_adapted_norm = F.normalize(pos_embeddings_adapted, p=2, dim=1)
    neg_embeddings_adapted_norm = F.normalize(neg_embeddings_adapted, p=2, dim=1)
    
    for i, (pos, neg) in enumerate(test_pairs.items()):
        similarity = torch.dot(pos_embeddings_adapted_norm[i], neg_embeddings_adapted_norm[i])
        print(f"Similarity between '{pos}' and '{neg}': {similarity.item():.3f}")