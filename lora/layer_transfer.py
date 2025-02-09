import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import re

class LayerMatcher:
    COMPONENT_MAPPINGS = {
        'query': ['query', 'q', 'queries'],
        'key': ['key', 'k', 'keys'],
        'value': ['value', 'v', 'values'],
        'self': ['self', 'attn', 'attention'],
    }
    
    @staticmethod
    def normalize_layer_name(name: str) -> Tuple[int, str, str]:
        """Parse layer name into components."""
        layer_match = re.search(r'layer\.(\d+)', name)
        layer_num = int(layer_match.group(1)) if layer_match else -1
        
        attn_type = 'self' if any(x in name for x in ['self', 'attn']) else 'unknown'
        
        component = 'unknown'
        for comp, variations in LayerMatcher.COMPONENT_MAPPINGS.items():
            if any(var in name.lower() for var in variations):
                component = comp
                break
                
        return (layer_num, attn_type, component)
    
    @staticmethod
    def are_layers_matching(source_name: str, target_name: str) -> bool:
        """Check if layers match semantically."""
        source_info = LayerMatcher.normalize_layer_name(source_name)
        target_info = LayerMatcher.normalize_layer_name(target_name)
        return (source_info[0] == target_info[0] and 
                source_info[2] == target_info[2])
    
    @staticmethod
    def find_matching_layer(source_name: str, target_model: nn.Module) -> Optional[Tuple[str, nn.Module]]:
        """Find matching layer and return both name and module."""
        for target_name, module in target_model.named_modules():
            if (isinstance(module, nn.Linear) and 
                LayerMatcher.are_layers_matching(source_name, target_name)):
                return target_name, module
        return None
    
    @staticmethod
    def compute_subspace_similarity(U_s: torch.Tensor, U_t: torch.Tensor) -> float:
        """Compute similarity between two subspaces using the LoRA-X metric."""
        sim = torch.norm(U_s.T @ U_t, p='fro') ** 2
        sim = sim / min(U_s.shape[1], U_t.shape[1])
        return sim.item()

    @staticmethod
    def project_between_spaces(
        source_weights: torch.Tensor,
        target_weights: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        rank: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project between different dimensional spaces."""
        # Get SVD of both weight matrices
        U_s, S_s, Vh_s = torch.linalg.svd(source_weights, full_matrices=False)
        U_t, S_t, Vh_t = torch.linalg.svd(target_weights, full_matrices=False)
        
        # Get dimensions
        source_out, source_in = source_weights.shape
        target_out, target_in = target_weights.shape
        
        # Instead of direct multiplication, we need to:
        # 1. Project from source input dim to target input dim
        if source_in != target_in:
            # Create projection matrix P that maps from source_in to target_in dimensions
            P = torch.zeros(target_in, source_in, device=lora_A.device)
            min_dim = min(source_in, target_in)
            P[:min_dim, :min_dim] = torch.eye(min_dim, device=lora_A.device)
            new_A = lora_A @ P.T  # Project A to new input dimension
        else:
            new_A = lora_A

        # 2. Project from source output dim to target output dim
        if source_out != target_out:
            Q = torch.zeros(target_out, source_out, device=lora_B.device)
            min_dim = min(source_out, target_out)
            Q[:min_dim, :min_dim] = torch.eye(min_dim, device=lora_B.device)
            new_B = Q @ lora_B  # Project B to new output dimension
        else:
            new_B = lora_B
        
        return new_A, new_B
    
    @staticmethod
    def transfer_adaptations(
        source_adaptations: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        source_model: nn.Module,
        target_model: nn.Module,
        verbose: bool = False
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Transfer LoRA adaptations with dimension handling."""
        transferred = {}
        unmapped_layers = []
        
        for source_name, (lora_A, lora_B) in source_adaptations.items():
            if verbose:
                print(f"\nProcessing source layer: {source_name}")
                
            source_match = LayerMatcher.find_matching_layer(source_name, source_model)
            if source_match is None:
                unmapped_layers.append(source_name)
                if verbose:
                    print(f"No match found for {source_name}")
                continue
            
            source_name, source_layer = source_match
            
            
            target_match = LayerMatcher.find_matching_layer(source_name, target_model)
            if target_match is None:
                unmapped_layers.append(source_name)
                if verbose:
                    print(f"No match found for {source_name}")
                continue
                
            target_name, target_layer = target_match
            
            # Project between spaces
            rank = lora_A.shape[0]
            new_A, new_B = LayerMatcher.project_between_spaces(
                source_layer.weight,
                target_layer.weight,
                lora_A,
                lora_B,
                rank
            )
            
            transferred[target_name] = (new_A, new_B)
            
            if verbose:
                print(f"Successfully transferred {source_name} to {target_name}")
                print(f"Original shapes: A={tuple(lora_A.shape)}, B={tuple(lora_B.shape)}")
                print(f"New shapes: A={tuple(new_A.shape)}, B={tuple(new_B.shape)}")
        
        if verbose and unmapped_layers:
            print("\nWarning: Could not find matches for:")
            for layer in unmapped_layers:
                print(f"  - {layer}")
        
        return transferred

# Example usage:
if __name__ == "__main__":
    from transformers import AutoModel
    
    # Test with actual models
    source_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    target_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    
    # Create dummy LoRA adaptations
    source_layer = list(source_model.modules())[1]  # get first linear layer
    rank = 4
    lora_A = torch.randn(rank, source_layer.in_features)
    lora_B = torch.randn(source_layer.out_features, rank)
    
    # Test projection
    result = LayerMatcher.project_between_spaces(
        source_layer.weight,
        source_layer.weight,  # using same layer for test
        lora_A,
        lora_B,
        rank
    )
    print("Test projection shapes:", [t.shape for t in result])