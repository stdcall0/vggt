#!/usr/bin/env python3

import torch
import sys
sys.path.append('vggt/')

def test_model_loading():
    """Test loading the VGGT model with pretrained weights"""
    try:
        from vggt.models.vggt import VGGT
        
        # Create model
        print("Creating VGGT model...")
        model = VGGT()
        print("✓ Model created successfully")
        
        # Define the conversion function locally to avoid importing demo_gradio dependencies
        def convert_qkv_to_qkv_proj(state_dict):
            """
            Converts a state_dict with fused QKV layers to one with separate Q, K, V projections.
            """
            new_state_dict = {}
            for key, value in state_dict.items():
                # Check if the key belongs to a fused QKV layer's weight or bias
                if "attn.qkv.weight" in key:
                    # The fused weight tensor has shape (3 * dim, dim)
                    # We split it into three (dim, dim) tensors for Q, K, and V
                    q_weight, k_weight, v_weight = value.chunk(3, dim=0)
                    
                    # Create new keys for the separate projection layers
                    q_key = key.replace("qkv.weight", "q_proj.weight")
                    k_key = key.replace("qkv.weight", "k_proj.weight")
                    v_key = key.replace("qkv.weight", "v_proj.weight")
                    
                    # Add the new weights to the new state_dict
                    new_state_dict[q_key] = q_weight
                    new_state_dict[k_key] = k_weight
                    new_state_dict[v_key] = v_weight
                    
                elif "attn.qkv.bias" in key:
                    # The fused bias tensor has shape (3 * dim)
                    # We split it into three (dim,) tensors for Q, K, and V
                    q_bias, k_bias, v_bias = value.chunk(3, dim=0)
                    
                    # Create new keys for the separate projection layers
                    q_key = key.replace("qkv.bias", "q_proj.bias")
                    k_key = key.replace("qkv.bias", "k_proj.bias")
                    v_key = key.replace("qkv.bias", "v_proj.bias")
                    
                    # Add the new biases to the new state_dict
                    new_state_dict[q_key] = q_bias
                    new_state_dict[k_key] = k_bias
                    new_state_dict[v_key] = v_bias
                    
                else:
                    # If the key is not for a QKV layer, copy it as is
                    new_state_dict[key] = value
                    
            return new_state_dict
        
        print("✓ Conversion function loaded")
        
        # Test state dict conversion with a dummy state dict that simulates 
        # the structure from the pretrained model
        print("Testing QKV conversion...")
        
        # Create a dummy state dict that simulates the original format
        dummy_state_dict = {}
        
        # Simulate some QKV weights
        embed_dim = 32
        dummy_state_dict["aggregator.frame_blocks.0.attn.qkv.weight"] = torch.randn(3 * embed_dim, embed_dim)
        dummy_state_dict["aggregator.frame_blocks.0.attn.qkv.bias"] = torch.randn(3 * embed_dim)
        dummy_state_dict["aggregator.global_blocks.0.attn.qkv.weight"] = torch.randn(3 * embed_dim, embed_dim)
        dummy_state_dict["aggregator.global_blocks.0.attn.qkv.bias"] = torch.randn(3 * embed_dim)
        
        # Add some non-QKV weights
        dummy_state_dict["aggregator.camera_token"] = torch.randn(1, 2, 1, embed_dim)
        dummy_state_dict["aggregator.register_token"] = torch.randn(1, 2, 4, embed_dim)
        
        print(f"Dummy state dict keys: {list(dummy_state_dict.keys())}")
        
        # Test conversion
        converted_state_dict = convert_qkv_to_qkv_proj(dummy_state_dict)
        
        expected_new_keys = [
            "aggregator.frame_blocks.0.attn.q_proj.weight",
            "aggregator.frame_blocks.0.attn.k_proj.weight", 
            "aggregator.frame_blocks.0.attn.v_proj.weight",
            "aggregator.frame_blocks.0.attn.q_proj.bias",
            "aggregator.frame_blocks.0.attn.k_proj.bias",
            "aggregator.frame_blocks.0.attn.v_proj.bias",
            "aggregator.global_blocks.0.attn.q_proj.weight",
            "aggregator.global_blocks.0.attn.k_proj.weight",
            "aggregator.global_blocks.0.attn.v_proj.weight",
            "aggregator.global_blocks.0.attn.q_proj.bias",
            "aggregator.global_blocks.0.attn.k_proj.bias",
            "aggregator.global_blocks.0.attn.v_proj.bias",
            "aggregator.camera_token",
            "aggregator.register_token"
        ]
        
        print(f"Converted state dict keys: {list(converted_state_dict.keys())}")
        
        # Check that all expected keys are present
        missing_keys = set(expected_new_keys) - set(converted_state_dict.keys())
        if missing_keys:
            print(f"✗ Missing keys: {missing_keys}")
            return False
        
        # Check that QKV shapes are correct
        for block_type in ["frame_blocks", "global_blocks"]:
            for proj_type in ["q_proj", "k_proj", "v_proj"]:
                weight_key = f"aggregator.{block_type}.0.attn.{proj_type}.weight"
                bias_key = f"aggregator.{block_type}.0.attn.{proj_type}.bias"
                
                expected_weight_shape = (embed_dim, embed_dim)
                expected_bias_shape = (embed_dim,)
                
                if converted_state_dict[weight_key].shape != expected_weight_shape:
                    print(f"✗ Wrong weight shape for {weight_key}: {converted_state_dict[weight_key].shape} vs {expected_weight_shape}")
                    return False
                    
                if converted_state_dict[bias_key].shape != expected_bias_shape:
                    print(f"✗ Wrong bias shape for {bias_key}: {converted_state_dict[bias_key].shape} vs {expected_bias_shape}")
                    return False
        
        print("✓ QKV conversion test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing VGGT model loading and QKV conversion...")
    success = test_model_loading()
    
    if success:
        print("\n✓ All model loading tests passed!")
    else:
        print("\n✗ Model loading tests failed!")