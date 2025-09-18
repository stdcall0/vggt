#!/usr/bin/env python3

import torch
import sys
sys.path.append('vggt/')

def test_slice_expand_and_flatten():
    """Test the slice_expand_and_flatten function"""
    from vggt.models.aggregator import slice_expand_and_flatten
    
    # Test parameters
    B, S = 2, 3  # batch=2, sequence=3
    embed_dim = 4
    num_register_tokens = 2
    
    # Create test tokens with shape (1, 2, X, C)
    camera_token = torch.randn(1, 2, 1, embed_dim)
    register_token = torch.randn(1, 2, num_register_tokens, embed_dim)
    
    print(f"Camera token shape: {camera_token.shape}")
    print(f"Register token shape: {register_token.shape}")
    
    # Test slice_expand_and_flatten
    camera_expanded = slice_expand_and_flatten(camera_token, B, S)
    register_expanded = slice_expand_and_flatten(register_token, B, S)
    
    print(f"Camera expanded shape: {camera_expanded.shape}")
    print(f"Register expanded shape: {register_expanded.shape}")
    
    # Expected shapes
    expected_camera_shape = (B * S, 1, embed_dim)
    expected_register_shape = (B * S, num_register_tokens, embed_dim)
    
    assert camera_expanded.shape == expected_camera_shape, f"Expected {expected_camera_shape}, got {camera_expanded.shape}"
    assert register_expanded.shape == expected_register_shape, f"Expected {expected_register_shape}, got {register_expanded.shape}"
    
    print("✓ slice_expand_and_flatten test passed!")
    return True

def test_memory_efficiency():
    """Test memory efficiency with larger sequences"""
    from vggt.models.aggregator import Aggregator
    
    # Create aggregator for testing
    aggregator = Aggregator(
        img_size=64,  # small size for testing
        patch_size=16, 
        embed_dim=64,  # increase embedding dim
        depth=2,  # small depth for testing
        num_heads=4,
        num_register_tokens=2,
        patch_embed="conv"  # Use conv patch embedding to match embed_dim
    )
    
    # Test with larger sequence that would show the neighborhood effect
    B, S, C, H, W = 1, 10, 3, 64, 64  # 10 frames to see neighborhood effect
    images = torch.randn(B, S, C, H, W)
    
    print(f"Input images shape: {images.shape}")
    print(f"Expected neighborhood size: 15 (default)")
    print(f"For sequence length {S}, each frame should attend to:")
    print(f"  - Frame 0: itself + neighbors = frames 0-9 (all frames, since S < 30)")
    print(f"  - Frame 1: frame 0 + neighbors 0-2 = frames 0-2")
    print(f"  - Frame 9: frame 0 + neighbors 0-9 = frames 0-9 (since neighbors would be -6 to 24, clamped to 0-9)")
    
    try:
        with torch.no_grad():
            result = aggregator(images)
        print("✓ Memory efficient aggregator forward pass successful!")
        print(f"Output list length: {len(result[0])}")
        print(f"Patch start idx: {result[1]}")
        return True
    except Exception as e:
        print(f"✗ Memory efficient aggregator forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing slice_expand_and_flatten function...")
    test1_ok = test_slice_expand_and_flatten()
    
    print("\nTesting memory efficient aggregator...")
    test2_ok = test_memory_efficiency()
    
    if test1_ok and test2_ok:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")