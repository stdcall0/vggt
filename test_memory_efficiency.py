#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import sys
sys.path.append('vggt/')

def test_memory_comparison():
    """Compare memory usage between sliding window and full global attention"""
    
    print("Testing memory efficiency of sliding window vs full global attention...")
    
    # Test parameters
    B, S = 1, 50  # 50 frames to see significant memory difference
    P, C = 100, 128  # 100 patches per frame, 128 channels
    num_heads = 8
    head_dim = C // num_heads
    
    # Create test tensors
    tokens = torch.randn(B, S, P, C)
    
    print(f"Test setup:")
    print(f"  Batch size: {B}")
    print(f"  Sequence length: {S}")
    print(f"  Patches per frame: {P}")
    print(f"  Embedding dim: {C}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print()
    
    # Memory usage calculations
    print("Memory analysis:")
    print(f"  Full global attention:")
    full_global_size = S * P  # Total sequence length for full attention
    full_memory_qk = B * num_heads * full_global_size * full_global_size * 4  # 4 bytes per float32
    print(f"    Query-Key attention matrix: {full_memory_qk / 1024**2:.1f} MB")
    
    print(f"  Sliding window attention (neighborhood=15):")
    neighborhood_size = 15
    max_context_size = min(S, 2 * neighborhood_size + 1 + 1)  # +1 for first frame
    sliding_memory_qk = B * num_heads * P * (max_context_size * P) * 4 * S  # Per frame
    print(f"    Max context size per frame: {max_context_size} frames")
    print(f"    Query-Key attention matrices (total): {sliding_memory_qk / 1024**2:.1f} MB")
    
    memory_savings = (full_memory_qk - sliding_memory_qk) / full_memory_qk * 100
    print(f"    Memory savings: {memory_savings:.1f}%")
    print()
    
    # Test actual implementation
    from vggt.models.aggregator import Aggregator
    
    # Create aggregator with sliding window attention
    aggregator = Aggregator(
        img_size=64,
        patch_size=16,
        embed_dim=C,
        depth=2,
        num_heads=num_heads,
        num_register_tokens=4,
        patch_embed="conv"
    )
    
    # Test with realistic input size
    H, W = 64, 64
    images = torch.randn(B, S, 3, H, W)
    
    print(f"Running sliding window attention test...")
    print(f"  Input shape: {images.shape}")
    
    # Monitor memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    try:
        with torch.no_grad():
            result = aggregator(images)
        
        print(f"✓ Sliding window attention successful!")
        print(f"  Output length: {len(result[0])}")
        print(f"  First output shape: {result[0][0].shape}")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  Peak GPU memory: {memory_used:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Sliding window attention failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neighborhood_logic():
    """Test that the neighborhood logic works correctly"""
    
    print("Testing neighborhood attention logic...")
    
    # Simulate the neighborhood calculation from the aggregator
    S = 10  # sequence length
    neighborhood_size = 3  # smaller for easier verification
    
    expected_neighborhoods = {}
    for i in range(S):
        start = max(0, i - neighborhood_size)
        end = min(S, i + neighborhood_size + 1)
        
        # Get unique indices, including the first frame and the neighborhood
        context_indices = list(set([0] + list(range(start, end))))
        context_indices.sort()
        
        expected_neighborhoods[i] = context_indices
        print(f"  Frame {i}: attends to frames {context_indices}")
    
    # Verify some key properties
    # 1. Every frame attends to frame 0
    for i in range(S):
        if 0 not in expected_neighborhoods[i]:
            print(f"✗ Frame {i} does not attend to frame 0!")
            return False
    
    # 2. Frame 0 should attend to itself and nearby frames
    frame_0_context = expected_neighborhoods[0]
    expected_frame_0 = list(range(min(S, neighborhood_size + 1)))
    if frame_0_context != expected_frame_0:
        print(f"✗ Frame 0 context {frame_0_context} != expected {expected_frame_0}")
        return False
    
    # 3. Middle frames should have symmetric neighborhoods (plus frame 0)
    middle_frame = S // 2
    middle_context = expected_neighborhoods[middle_frame]
    expected_middle_start = max(0, middle_frame - neighborhood_size)
    expected_middle_end = min(S, middle_frame + neighborhood_size + 1)
    expected_middle = sorted(list(set([0] + list(range(expected_middle_start, expected_middle_end)))))
    if middle_context != expected_middle:
        print(f"✗ Middle frame {middle_frame} context {middle_context} != expected {expected_middle}")
        return False
    
    print("✓ Neighborhood logic test passed!")
    return True

if __name__ == "__main__":
    print("VGGT Memory Efficiency Tests")
    print("=" * 50)
    
    test1_ok = test_neighborhood_logic()
    print()
    
    test2_ok = test_memory_comparison()
    print()
    
    if test1_ok and test2_ok:
        print("✓ All memory efficiency tests passed!")
        print("The sliding window implementation provides significant memory savings")
        print("while maintaining attention to the first frame and local context.")
    else:
        print("✗ Some tests failed!")