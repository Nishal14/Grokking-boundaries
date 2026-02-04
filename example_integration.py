"""
Example: Integrating the decoder-only Transformer with tokenizer and dataset.

REQUIREMENTS:
    - CUDA-capable GPU required
    - No CPU fallback provided

This demonstrates:
1. GPU availability check
2. Loading the tokenizer and dataset
3. Creating the model on GPU
4. Running a forward pass on GPU
5. Computing loss
6. Inspecting attention patterns
"""
import torch
import torch.nn.functional as F
from src.tokenizer import ModularArithmeticTokenizer
from src.dataset import ModularArithmeticDataset
from src.model import DecoderOnlyTransformer
from src.gpu_check import check_cuda_available


def main():
    print("="*60)
    print("Integration Example: Model + Tokenizer + Dataset (GPU)")
    print("="*60)
    print()

    # ========================================================================
    # 0. Check CUDA Availability
    # ========================================================================
    device = check_cuda_available()

    # ========================================================================
    # 1. Setup Components
    # ========================================================================
    print("1. Setting up components...")

    # Tokenizer
    tokenizer = ModularArithmeticTokenizer(modulus_p=97)
    print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")

    # Dataset
    train_dataset = ModularArithmeticDataset(
        modulus_p=97,
        split='train',
        train_split=0.8,
        seed=42
    )
    print(f"   Train dataset size: {len(train_dataset):,}")

    # Model (on GPU)
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_layers=4,
        num_heads=4,
        d_ff=512,
        max_seq_len=32,
        dropout=0.1
    )
    model = model.to(device)  # Move to GPU
    print(f"   Model parameters: {model.count_parameters():,}")
    print(f"   Model device: {next(model.parameters()).device}")
    print()

    # ========================================================================
    # 2. Get a Sample from Dataset
    # ========================================================================
    print("2. Getting a sample from dataset...")
    sample = train_dataset[0]

    print(f"   Sample keys: {list(sample.keys())}")
    print(f"   Input IDs shape: {sample['input_ids'].shape}")
    print(f"   Labels shape: {sample['labels'].shape}")
    print(f"   a = {sample['a']}, b = {sample['b']}, result = {sample['result']}")

    # Decode to see the sequence
    text = tokenizer.decode(sample['input_ids'].tolist())
    print(f"   Text: {text}")
    print()

    # ========================================================================
    # 3. Run Forward Pass
    # ========================================================================
    print("3. Running forward pass...")

    # Add batch dimension and move to GPU
    input_ids = sample['input_ids'].unsqueeze(0).to(device)  # (1, seq_len)
    labels = sample['labels'].unsqueeze(0).to(device)        # (1, seq_len)

    # Forward pass on GPU (no gradient for this example)
    with torch.no_grad():
        logits, attentions = model(input_ids)

    print(f"   Logits shape: {logits.shape}")
    print(f"   Number of attention tensors: {len(attentions)}")
    print(f"   Attention shape (per layer): {attentions[0].shape}")
    print()

    # ========================================================================
    # 4. Compute Loss
    # ========================================================================
    print("4. Computing loss...")

    # Cross-entropy loss (flatten batch and sequence dimensions)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),  # (batch*seq_len, vocab_size)
        labels.view(-1)                     # (batch*seq_len,)
    )

    print(f"   Loss: {loss.item():.4f}")
    print()

    # ========================================================================
    # 5. Inspect Predictions
    # ========================================================================
    print("5. Inspecting predictions...")

    # Get predicted tokens
    predicted_ids = logits.argmax(dim=-1)  # (1, seq_len)
    predicted_text = tokenizer.decode(predicted_ids[0].tolist())

    print(f"   Ground truth: {tokenizer.decode(labels[0].tolist())}")
    print(f"   Predicted:    {predicted_text}")

    # Check accuracy at result position (position 6)
    # Sequence: <bos> a + b mod p = c <eos>
    # Labels:   a + b mod p = c <eos>  (positions 0-7)
    # Position 6 is 'c' (the result)
    result_position = 6
    predicted_result = predicted_ids[0, result_position].item()
    true_result = sample['result']

    print(f"   True result (c): {true_result}")
    print(f"   Predicted result: {predicted_result}")
    print(f"   Correct: {predicted_result == true_result}")
    print()

    # ========================================================================
    # 6. Inspect Attention Patterns
    # ========================================================================
    print("6. Inspecting attention patterns...")

    # Look at first layer, first head
    first_layer_attn = attentions[0]  # (1, num_heads, seq_len, seq_len)
    first_head_attn = first_layer_attn[0, 0]  # (seq_len, seq_len)

    print(f"   Attention matrix shape (Layer 0, Head 0): {first_head_attn.shape}")
    print(f"   Attention is causal (lower-triangular):")
    print(f"   First row (position 0 can only attend to itself):")
    print(f"      {first_head_attn[0].tolist()[:3]} ... (only first position non-zero)")
    print(f"   Last row (position 7 can attend to all previous):")
    print(f"      {[f'{x:.3f}' for x in first_head_attn[-1].tolist()[:8]]}")
    print()

    # Check that attention sums to 1
    attn_sums = first_head_attn.sum(dim=-1)
    print(f"   Attention sums (should be 1.0): {attn_sums.tolist()[:4]} ...")
    print()

    # ========================================================================
    # 7. Batch Processing
    # ========================================================================
    print("7. Testing batch processing...")

    # Get multiple samples
    batch_size = 4
    batch = [train_dataset[i] for i in range(batch_size)]

    # Stack into batch tensors and move to GPU
    input_ids_batch = torch.stack([b['input_ids'] for b in batch]).to(device)
    labels_batch = torch.stack([b['labels'] for b in batch]).to(device)

    print(f"   Batch input shape: {input_ids_batch.shape}")
    print(f"   Batch device: {input_ids_batch.device}")

    # Forward pass on batch (GPU)
    with torch.no_grad():
        logits_batch, attentions_batch = model(input_ids_batch)

    print(f"   Batch logits shape: {logits_batch.shape}")
    print(f"   Batch attention shape: {attentions_batch[0].shape}")

    # Batch loss
    loss_batch = F.cross_entropy(
        logits_batch.view(-1, logits_batch.size(-1)),
        labels_batch.view(-1)
    )
    print(f"   Batch loss: {loss_batch.item():.4f}")
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("="*60)
    print("Integration Complete!")
    print("="*60)
    print()
    print("The model successfully integrates with:")
    print("  [OK] Tokenizer (vocab size 104)")
    print("  [OK] Dataset (modular arithmetic)")
    print("  [OK] Forward pass (single and batch)")
    print("  [OK] Loss computation")
    print("  [OK] Attention inspection")
    print()
    print("Ready for training loop implementation!")
    print()


if __name__ == "__main__":
    main()
