# debug.py - Run from project root: python src/debug.py
import os
import sys

# Try to find the project root and data
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}")

# Search for data in common locations
possible_data_paths = [
    os.path.join(project_root, 'data', 'dcase'),
    os.path.join(script_dir, 'data', 'dcase'),
    os.path.join(script_dir, '..', 'data', 'dcase'),
    './data/dcase',
    '../data/dcase',
    'C:/Users/jamil/PycharmProjects/fibonacci/dcase/data/dcase',
    'C:/Users/jamil/PycharmProjects/fibonacci/data/dcase',
]

data_path = None
for path in possible_data_paths:
    abs_path = os.path.abspath(path)
    train_meta = os.path.join(abs_path, 'train', 'meta.txt')
    print(f"Checking: {abs_path}")
    if os.path.exists(train_meta):
        data_path = abs_path
        print(f"  ✓ FOUND: {train_meta}")
        break
    else:
        print(f"  ✗ Not found")

if data_path is None:
    print("\n" + "=" * 60)
    print("ERROR: Could not find data directory!")
    print("=" * 60)
    print("\nPlease tell me the output of these commands:")
    print("1. Where is your dcase data folder?")
    print("2. Run this in Python:")
    print("""
import os
for root, dirs, files in os.walk('C:/Users/jamil/PycharmProjects/fibonacci'):
    if 'meta.txt' in files:
        print(f"Found meta.txt in: {root}")
    """)
    sys.exit(1)

print(f"\n✓ Using data path: {data_path}")

# Change to project root for imports
os.chdir(project_root)
sys.path.insert(0, script_dir)

import torch
import numpy as np
import soundfile as sf
import pandas as pd

print("\n" + "=" * 70)
print("STEP 1: CHECKING RAW AUDIO FILES")
print("=" * 70)

train_meta_path = os.path.join(data_path, 'train', 'meta.txt')
meta_df = pd.read_csv(train_meta_path, delimiter='\t', header=None,
                      names=['audio_path', 'scene_name', 'scene_id'])

print(f"Found {len(meta_df)} training samples")
print(f"Scene labels: {sorted(meta_df['scene_name'].unique())}")

# Check first audio file
sample_audio_path = os.path.join(data_path, 'train', meta_df.iloc[0]['audio_path'])
print(f"\nLoading: {sample_audio_path}")

audio_data, sr = sf.read(sample_audio_path, dtype=np.float32)
print(f"  Shape: {audio_data.shape}")  # Should be (SAMPLES, 2) for stereo
print(f"  Sample rate: {sr}")
print(f"  Duration: {audio_data.shape[0] / sr:.2f} seconds")

if len(audio_data.shape) == 1:
    print("  ⚠️ CRITICAL: Audio is MONO!")
elif audio_data.shape[1] == 2:
    left = audio_data[:, 0]
    right = audio_data[:, 1]
    print(f"  Left:  min={left.min():.4f}, max={left.max():.4f}, std={left.std():.4f}")
    print(f"  Right: min={right.min():.4f}, max={right.max():.4f}, std={right.std():.4f}")

    if np.allclose(left, right, atol=1e-6):
        print("\n  ⚠️⚠️⚠️ CRITICAL: LEFT AND RIGHT CHANNELS ARE IDENTICAL! ⚠️⚠️⚠️")
        print("  This means your audio is DUAL-MONO (mono stored as stereo)")
        print("  The 'side' channel (L-R)/2 will be ~0!")
        print("  SOLUTION: Use 'left'/'right' or 'harmonic'/'percussive' instead of 'mid'/'side'")
    else:
        diff = np.abs(left - right)
        print(f"  L-R difference: mean={diff.mean():.6f}, max={diff.max():.6f}")
        print("  ✓ Audio is TRUE STEREO")

print("\n" + "=" * 70)
print("STEP 2: CHECKING PREPROCESSING")
print("=" * 70)

from preprocessing import MultiStreamPreprocessor

# Use a test cache dir
test_cache_dir = os.path.join(data_path, 'debug_cache')
os.makedirs(test_cache_dir, exist_ok=True)

preprocessor = MultiStreamPreprocessor(sample_rate=44100, cache_dir=test_cache_dir)

# Convert to tensor
audio_tensor = torch.from_numpy(audio_data.T).float()  # (2, TIME)
print(f"Audio tensor shape: {audio_tensor.shape}")

# Clear test cache
test_cache_file = os.path.join(test_cache_dir, "debug_test_streams.pt")
if os.path.exists(test_cache_file):
    os.remove(test_cache_file)

streams = preprocessor.process(audio_tensor, cache_key="debug_test")

print(f"\nStream keys: {list(streams.keys())}")
for name, stream in streams.items():
    abs_max = stream.abs().max().item()
    print(f"  {name:12s}: shape={stream.shape}, min={stream.min():.4f}, max={stream.max():.4f}, std={stream.std():.4f}")
    if abs_max < 1e-6:
        print(f"               ⚠️ WARNING: '{name}' is ALL ZEROS!")
    elif abs_max < 1e-3:
        print(f"               ⚠️ WARNING: '{name}' is nearly silent!")

# Check mid vs side
if 'mid' in streams and 'side' in streams:
    side_max = streams['side'].abs().max().item()
    if side_max < 1e-4:
        print("\n  ⚠️⚠️⚠️ SIDE CHANNEL IS ~0! Audio is dual-mono! ⚠️⚠️⚠️")

print("\n" + "=" * 70)
print("STEP 3: CHECKING CACHE KEY BUG")
print("=" * 70)

# Check if cache key includes dataset name
print("Current cache_key generation in dataset.py:")
print('  cache_key = os.path.splitext(os.path.basename(example["audio_path"]))[0]')
print("")
print("This does NOT include dataset_name! Example:")
print(f"  train file: {meta_df.iloc[0]['audio_path']}")
train_cache_key = os.path.splitext(os.path.basename(meta_df.iloc[0]['audio_path']))[0]
print(f"  train cache_key: {train_cache_key}")

# Check val
val_meta_path = os.path.join(data_path, 'val', 'meta.txt')
if os.path.exists(val_meta_path):
    val_df = pd.read_csv(val_meta_path, delimiter='\t', header=None,
                         names=['audio_path', 'scene_name', 'scene_id'])
    val_cache_key = os.path.splitext(os.path.basename(val_df.iloc[0]['audio_path']))[0]
    print(f"  val file: {val_df.iloc[0]['audio_path']}")
    print(f"  val cache_key: {val_cache_key}")

    if train_cache_key == val_cache_key:
        print("\n  ⚠️⚠️⚠️ CACHE KEY COLLISION! Train and val would share cached features! ⚠️⚠️⚠️")

print("\n" + "=" * 70)
print("STEP 4: TESTING DATASET")
print("=" * 70)

from dataset import AcousticScenesDataset

# Test with mid/side
print("\nTesting dataset with input_channels=['mid', 'side']:")
try:
    ds = AcousticScenesDataset(
        dataset_name='train',
        base_data_path=data_path,
        sample_rate=44100,
        mono=False,
        normalize_audio=True,
        multi_stream=True,
        stream_cache_dir=test_cache_dir,
        input_channels=['mid', 'side'],
    )

    sample = ds[0]
    print(f"  Sample keys: {list(sample.keys())}")
    if 'audio_ch1' in sample:
        ch1 = sample['audio_ch1']
        ch2 = sample['audio_ch2']
        print(f"  audio_ch1 (mid):  shape={ch1.shape}, max={ch1.abs().max():.4f}, std={ch1.std():.4f}")
        print(f"  audio_ch2 (side): shape={ch2.shape}, max={ch2.abs().max():.4f}, std={ch2.std():.4f}")

        if ch2.abs().max() < 1e-4:
            print("\n  ⚠️ audio_ch2 (side) is ~0!")
            print("  This will break training - the model sees zeros as one input!")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 70)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 70)

print("""
If 'side' channel is ~0 (dual-mono audio):
  → Change dccnn_ms.yaml: input_channels: ["harmonic", "percussive"]
  → Or use: input_channels: ["left", "right"] (they'll be identical but won't be zeros)

If cache key collision exists:
  → Fix dataset.py line ~140, change:
    cache_key = os.path.splitext(os.path.basename(example['audio_path']))[0]
  → To:
    cache_key = f"{self.dataset_name}_{os.path.splitext(os.path.basename(example['audio_path']))[0]}"
  → Then DELETE your cache: rm -rf ./data/dcase/preprocessed_features/*
""")

print("=" * 70)