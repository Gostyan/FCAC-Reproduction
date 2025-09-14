import pandas as pd
import numpy as np
import os
import glob
import json
from collections import defaultdict

def augment_nsynth_dataset(nsynth_root, output_file):
    """
    Augment NSynth-100 training dataset by finding additional samples
    for each instrument from unused NSynth data.
    """
    print("Loading original dataset...")
    
    # Load original training data
    train_df = pd.read_csv(os.path.join(nsynth_root, 'nsynth-100-fs_train.csv'))
    
    # Load label mapping
    with open(os.path.join(nsynth_root, 'nsynth-100-fs_vocab.json'), 'r') as f:
        label_mapping = json.load(f)
    
    train_df['label'] = train_df['instrument'].map(label_mapping)
    
    print(f"Original training samples: {len(train_df)}")
    
    # Focus on novel classes (60-99)
    novel_instruments = train_df[train_df['label'] >= 60]['instrument'].unique()
    print(f"Novel instruments to augment: {len(novel_instruments)}")
    
    # Collect all used filenames
    used_files = set(train_df['filename'].tolist())
    
    # Find additional files for each novel instrument
    augmented_rows = []
    
    for instrument in novel_instruments:
        print(f"Processing {instrument}...")
        
        # Find all available files for this instrument
        additional_files = []
        for subdir in ['nsynth-train', 'nsynth-valid', 'nsynth-test']:
            audio_dir = os.path.join(nsynth_root, 'The_NSynth_Dataset', subdir, 'audio')
            if os.path.exists(audio_dir):
                pattern = os.path.join(audio_dir, f'{instrument}-*.wav')
                files = glob.glob(pattern)
                for f in files:
                    basename = os.path.basename(f).replace('.wav', '')
                    if basename not in used_files:
                        additional_files.append((basename, subdir))
        
        print(f"  Found {len(additional_files)} additional files")
        
        # Add to augmented dataset
        for filename, audio_source in additional_files:
            augmented_rows.append({
                'filename': filename,
                'instrument': instrument,
                'audio_source': audio_source
            })
    
    # Combine original and augmented data
    print("Combining datasets...")
    augmented_df = pd.DataFrame(augmented_rows)
    final_df = pd.concat([train_df, augmented_df], ignore_index=True)
    
    # Save result
    final_df.to_csv(output_file, index=False)
    
    print(f"Augmented dataset saved: {output_file}")
    print(f"Total samples: {len(final_df)} (was {len(train_df)})")
    print(f"Added {len(augmented_df)} new samples")
    
    return final_df

if __name__ == "__main__":
    nsynth_root = "/path/to/NSynth-100"
    output_file = os.path.join(nsynth_root, "nsynth-100-fs_train_augmented.csv")
    augment_nsynth_dataset(nsynth_root, output_file)
