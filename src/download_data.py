import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Mapping for FER2013
# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
emotion_map = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')

def download_and_save_data(max_samples_per_class=1000):
    print("Loading FER2013 dataset from Hugging Face...")
    candidates = [
        "AutumnQiu/fer2013",
        "Jeneral/fer-2013",
        "darragh/fer2013",
        "Pavan-P/fer2013"
    ]
    
    dataset = None
    for candidate in candidates:
        try:
            print(f"Trying to load {candidate}...")
            dataset = load_dataset(candidate, split="train")
            print(f"Successfully loaded {candidate}!")
            break
        except Exception as e:
            print(f"Failed to load {candidate}: {e}")
            
    if dataset is None:
        print("Could not load any Hugging Face dataset automatically.")
        print("Please download FER2013 manually.")
        return

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Counters
    counts = {label: 0 for label in emotion_map.values()}
    
    print(f"Saving images to {DATA_DIR}...")
    
    # Iterate and save
    for item in tqdm(dataset):
        try:
            # Adjust based on dataset structure. usually 'image' and 'label' or 'emotion'
            img = item.get('image')
            label_idx = item.get('label') if 'label' in item else item.get('emotion')
            
            if img is None or label_idx is None:
                continue
                
            label_name = emotion_map.get(label_idx)
            if not label_name:
                continue

            if counts[label_name] >= max_samples_per_class:
                continue

            # Save
            label_dir = os.path.join(DATA_DIR, label_name)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            
            # Ensure 48x48 and grayscale
            img = img.resize((48, 48)).convert('L')
            
            save_path = os.path.join(label_dir, f"{label_name}_{counts[label_name]}.jpg")
            img.save(save_path)
            
            counts[label_name] += 1
            
            # Stop if we have enough of everything
            if all(c >= max_samples_per_class for c in counts.values()):
                break
                
        except Exception as e:
            # print(f"Error saving image: {e}")
            continue
            
    print("Download and extraction complete.")
    for label, count in counts.items():
        print(f"  - {label}: {count} images")

if __name__ == "__main__":
    download_and_save_data(max_samples_per_class=2000) # Get 2000 per class for good accuracy
