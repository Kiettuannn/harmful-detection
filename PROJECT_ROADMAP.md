# 🎓 Lộ Trình Đồ Án: Nhận Diện Video Độc Hại TikTok

## 📊 Tổng Quan Dự Án

**Mục tiêu:** Phát triển hệ thống nhận diện video độc hại dành cho trẻ em trên TikTok sử dụng kiến trúc multimodal (Audio + Text + Video)

**Dataset:**

- Tổng số videos: ~2,000 videos
- Split: Train (70%) / Val (15%) / Test (15%)
- Labels: 15 categories (multilabel classification)
- Storage: ~20GB total

---

## 💡 Quyết Định Kiến Trúc

### ❓ Tại sao KHÔNG pre-extract ViViT embeddings?

**ViViT là TRAINABLE model (khác với wav2vec2):**

| Feature      | Audio (MFCC/Transcript)     | Video (ViViT)                  |
| ------------ | --------------------------- | ------------------------------ |
| Model        | wav2vec2 (Frozen)           | ViViT (Trainable)              |
| Purpose      | Feature extraction only     | Feature learning + extraction  |
| Pre-extract? | ✅ YES                      | ❌ NO                          |
| Lý do        | Không train, output cố định | Cần fine-tune, output thay đổi |

```python
# Audio: Pre-extract OK
with torch.no_grad():
    transcript = wav2vec2.transcribe(audio)  # Frozen → output không đổi

# Video: KHÔNG pre-extract
video_features = vivit(frames)  # Trainable → output thay đổi mỗi epoch
```

### ❓ Có cần Hugging Face không?

**Không cần!** (với dataset 2K videos, đồ án cá nhân)

**Lý do:**

- Dataset 20GB → Vừa phải, lưu local OK
- Không cần share public
- Đơn giản hơn, nhanh hơn

**Alternative backup:**

- Google Drive / OneDrive
- External HDD
- Git LFS (nếu cần version control)

---

## 📁 Cấu Trúc Thư Mục Final

```
harmful-detection/
│
├── PROJECT_ROADMAP.md                    # File này
├── README.md
│
├── hf_dataset_merged/
│   ├── metadata_train.csv                # Master metadata file
│   ├── metadata_val.csv
│   ├── metadata_test.csv
│   │
│   ├── data/                             # RAW DATA
│   │   ├── videos/                       # (Optional - có thể xóa sau khi extract)
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   │
│   │   ├── audio/                        # WAV files (16kHz mono)
│   │   │   ├── train/
│   │   │   │   └── {video_id}.wav
│   │   │   ├── val/
│   │   │   └── test/
│   │   │
│   │   └── frames/                       # Extracted frames (JPG)
│   │       ├── train/
│   │       │   └── {video_id}/
│   │       │       ├── frame_000.jpg
│   │       │       ├── frame_001.jpg
│   │       │       └── ...
│   │       ├── val/
│   │       └── test/
│   │
│   └── features/                         # PRE-EXTRACTED FEATURES
│       ├── transcripts/                  # ASR output (JSON)
│       │   ├── train/
│       │   │   └── {video_id}.json
│       │   ├── val/
│       │   └── test/
│       │
│       └── mfcc/                         # MFCC arrays (NPY)
│           ├── train/
│           │   └── {video_id}.npy
│           ├── val/
│           └── test/
│
├── preprocessing/
│   ├── 1_extract_audio.py                # Extract audio từ video
│   ├── 2_extract_frames.py               # Extract frames từ video
│   ├── 3_extract_mfcc.py                 # Extract MFCC từ audio
│   ├── 4_extract_transcripts.ipynb       # ASR với wav2vec2 (Colab)
│   └── 5_create_metadata.py              # Generate metadata CSV
│
├── dataset.py                            # PyTorch Dataset class
│
├── models/
│   ├── audio_branch.py                   # CNN cho MFCC
│   ├── text_branch.py                    # BARTpho cho transcript
│   ├── video_branch.py                   # ViViT cho frames
│   └── multimodal.py                     # Combined model
│
├── training/
│   ├── train_audio.ipynb                 # Train audio branch (Colab)
│   ├── train_text.ipynb                  # Train text branch (Colab)
│   ├── train_video.ipynb                 # Train video branch (Kaggle)
│   └── train_multimodal.ipynb            # Train full model (Kaggle)
│
├── evaluation/
│   └── evaluate.ipynb                    # Evaluation & metrics
│
└── utils/
    ├── transforms.py                     # Data augmentation
    └── metrics.py                        # Evaluation metrics
```

---

## 📋 Metadata CSV Structure

### `metadata_train.csv`:

```csv
video_id,labels,video_path,audio_path,frames_dir,transcript_path,mfcc_path,duration,has_audio,num_frames,split
61f8c684-7279806259790925074,cultural_violation,data/videos/train/61f8c684-7279806259790925074.mp4,data/audio/train/61f8c684-7279806259790925074.wav,data/frames/train/61f8c684-7279806259790925074,features/transcripts/train/61f8c684-7279806259790925074.json,features/mfcc/train/61f8c684-7279806259790925074.npy,45.2,true,30,train
d18398c5-7564193864886586680,sexual_harm,data/videos/train/d18398c5-7564193864886586680.mp4,data/audio/train/d18398c5-7564193864886586680.wav,data/frames/train/d18398c5-7564193864886586680,features/transcripts/train/d18398c5-7564193864886586680.json,features/mfcc/train/d18398c5-7564193864886586680.npy,38.5,true,25,train
f941ff2b-Download_18.mp4,"horror_scary,ai_slop",data/videos/train/f941ff2b-Download_18.mp4,data/audio/train/f941ff2b-Download_18.wav,data/frames/train/f941ff2b-Download_18,features/transcripts/train/f941ff2b-Download_18.json,features/mfcc/train/f941ff2b-Download_18.npy,52.1,true,35,train
```

**Columns:**

- `video_id`: Unique identifier
- `labels`: Comma-separated labels (multilabel)
- `video_path`: Path to original video
- `audio_path`: Path to extracted WAV
- `frames_dir`: Directory containing extracted frames
- `transcript_path`: Path to transcript JSON
- `mfcc_path`: Path to MFCC numpy array
- `duration`: Video duration in seconds
- `has_audio`: Boolean (true/false)
- `num_frames`: Number of extracted frames
- `split`: train/val/test

---

## 💾 Storage Requirements

```
Videos (raw):      ~10GB  → Có thể xóa sau khi extract
Audio WAV:         ~4GB   ✅ Giữ
Frames JPG:        ~6GB   ✅ Giữ
Transcripts JSON:  ~20MB  ✅ Giữ
MFCC NPY:          ~400MB ✅ Giữ
─────────────────────────────────
Total cần giữ:     ~10.5GB ✅ Chấp nhận được
```

---

## 🚀 LỘ TRÌNH THỰC HIỆN

### **PHASE 2: PRE-PROCESSING** (Tuần 2-3)

**Chiến lược:** Xử lý theo batches nhỏ (200 videos/lần) để tránh crash

#### **Bước 2.1: Extract Audio** ⚡ (Local)

**Script:** `preprocessing/1_extract_audio.py`

```python
from moviepy.editor import VideoFileClip
import os
from tqdm import tqdm

def extract_audio(video_path, output_path, sr=16000):
    """Extract audio từ video và convert sang 16kHz mono WAV"""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is not None:
            audio.write_audiofile(
                output_path,
                fps=sr,
                codec='pcm_s16le',
                nbytes=2,
                ffmpeg_params=["-ac", "1"]  # Mono
            )
        video.close()
        return True
    except Exception as e:
        print(f"Error: {video_path} - {e}")
        return False

# Process all videos
for split in ['train', 'val', 'test']:
    video_dir = f'hf_dataset_merged/data/videos/{split}'
    audio_dir = f'hf_dataset_merged/data/audio/{split}'
    os.makedirs(audio_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    for video_file in tqdm(video_files, desc=f'Extracting {split}'):
        video_path = os.path.join(video_dir, video_file)
        video_id = os.path.splitext(video_file)[0]
        audio_path = os.path.join(audio_dir, f'{video_id}.wav')

        if not os.path.exists(audio_path):
            extract_audio(video_path, audio_path)
```

**Thời gian:** ~10-15 phút cho 200 videos
**Output:** `data/audio/{split}/{video_id}.wav`

---

#### **Bước 2.2: Extract Frames** 🎬 (Local)

**Script:** `preprocessing/2_extract_frames.py`

**Best Practices (theo ViViT paper + Kinetics standard + MMAction2):**

- ✅ **FPS Normalization**: KHÔNG CẦN - Confirmed từ ViViT paper, Kinetics, MMAction2
- ✅ **Short Videos**: Loop frames (consistent với research implementations)
- ✅ **Aspect Ratio**: Resize shorter edge → 256, center crop 224x224 (Kinetics standard)
- ✅ **Watermark**: Chấp nhận (standard practice, model learns robustness)
- ✅ **Video Loading**: Decord recommended (10-100x faster, used in MMAction2)
- ✅ **Robustness**: Try-except với OpenCV fallback

```python
"""
Frame Extraction following Best Practices:
- ViViT Paper (ICCV 2021)
- Kinetics Dataset preprocessing
- MMAction2 framework standards

Requirements:
    pip install decord opencv-python numpy tqdm
"""
import cv2
import os
import numpy as np
from tqdm import tqdm
import logging
from decord import VideoReader, cpu

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def center_crop_resize(frame, target_size=224, resize_size=256):
    """
    Kinetics-style preprocessing: Resize shorter edge → 256, center crop 224
    This is the standard from ViViT paper and Kinetics dataset

    Args:
        frame: Input frame
        target_size: Final crop size (default: 224)
        resize_size: Intermediate resize (default: 256, Kinetics standard)
    """
    h, w = frame.shape[:2]

    # Step 1: Resize shorter edge to resize_size (Kinetics standard: 256)
    scale = resize_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Use INTER_LINEAR (standard) instead of LANCZOS4 (faster, similar quality)
    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Step 2: Center crop to target_size x target_size
    start_y = (new_h - target_size) // 2
    start_x = (new_w - target_size) // 2
    frame_cropped = frame_resized[start_y:start_y + target_size,
                                  start_x:start_x + target_size]

    return frame_cropped


def extract_frames_uniform(video_path, output_dir, num_frames=16, target_size=224, resize_size=256):
    """
    Extract frames using Decord (10-100x faster than OpenCV)
    Following best practices from ViViT paper + Kinetics dataset

    Features:
    - Decord backend (fast C++ decoder)
    - Uniform temporal sampling (no FPS normalization needed)
    - Handle short videos → loop frames (consistent with research)
    - Kinetics-style preprocessing: resize 256 → crop 224
    - Batch frame loading (efficient memory usage)

    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        num_frames: Number of frames to extract (default: 16, ViViT standard)
        target_size: Final crop size (default: 224)
        resize_size: Intermediate resize (default: 256, Kinetics standard)

    Returns:
        dict: {
            'success': bool,
            'extracted': int,
            'total_frames': int,
            'looped': bool
        }
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Initialize Decord VideoReader
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        if total_frames == 0:
            logger.warning(f"Video has 0 frames: {video_path}")
            return {'success': False, 'extracted': 0, 'total_frames': 0, 'looped': False}

        # Handle short videos: Loop frames nếu total_frames < num_frames
        looped = False
        if total_frames < num_frames:
            logger.info(f"Short video ({total_frames} frames < {num_frames}). Will loop frames.")
            base_indices = np.arange(total_frames)
            repeats = (num_frames // total_frames) + 1
            frame_indices = np.tile(base_indices, repeats)[:num_frames]
            looped = True
        else:
            # Uniform sampling - ViViT standard
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        # Batch load frames (FAST!) - Decord's killer feature
        frames_batch = vr.get_batch(frame_indices).asnumpy()  # [N, H, W, 3] RGB

        # Process and save frames
        extracted = 0
        for i, frame in enumerate(frames_batch):
            try:
                # Validate frame
                if frame.size == 0:
                    logger.warning(f"Empty frame at index {i}")
                    continue

                # Kinetics-style preprocessing: resize 256 → crop 224
                frame_processed = center_crop_resize(frame,
                                                    target_size=target_size,
                                                    resize_size=resize_size)

                # Verify processed frame
                if frame_processed.shape != (target_size, target_size, 3):
                    logger.error(f"Invalid processed frame shape: {frame_processed.shape}")
                    continue

                # Save frame (convert RGB → BGR for OpenCV)
                frame_path = os.path.join(output_dir, f'frame_{extracted:03d}.jpg')
                cv2.imwrite(frame_path,
                           cv2.cvtColor(frame_processed, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted += 1

            except Exception as e:
                logger.error(f"Error processing frame {i}: {e}")
                continue

        # Success nếu extract được >= 80% frames
        success = extracted >= num_frames * 0.8

        if not success:
            logger.error(f"Extraction failed: {video_path} - Only {extracted}/{num_frames} frames")

        return {
            'success': success,
            'extracted': extracted,
            'total_frames': total_frames,
            'looped': looped
        }

    except Exception as e:
        logger.error(f"Decord error for {video_path}: {e}")
        return {'success': False, 'extracted': 0, 'total_frames': 0, 'looped': False}


# Process all videos
NUM_FRAMES = 16  # Sample 16 frames per video (ViViT standard)
TARGET_SIZE = 224  # ViViT input size

if __name__ == "__main__":
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'looped': 0,
        'skipped': 0
    }

    failed_videos = []

    for split in ['train', 'val', 'test']:
        video_dir = f'hf_dataset_merged/data/videos/{split}'
        frames_dir = f'hf_dataset_merged/data/frames/{split}'

        if not os.path.exists(video_dir):
            logger.warning(f"Video directory not found: {video_dir}")
            continue

        os.makedirs(frames_dir, exist_ok=True)

        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        logger.info(f"\nProcessing {split} split: {len(video_files)} videos")

        for video_file in tqdm(video_files, desc=f'Extracting {split}'):
            video_path = os.path.join(video_dir, video_file)
            video_id = os.path.splitext(video_file)[0]
            output_dir = os.path.join(frames_dir, video_id)

            stats['total'] += 1

            # Skip if already processed
            if os.path.exists(output_dir):
                existing_frames = len([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
                if existing_frames >= NUM_FRAMES:
                    stats['skipped'] += 1
                    continue

            result = extract_frames_uniform(
                video_path,
                output_dir,
                num_frames=NUM_FRAMES,
                target_size=TARGET_SIZE
            )

            if result['success']:
                stats['success'] += 1
                if result['looped']:
                    stats['looped'] += 1
            else:
                stats['failed'] += 1
                failed_videos.append({
                    'path': video_path,
                    'extracted': result['extracted'],
                    'total_frames': result['total_frames']
                })

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("EXTRACTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total videos: {stats['total']}")
    logger.info(f"✅ Success: {stats['success']} ({stats['success']/max(stats['total']-stats['skipped'],1)*100:.1f}%)")
    logger.info(f"⏭️  Skipped: {stats['skipped']}")
    logger.info(f"🔄 Looped (short videos): {stats['looped']}")
    logger.info(f"❌ Failed: {stats['failed']}")

    if failed_v
- ~12-15 phút với Decord (recommended)
- ~30-40 phút với OpenCV (fallback)

**Output:** `data/frames/{split}/{video_id}/frame_*.jpg` (224x224 JPG)

**Note:** Code follow best practices từ:
- ViViT Paper (Google Research, ICCV 2021)
- Kinetics Dataset preprocessing (DeepMind)
- MMAction2 framework (OpenMMLabencoding='utf-8') as f:
            for fail in failed_videos:
                f.write(f"{fail['path']}\n")
```

**Thời gian:** ~30-40 phút cho 2,800 videos (với resize + crop)
**Output:** `data/frames/{split}/{video_id}/frame_*.jpg` (224x224 JPG)

---

#### **Bước 2.3: Extract MFCC** 🎼 (Local)

**Script:** `preprocessing/3_extract_mfcc.py`

```python
import librosa
import numpy as np
import os
from tqdm import tqdm

def extract_mfcc(audio_path, n_mfcc=40):
    """Extract MFCC features từ audio"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=2048,
            hop_length=512
        )

        # Optional: Add delta and delta-delta
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Stack all features
        mfcc_features = {
            'mfcc': mfcc,                    # [40, T]
            'delta': mfcc_delta,             # [40, T]
            'delta2': mfcc_delta2,           # [40, T]
            'sr': sr,
            'duration': librosa.get_duration(y=audio, sr=sr)
        }

        return mfcc_features
    except Exception as e:
        print(f"Error: {audio_path} - {e}")
        return None

# Process all audio files
for split in ['train', 'val', 'test']:
    audio_dir = f'hf_dataset_merged/data/audio/{split}'
    mfcc_dir = f'hf_dataset_merged/features/mfcc/{split}'
    os.makedirs(mfcc_dir, exist_ok=True)

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

    for audio_file in tqdm(audio_files, desc=f'Extracting MFCC {split}'):
        audio_path = os.path.join(audio_dir, audio_file)
        video_id = os.path.splitext(audio_file)[0]
        mfcc_path = os.path.join(mfcc_dir, f'{video_id}.npy')

        if not os.path.exists(mfcc_path):
            mfcc_features = extract_mfcc(audio_path, n_mfcc=40)
            if mfcc_features is not None:
                np.save(mfcc_path, mfcc_features)
```

**Thời gian:** ~5-10 phút cho 200 videos
**Output:** `features/mfcc/{split}/{video_id}.npy`

---

#### **Bước 2.4: Extract Transcripts (ASR)** 🗣️ (Google Colab)

**Tại sao Colab?** wav2vec2 cần GPU để nhanh

**Notebook:** `preprocessing/4_extract_transcripts.ipynb` (Chạy trên Colab)

```python
# === COLAB NOTEBOOK ===

# 1. Mount Google Drive (upload audio files trước)
from google.colab import drive
drive.mount('/content/drive')

# 2. Install dependencies
!pip install transformers torch torchaudio

# 3. Load model
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import torchaudio

MODEL_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"

processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to('cuda')
model.eval()

# 4. Transcribe function
def transcribe(audio_path):
    """Transcribe audio thành text"""
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample nếu cần
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Process
        input_values = processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values.to('cuda')

        # Inference
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = processor.decode(predicted_ids[0])

        return transcript
    except Exception as e:
        print(f"Error: {audio_path} - {e}")
        return ""

# 5. Process all audio files
import os
import json
from tqdm import tqdm

for split in ['train', 'val', 'test']:
    audio_dir = f'/content/drive/MyDrive/harmful-detection/data/audio/{split}'
    transcript_dir = f'/content/drive/MyDrive/harmful-detection/features/transcripts/{split}'
    os.makedirs(transcript_dir, exist_ok=True)

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

    for audio_file in tqdm(audio_files, desc=f'Transcribing {split}'):
        audio_path = os.path.join(audio_dir, audio_file)
        video_id = os.path.splitext(audio_file)[0]
        transcript_path = os.path.join(transcript_dir, f'{video_id}.json')

        if not os.path.exists(transcript_path):
            transcript = transcribe(audio_path)

            # Save
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'video_id': video_id,
                    'raw_transcript': transcript,
                    'normalized_transcript': transcript,  # TODO: Add BARTpho normalization
                    'has_audio': len(transcript) > 0
                }, f, ensure_ascii=False, indent=2)

        # Save checkpoint mỗi 50 files
        if (audio_files.index(audio_file) + 1) % 50 == 0:
            print(f"Checkpoint: Processed {audio_files.index(audio_file) + 1} files")
```

**Thời gian:** ~1-2 giờ cho 2,000 videos (với GPU T4)
**Output:** `features/transcripts/{split}/{video_id}.json`

**⚠️ Tip:** Colab Free timeout sau 12h → Chia thành nhiều sessions, save progress thường xuyên

---

#### **Bước 2.5: Create Metadata CSV** 📊 (Local)

**Script:** `preprocessing/5_create_metadata.py`

```python
import pandas as pd
import os
import json
import librosa

def create_metadata(original_metadata_path, output_path, split):
    """Create enhanced metadata CSV"""

    # Load original metadata
    df = pd.read_csv(original_metadata_path)

    # Add new columns
    enhanced_data = []

    for idx, row in df.iterrows():
        video_file = row['file_name']
        video_id = os.path.splitext(os.path.basename(video_file))[0]

        # Paths
        video_path = f"data/videos/{split}/{video_id}.mp4"
        audio_path = f"data/audio/{split}/{video_id}.wav"
        frames_dir = f"data/frames/{split}/{video_id}"
        transcript_path = f"features/transcripts/{split}/{video_id}.json"
        mfcc_path = f"features/mfcc/{split}/{video_id}.npy"

        # Get duration
        duration = 0.0
        if os.path.exists(audio_path):
            try:
                duration = librosa.get_duration(filename=audio_path)
            except:
                pass

        # Check has_audio
        has_audio = os.path.exists(audio_path) and duration > 0

        # Count frames
        num_frames = 0
        if os.path.exists(frames_dir):
            num_frames = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

        enhanced_data.append({
            'video_id': video_id,
            'labels': row['label'],
            'video_path': video_path,
            'audio_path': audio_path,
            'frames_dir': frames_dir,
            'transcript_path': transcript_path,
            'mfcc_path': mfcc_path,
            'duration': round(duration, 2),
            'has_audio': has_audio,
            'num_frames': num_frames,
            'split': split
        })

    # Create DataFrame
    enhanced_df = pd.DataFrame(enhanced_data)
    enhanced_df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(enhanced_df)} records")

    return enhanced_df

# Create metadata for all splits
for split in ['train', 'val', 'test']:
    original_path = f'hf_dataset_merged/metadata_{split}.csv'
    output_path = f'hf_dataset_merged/metadata_{split}_enhanced.csv'
    create_metadata(original_path, output_path, split)
```

**Output:** Enhanced metadata CSV files with all paths and statistics

---

**PHASE 2 Summary:**

✅ Audio extracted (WAV files)
✅ Frames extracted (JPG files)
✅ MFCC features extracted (NPY files)
✅ Transcripts extracted (JSON files)
✅ Enhanced metadata CSV created

**Platform:** 💻 Local + ☁️ Colab
**Chi phí:** 🆓 Free

---

### **PHASE 3: DATASET CLASS & DATALOADER** (Tuần 4)

#### **Bước 3.1: Tạo Dataset Class**

**File:** `dataset.py`

```python
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
from PIL import Image
import os
from torchvision import transforms

class HarmfulVideoDataset(Dataset):
    """
    Multimodal dataset cho harmful video detection
    Load pre-extracted features (MFCC, transcripts) và raw frames
    """

    def __init__(self,
                 metadata_path,
                 root_dir='hf_dataset_merged',
                 max_frames=16,
                 max_mfcc_length=1000,
                 transform=None):

        self.metadata = pd.read_csv(metadata_path)
        self.root_dir = root_dir
        self.max_frames = max_frames
        self.max_mfcc_length = max_mfcc_length

        # Define labels
        self.all_labels = [
            "normal_content", "politically_sensitive", "cultural_violation",
            "sexual_harm", "graphic_violence", "horror_scary", "dangerous_acts",
            "self_harm_suicide", "toxic_behavior", "gambling_drugs", "scam_fraud",
            "ai_slop", "game", "game_risk_impact", "educational_history"
        ]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.all_labels)}
        self.num_classes = len(self.all_labels)

        # Image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        video_id = row['video_id']

        # === 1. Load MFCC (Pre-extracted) ===
        mfcc_path = os.path.join(self.root_dir, row['mfcc_path'])
        if os.path.exists(mfcc_path):
            mfcc_data = np.load(mfcc_path, allow_pickle=True).item()
            mfcc = mfcc_data['mfcc']  # [40, T]

            # Pad or truncate
            if mfcc.shape[1] > self.max_mfcc_length:
                mfcc = mfcc[:, :self.max_mfcc_length]
            else:
                pad_width = self.max_mfcc_length - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

            mfcc_tensor = torch.FloatTensor(mfcc)
        else:
            # Dummy MFCC nếu không có audio
            mfcc_tensor = torch.zeros(40, self.max_mfcc_length)

        # === 2. Load Transcript (Pre-extracted) ===
        transcript_path = os.path.join(self.root_dir, row['transcript_path'])
        if os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            transcript = transcript_data.get('normalized_transcript', '')
        else:
            transcript = ""

        # === 3. Load Video Frames (Raw JPG) ===
        frames_dir = os.path.join(self.root_dir, row['frames_dir'])
        frames = []

        if os.path.exists(frames_dir):
            frame_files = sorted([
                f for f in os.listdir(frames_dir)
                if f.endswith('.jpg')
            ])

            # Load frames
            for frame_file in frame_files[:self.max_frames]:
                frame_path = os.path.join(frames_dir, frame_file)
                try:
                    frame = Image.open(frame_path).convert('RGB')
                    frame = self.transform(frame)
                    frames.append(frame)
                except Exception as e:
                    print(f"Error loading frame: {frame_path} - {e}")

        # Pad frames nếu không đủ
        while len(frames) < self.max_frames:
            frames.append(torch.zeros(3, 224, 224))

        frames_tensor = torch.stack(frames[:self.max_frames])  # [N, 3, 224, 224]

        # === 4. Parse Labels (Multilabel) ===
        labels_str = row['labels']
        label_list = [l.strip() for l in labels_str.split(',')]

        # Binary vector
        label_vector = torch.zeros(self.num_classes)
        for label in label_list:
            if label in self.label_to_idx:
                label_vector[self.label_to_idx[label]] = 1

        return {
            'video_id': video_id,
            'frames': frames_tensor,        # [16, 3, 224, 224]
            'mfcc': mfcc_tensor,           # [40, 1000]
            'transcript': transcript,       # str
            'labels': label_vector,        # [15]
            'has_audio': row['has_audio']
        }

# Test dataset
if __name__ == "__main__":
    dataset = HarmfulVideoDataset(
        metadata_path='hf_dataset_merged/metadata_train_enhanced.csv',
        root_dir='hf_dataset_merged'
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")

    # Test loading
    sample = dataset[0]
    print(f"\nSample:")
    print(f"  Video ID: {sample['video_id']}")
    print(f"  Frames shape: {sample['frames'].shape}")
    print(f"  MFCC shape: {sample['mfcc'].shape}")
    print(f"  Transcript length: {len(sample['transcript'])} chars")
    print(f"  Labels: {sample['labels'].sum().item()} active")
```

#### **Bước 3.2: Test DataLoader**

**Notebook:** `test_dataloader.ipynb`

```python
from torch.utils.data import DataLoader
from dataset import HarmfulVideoDataset
import time

# Create dataset
train_dataset = HarmfulVideoDataset(
    metadata_path='hf_dataset_merged/metadata_train_enhanced.csv'
)

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Test loading speed
print("Testing dataloader speed...")
start_time = time.time()

for i, batch in enumerate(train_loader):
    if i >= 10:  # Test 10 batches
        break
    print(f"Batch {i+1}:")
    print(f"  Frames: {batch['frames'].shape}")
    print(f"  MFCC: {batch['mfcc'].shape}")
    print(f"  Labels: {batch['labels'].shape}")

elapsed = time.time() - start_time
print(f"\nLoaded 10 batches in {elapsed:.2f}s")
print(f"Average: {elapsed/10:.2f}s per batch")

# Visualize samples
import matplotlib.pyplot as plt

sample = train_dataset[0]
frames = sample['frames']

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(16):
    ax = axes[i // 8, i % 8]
    # Denormalize
    frame = frames[i].permute(1, 2, 0).numpy()
    frame = frame * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    frame = np.clip(frame, 0, 1)
    ax.imshow(frame)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

**Output:**

- ✅ Dataset class hoạt động
- ✅ DataLoader load nhanh (nhờ pre-extracted features)
- ✅ Visualizations OK

**Platform:** 💻 Local
**Chi phí:** 🆓 Free

---

### **PHASE 4: MODEL DEVELOPMENT** (Tuần 5-9)

**Chiến lược:** Incremental development - Train từng branch riêng trước, sau đó combine

#### **Stage 1: Audio Branch** (Tuần 5)

**File:** `models/audio_branch.py`

```python
import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    """CNN cho MFCC features"""

    def __init__(self, n_mfcc=40, num_classes=15, embedding_dim=256):
        super().__init__()

        # CNN layers
        self.conv_layers = nn.Sequential(
            # Conv1: [B, 1, 40, T] -> [B, 64, 40, T/2]
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            # Conv2: [B, 64, 40, T/2] -> [B, 128, 40, T/4]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            # Conv3: [B, 128, 40, T/4] -> [B, 256, 40, T/8]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, mfcc):
        # mfcc: [B, 40, T]
        x = mfcc.unsqueeze(1)  # [B, 1, 40, T]

        x = self.conv_layers(x)
        x = self.global_pool(x)  # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 256]

        logits = self.fc(x)  # [B, num_classes]
        return logits
```

**Training Notebook:** `training/train_audio.ipynb` (Colab)

```python
# Train chỉ với MFCC features
# Target: Baseline accuracy ~50-60%
# Time: ~4-6 giờ với T4 GPU
```

---

#### **Stage 2: Text Branch** (Tuần 6)

**File:** `models/text_branch.py`

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextBranch(nn.Module):
    """BARTpho cho transcripts"""

    def __init__(self, num_classes=15, max_length=256):
        super().__init__()

        # BARTpho model
        self.model_name = "vinai/bartpho-syllable"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert = AutoModel.from_pretrained(self.model_name)

        self.max_length = max_length

        # Classifier
        hidden_size = self.bert.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, transcripts):
        """
        transcripts: list of strings
        """
        # Tokenize
        encoded = self.tokenizer(
            transcripts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoded['input_ids'].to(self.bert.device)
        attention_mask = encoded['attention_mask'].to(self.bert.device)

        # Get embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        # Classify
        logits = self.classifier(cls_embedding)
        return logits
```

**Training Notebook:** `training/train_text.ipynb` (Colab)

---

#### **Stage 3: Video Branch** (Tuần 7-8)

**File:** `models/video_branch.py`

```python
import torch
import torch.nn as nn
from transformers import VivitModel, VivitConfig

class VideoBranch(nn.Module):
    """ViViT cho video frames"""

    def __init__(self, num_classes=15, num_frames=16, pretrained=True):
        super().__init__()

        if pretrained:
            # Load pretrained ViViT
            self.vivit = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        else:
            config = VivitConfig(num_frames=num_frames)
            self.vivit = VivitModel(config)

        # Classifier
        hidden_size = self.vivit.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, frames):
        """
        frames: [B, N, 3, H, W] where N=num_frames
        """
        # Reshape for ViViT: [B, N, C, H, W] -> [B, C, N, H, W]
        frames = frames.permute(0, 2, 1, 3, 4)

        # Get features
        outputs = self.vivit(frames)
        features = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]

        # Classify
        logits = self.classifier(features)
        return logits
```

**Training Notebook:** `training/train_video.ipynb` (Kaggle - có GPU tốt hơn)

**⚠️ Memory optimization:**

```python
# Gradient checkpointing
model.vivit.gradient_checkpointing_enable()

# Mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Small batch size
batch_size = 4  # Thay vì 16
```

---

#### **Stage 4: Multimodal Fusion** (Tuần 9)

**File:** `models/multimodal.py`

```python
import torch
import torch.nn as nn
from models.audio_branch import AudioCNN
from models.text_branch import TextBranch
from models.video_branch import VideoBranch

class MultimodalHarmfulDetector(nn.Module):
    """
    Combined multimodal model
    """

    def __init__(self, num_classes=15, fusion_type='concat'):
        super().__init__()

        # Three branches
        self.audio_branch = AudioCNN(num_classes=num_classes, embedding_dim=256)
        self.text_branch = TextBranch(num_classes=num_classes)
        self.video_branch = VideoBranch(num_classes=num_classes)

        # Remove final classifiers (keep feature extractors)
        self.audio_branch.fc = nn.Identity()
        self.text_branch.classifier = nn.Identity()
        self.video_branch.classifier = nn.Identity()

        # Fusion
        self.fusion_type = fusion_type
        if fusion_type == 'concat':
            # Concatenate features
            self.fusion = nn.Sequential(
                nn.Linear(256 + 768 + 768, 512),  # audio + text + video
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

    def forward(self, mfcc, transcripts, frames):
        # Extract features from each branch
        audio_features = self.audio_branch(mfcc)       # [B, 256]
        text_features = self.text_branch(transcripts)  # [B, 768]
        video_features = self.video_branch(frames)     # [B, 768]

        # Concatenate
        combined = torch.cat([audio_features, text_features, video_features], dim=1)

        # Final classification
        logits = self.fusion(combined)
        return logits
```

**Training Notebook:** `training/train_multimodal.ipynb` (Kaggle)

**Strategy:**

```python
# Load pretrained weights từ các branch
audio_state = torch.load('checkpoints/audio_best.pth')
text_state = torch.load('checkpoints/text_best.pth')
video_state = torch.load('checkpoints/video_best.pth')

# Freeze branches, chỉ train fusion layer
for param in model.audio_branch.parameters():
    param.requires_grad = False
for param in model.text_branch.parameters():
    param.requires_grad = False
# ViViT vẫn fine-tune
# for param in model.video_branch.parameters():
#     param.requires_grad = False

# Train fusion + video branch
```

---

**PHASE 4 Summary:**

✅ Audio branch trained (~55% accuracy)
✅ Text branch trained (~60% accuracy)  
✅ Video branch trained (~65% accuracy)
✅ Multimodal model trained (~75-80% accuracy)

**Platform:** ☁️ Colab + Kaggle
**Chi phí:** 🆓 Free (Colab Free + Kaggle Free)

---

### **PHASE 5: EVALUATION & REFINEMENT** (Tuần 10)

#### **Bước 5.1: Comprehensive Evaluation**

**Notebook:** `evaluation/evaluate.ipynb`

```python
from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score
)

# Load model
model = MultimodalHarmfulDetector(num_classes=15)
model.load_state_dict(torch.load('checkpoints/multimodal_best.pth'))
model.eval()

# Evaluate on test set
test_dataset = HarmfulVideoDataset(metadata_path='metadata_test_enhanced.csv')
test_loader = DataLoader(test_dataset, batch_size=16)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        mfcc = batch['mfcc'].to(device)
        transcripts = batch['transcript']
        frames = batch['frames'].to(device)
        labels = batch['labels']

        logits = model(mfcc, transcripts, frames)
        preds = (torch.sigmoid(logits) > 0.5).float()

        all_preds.append(preds.cpu())
        all_labels.append(labels)

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

# Metrics
print("=== Multilabel Metrics ===")
print(f"Hamming Loss: {hamming_loss(all_labels, all_preds):.4f}")
print(f"Jaccard Score: {jaccard_score(all_labels, all_preds, average='samples'):.4f}")
print(f"F1 Score (micro): {f1_score(all_labels, all_preds, average='micro'):.4f}")
print(f"F1 Score (macro): {f1_score(all_labels, all_preds, average='macro'):.4f}")

# Per-class metrics
print("\n=== Per-class Report ===")
print(classification_report(all_labels, all_preds, target_names=test_dataset.all_labels))

# Confusion matrices
conf_matrices = multilabel_confusion_matrix(all_labels, all_preds)
```

#### **Bước 5.2: Error Analysis**

```python
# Analyze errors
errors = []
for i in range(len(all_labels)):
    if not np.array_equal(all_labels[i], all_preds[i]):
        errors.append({
            'index': i,
            'true_labels': all_labels[i],
            'pred_labels': all_preds[i]
        })

print(f"Total errors: {len(errors)}")

# Most confused pairs
from collections import Counter
confused_pairs = []
for error in errors:
    true_set = set(np.where(error['true_labels'])[0])
    pred_set = set(np.where(error['pred_labels'])[0])

    # False positives
    fp = pred_set - true_set
    # False negatives
    fn = true_set - pred_set

    for label_idx in fp:
        confused_pairs.append(('FP', test_dataset.all_labels[label_idx]))
    for label_idx in fn:
        confused_pairs.append(('FN', test_dataset.all_labels[label_idx]))

print("\n=== Most Common Errors ===")
for (error_type, label), count in Counter(confused_pairs).most_common(10):
    print(f"{error_type} {label}: {count} times")
```

#### **Bước 5.3: Visualization**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix heatmap
for i, label in enumerate(test_dataset.all_labels):
    cm = conf_matrices[i]
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {label}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

# F1 scores per class
f1_per_class = f1_score(all_labels, all_preds, average=None)
plt.figure(figsize=(12, 6))
plt.bar(range(len(test_dataset.all_labels)), f1_per_class)
plt.xticks(range(len(test_dataset.all_labels)), test_dataset.all_labels, rotation=45, ha='right')
plt.ylabel('F1 Score')
plt.title('F1 Score per Class')
plt.tight_layout()
plt.show()
```

---

## 📊 EXPECTED RESULTS

### **Baseline Performance (Single Branch):**

- Audio only: ~55% F1
- Text only: ~60% F1
- Video only: ~65% F1

### **Multimodal Performance:**

- Combined model: ~75-80% F1
- Best classes: normal_content, educational_history
- Challenging classes: ai_slop, game_risk_impact (ít samples)

---

## 💰 COST BREAKDOWN

| Resource              | Cost      | Usage                               |
| --------------------- | --------- | ----------------------------------- |
| **Local compute**     | 🆓 Free   | Pre-processing, dataset, evaluation |
| **Google Colab Free** | 🆓 Free   | Audio branch, text branch, ASR      |
| **Kaggle Free**       | 🆓 Free   | Video branch, multimodal training   |
| **Storage**           | 🆓 Free   | ~20GB local storage                 |
| **TOTAL**             | 🆓 **$0** | 100% miễn phí                       |

**Optional (nếu gấp):**

- Colab Pro: $10/tháng → Faster training, longer sessions
- Kaggle GPUs: Free 30h/week (đủ dùng)

---

## ⏰ TIME ESTIMATE

| Phase                | Duration    | Effort      |
| -------------------- | ----------- | ----------- |
| Setup & Exploration  | 1 tuần      | 5-10h       |
| Pre-processing       | 2 tuần      | 10-15h      |
| Dataset & DataLoader | 1 tuần      | 5h          |
| Audio Branch         | 1 tuần      | 10h         |
| Text Branch          | 1 tuần      | 10h         |
| Video Branch         | 2 tuần      | 20h         |
| Multimodal Fusion    | 1 tuần      | 15h         |
| Evaluation           | 1 tuần      | 5h          |
| **TOTAL**            | **10 tuần** | **80-100h** |

---

## 🎯 SUCCESS CRITERIA

✅ **Minimum (Pass đồ án):**

- Dataset preprocessed hoàn chỉnh
- 3 branches hoạt động độc lập
- Multimodal model đạt ~70% F1
- Report + code đầy đủ

✅ **Target (Điểm tốt):**

- Multimodal model đạt ~75-80% F1
- Thorough evaluation & analysis
- Clean code, good documentation

✅ **Excellent (Điểm cao):**

- Model đạt >80% F1
- Novel fusion strategies
- Comprehensive error analysis
- Publication-ready results

---

## 🚨 RISK MITIGATION

### **Risk 1: Colab timeout**

**Solution:**

- Save checkpoints mỗi 10-15 phút
- Upload lên Google Drive tự động
- Resume từ checkpoint

### **Risk 2: Out of memory**

**Solution:**

- Gradient checkpointing
- Mixed precision training
- Smaller batch size (4 instead of 16)
- Process smaller video resolution

### **Risk 3: Poor single-branch performance**

**Solution:**

- More data augmentation
- Better hyperparameters
- Longer training
- Pretrained models

### **Risk 4: Deadline pressure**

**Solution:**

- Follow incremental approach
- Each stage produces usable results
- Can submit partial work if needed

---

## 📚 REFERENCES

### **Models:**

- wav2vec2: https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h
- BARTpho: https://huggingface.co/vinai/bartpho-syllable
- ViViT: https://huggingface.co/google/vivit-b-16x2-kinetics400

### **Papers:**

- ViViT: "Vivit: A video vision transformer" (ICCV 2021)
- wav2vec2: "wav2vec 2.0: A framework for self-supervised learning" (NeurIPS 2020)
- BARTpho: "BARTpho: Pre-trained Sequence-to-Sequence Models for Vietnamese"

### **Libraries:**

- PyTorch: https://pytorch.org/
- Transformers: https://huggingface.co/transformers/
- Librosa: https://librosa.org/
- OpenCV: https://opencv.org/

---

## ✅ NEXT STEPS

**Bước đầu tiên:**

1. Setup môi trường (install dependencies)
2. Chạy data exploration
3. Test preprocessing với 10 videos
4. Verify pipeline hoạt động

**Sau đó:** 5. Scale up preprocessing (full dataset) 6. Implement Dataset class 7. Train branches incrementally 8. Evaluate and refine

---

**Good luck với đồ án! 🚀**

_Document version: 1.0_
_Last updated: January 30, 2026_
