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

def center_crop_resize(frame, target_size = 224, aspect_thresh=1.3):
  h, w = frame.shape[:2]

  # Check video is vertical 
  if h / w  >= aspect_thresh:
    scale = target_size / h
    new_h = target_size
    new_w = int(w * scale)

    frame = cv2.resize(frame, (new_w, new_h),
                       interpolation=cv2.INTER_LINEAR)
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    frame = cv2.copyMakeBorder(
      frame,
      top=0,
      bottom=0,
      left=pad_left,
      right=pad_right,
      borderType=cv2.BORDER_CONSTANT,
      value=[114, 114, 114]
    )
    return frame
  else:
    resize_size = 256
    scale = resize_size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    frame = cv2.resize(frame, (new_w, new_h),
                       interpolation=cv2.INTER_LINEAR)
    start_y = (new_h - target_size) // 2
    start_x = (new_w - target_size) // 2
    return frame[start_y:start_y + target_size,
                 start_x:start_x + target_size]
  


def extract_frames_uniform(video_path, output_dir, num_frames=16, target_size=224):
  os.makedirs(output_dir, exist_ok=True)
  try:
    # Initialize Decord VideoReader
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames= len(vr)

    if total_frames == 0:
      logger.warning(f'Video {video_path} has no frames.')
      return {'success': False, 'extracted': 0, 'total_frames': 0, 'looped': False}
    
    # Handle short videos by looping frames: if total_frames < num_frames
    looped = False
    if total_frames < num_frames:
      logger.info(f"Short video ({total_frames} frames < {num_frames} frames). Looping frames.")
      base_indices = np.arange(total_frames)
      repeats = (num_frames // total_frames) + 1
      frame_indices = np.tile(base_indices, repeats)[:num_frames]
      looped = True
    else:
      # Uniform sampling: ViViT standard
      frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Batch load fames
    frames_batch = vr.get_batch(frame_indices).asnumpy() # [N, H, W, C] RGB format

    # Process and save frames
    extracted = 0
    for i, frame in enumerate(frames_batch):
      try:
        # Validate frame
        if frame.size == 0:
          logger.warning(f'Empty frame at index {frame_indices[i]} in video {video_path}. Skipping.')
          continue
        
        frame_processed = center_crop_resize(frame, target_size=target_size)

        # Verify processed frame size
        if frame_processed.shape != (target_size, target_size, 3):
          logger.warning(f'Processed frame at index {frame_indices[i]} has invalid shape {frame_processed.shape} in video {video_path}. Skipping.')
          continue

        #  Save frame
        frame_path = os.path.join(output_dir, f'frame_{extracted:03d}.jpg')
        cv2.imwrite(frame_path, cv2.cvtColor(frame_processed, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        extracted += 1
      except Exception as e:
        logger.error(f'Error processing frame at index {frame_indices[i]} in video {video_path}: {e}')
        continue
    
    # Success
    success = extracted >= num_frames*0.8

    if not success:
      logger.error(f"Extraction failed: {video_path} - Only {extracted}/{num_frames} frames")
    return {
      'success': success,
      'extracted': extracted,
      'total_frames': total_frames,
      'looped': looped
    }
  except Exception as e:
    logger.error(f'Decord error for {video_path}: {e}')
    return {'success': False, 'extracted': 0, 'total_frames': 0, 'looped': False}

# Process all videos
NUM_FRAMES = 16
TARGET_SIZE = 224

if __name__ == "__main__":
  # Get workspace root (parent of pre-processing directory)
  script_dir = os.path.dirname(os.path.abspath(__file__))
  workspace_root = os.path.dirname(script_dir)
  
  stats = {
    'total': 0,
    'success': 0,
    'failed': 0,
    'looped': 0,
    'skipped': 0
  }

  failed_videos = []

  # Read metadata to get video paths
  for split in ['train', 'val', 'test']:
    metadata_path = os.path.join(workspace_root, 'hf_dataset_merged', f'metadata_{split}.csv')
    
    if not os.path.exists(metadata_path):
      logger.warning(f'Metadata file {metadata_path} does not exist. Skipping {split} split.')
      continue
    
    # Read metadata to get video list
    import pandas as pd
    df = pd.read_csv(metadata_path)
    
    frames_dir = os.path.join(workspace_root, 'hf_dataset_merged', 'data', 'frames', split)
    os.makedirs(frames_dir, exist_ok=True)
    
    logger.info(f'Processing {len(df)} videos in {split} split.')

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f'Extracting frames for {split}'):
      # Get video path from metadata (already includes 'data/' prefix)
      video_path = os.path.join(workspace_root, 'hf_dataset_merged', row['file_name'])
      video_id = os.path.splitext(os.path.basename(row['file_name']))[0]
      output_dir = os.path.join(frames_dir, video_id)

      stats['total'] += 1

      # Skip if already extracted
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

  # Print final summary after all splits
  logger.info(f"\n{'='*60}")
  logger.info("EXTRACTION SUMMARY")
  logger.info(f"{'='*60}")
  logger.info(f"Total videos: {stats['total']}")
  logger.info(f"✅ Success: {stats['success']} ({stats['success']/max(stats['total']-stats['skipped'],1)*100:.1f}%)")
  logger.info(f"⏭️  Skipped: {stats['skipped']}")
  logger.info(f"🔄 Looped (short videos): {stats['looped']}")
  logger.info(f"❌ Failed: {stats['failed']}")
  
  # Save failed videos list
  if failed_videos:
    failed_log_path = os.path.join(workspace_root, 'failed_videos.txt')
    with open(failed_log_path, 'w', encoding='utf-8') as f:
      for fail in failed_videos:
        f.write(f"{fail['path']} - extracted {fail['extracted']}/{NUM_FRAMES} frames (total {fail['total_frames']})\n")
    logger.info(f"Failed videos log saved to: {failed_log_path}")