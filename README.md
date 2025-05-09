# Person Tracking with Re-Identification

This project implements a person tracking system using TorchVision's Faster R-CNN model, ByteTrack for tracking, and a custom re-identification module based on clothing appearance and height.

## Features

- Person detection using Faster R-CNN
- Tracking with ByteTrack
- Person re-identification based on:
  - Clothing appearance (color histograms of upper, middle, and lower body)
  - Person height
- Persistent IDs across video frames, even when people exit and re-enter the scene

## Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the person tracking script:

```bash
python person_tracking.py
```

This will use your webcam by default and display bounding boxes with persistent person IDs.

### Advanced Usage (Demo Script)

For more control, use the demo script:

```bash
python demo.py --input path/to/video.mp4 --output output.mp4
```

The demo script accepts several command-line arguments:

- `--input`, `-i`: Path to video file or camera index (default: 0 for webcam)
- `--output`, `-o`: Path to output video file (default: None)
- `--height-weight`: Weight for height feature (default: 0.3)
- `--appearance-weight`: Weight for appearance feature (default: 0.7)
- `--similarity-threshold`: Minimum similarity threshold (default: 0.6)
- `--save-features`: Path to save person features (default: person_features.pkl)
- `--confidence`: Confidence threshold for detections (default: 0.5)

Example:

```bash
# Process a video file with custom parameters
python demo.py --input videos/people.mp4 --output results.mp4 --similarity-threshold 0.7

# Use webcam with custom parameters
python demo.py --height-weight 0.4 --appearance-weight 0.6 --confidence 0.7
```

## Customization

You can adjust these parameters in the code:

- `height_weight`: How much to weigh height in similarity calculation (default: 0.3)
- `appearance_weight`: How much to weigh appearance in similarity calculation (default: 0.7)
- `similarity_threshold`: Minimum similarity to consider persons as the same (default: 0.6)

## How it Works

1. Faster R-CNN detects persons in each frame
2. ByteTrack assigns temporary tracking IDs
3. Our PersonReID class:
   - Extracts appearance features using color histograms
   - Calculates person height
   - Matches persons who exit and re-enter the frame
   - Maintains persistent IDs

## First Run

On the first run, the model file (`yolov8x.pt`) will be automatically downloaded if not present.

## Potential Improvements

For even better re-identification, consider implementing:

1. Deep learning-based appearance features (e.g., using torchreid)
2. Body part models for more robust matching
3. Temporal information for better tracking #   u n i q u e - p e o p l e - t r a c k e r  
 