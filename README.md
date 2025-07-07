# Face Trimming Test

A Streamlit application for face alignment and color correction. Uses MediaPipe to detect facial landmarks and performs facial image alignment, standardization, and color correction.

## Features

- **Face Alignment**: Aligns multiple face images to unified size and angle
- **Color Correction**: Adjustable color and correction intensity for each facial part (lips, eyes, eyebrows, face)
- **Real-time Preview**: View results in real-time while adjusting settings in the UI
- **Batch Processing**: Process multiple images at once
- **Download Functionality**: Download processed images

## Setup

### Requirements

- Python 3.7+
- pip

### Installation

1. Clone the repository
```bash
git clone [<repository-url>](https://github.com/milk0707/Face-Trimming-Test.git)
cd Face-Trimming-Test
```

2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

3. Install required packages
```bash
pip install streamlit opencv-python numpy mediapipe pillow
```

## Usage

1. Start the application
```bash
streamlit run app.py
```

2. Access http://localhost:8501 in your browser

3. Configure color and correction intensity for each part in the sidebar

4. Upload images (multiple selection supported)

5. Click the "Redraw" button to execute processing

6. Download processed images

## Parameter Settings

You can adjust the following parameters for each facial part:

- **Lips**: Color and correction intensity
- **Eyes**: Color and correction intensity  
- **Eyebrows**: Color and correction intensity
- **Face**: Color and correction intensity

## File Structure

- `app.py`: Main application
- `resize-app.py`: Resize application
- `config/make_up.yml`: Makeup configuration file
- `backup/`: Backup files

## Processing Flow

1. Detect facial landmarks using MediaPipe
2. Align faces based on eye positions (rotation, scaling, cropping)
3. Create masks for each part and apply color correction
4. Output unified 600x800px images

## Troubleshooting

- **Face not detected**: Ensure the image is bright and the face is facing forward
- **Slow processing**: Reduce image size or number of images being processed
- **Excessive color correction**: Lower the correction intensity in the sidebar

## Technical Specifications

- **Supported image formats**: JPG, JPEG, PNG
- **Output size**: 600x800px (fixed)
- **Eye position**: Placed at 40% of image height
- **Libraries used**: 
  - Streamlit (UI)
  - OpenCV (Image processing)
  - MediaPipe (Face detection)
  - PIL (Image conversion)
  - NumPy (Numerical computation)

## License

This project is created for personal use and educational purposes. 