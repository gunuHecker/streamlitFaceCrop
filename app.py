import streamlit as st
import os
import io
import cv2
import json
import numpy as np
import mediapipe as mp
from face_crop_plus import Cropper
from torch.cuda import is_available
from os.path import join, dirname, abspath
from tempfile import TemporaryDirectory
import shutil
from PIL import Image
from rembg import remove, new_session
from tempfile import TemporaryDirectory
import zipfile
import io
import atexit

# Set page config
st.set_page_config(page_title="Face Crop Tool", layout="wide")

# Create temporary directories for input and output
temp_input_dir = TemporaryDirectory()
temp_output_dir = TemporaryDirectory()

# Clean up function to be called at the end
def cleanup():
    try:
        temp_input_dir.cleanup()
        temp_output_dir.cleanup()
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Register cleanup function
atexit.register(cleanup)

# Cropper configuration
TEST_QUALITY_ENHANCEMENT = False  # Set to False if running on CPU
enh_threshold = None
if TEST_QUALITY_ENHANCEMENT:
    enh_threshold = 0.001  # Low threshold for enhancement

# Disable file watcher to prevent torch classes error
os.environ['STREAMLIT_FILE_WATCHER'] = 'false'

mp_face_mesh = mp.solutions.face_mesh

# Uses MediaPipe for face detection and returns landmarks
def get_landmarks_mediapipe(img, face_mesh=mp_face_mesh):
    with face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.multi_face_landmarks[0].landmark
            ])
            return landmarks
    return None

def multi_scale_detection(image):
    scales = [1.0, 1.5, 2.0, 2.5]
    for scale in scales:
        if scale != 1.0:
            h, w = image.shape[:2]
            scaled_image = cv2.resize(image, (int(w * scale), int(h * scale)))
        else:
            scaled_image = image.copy()
        landmarks = get_landmarks_mediapipe(scaled_image)
        if landmarks:
            if scale != 1.0:
                for landmark in landmarks:
                    landmark[0] /= scale
                    landmark[1] /= scale
            return landmarks
    return None

def opencv_face_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        return faces
    return None

def extract_face_region(image, face_bbox):
    x, y, w, h = face_bbox
    expansion = 0.5
    new_w = int(w * (1 + expansion))
    new_h = int(h * (1 + expansion))
    new_x = max(0, x - int(expansion * w / 2))
    new_y = max(0, y - int(expansion * h / 2))
    face_region = image[new_y:new_y + new_h, new_x:new_x + new_w]
    resized_face = cv2.resize(face_region, (512, 512))
    return resized_face, (new_x, new_y, new_w, new_h)

def adjust_landmarks_to_original(landmarks, bbox, orig_shape):
    x, y, w, h = bbox
    scale_x = w / 512
    scale_y = h / 512
    # Convert landmarks to a list of dicts for extract_face_metrics
    landmark_dicts = []
    for landmark in landmarks:
        landmark_dicts.append({
            'x': (landmark[0] * 512 * scale_x + x) / orig_shape[1],
            'y': (landmark[1] * 512 * scale_y + y) / orig_shape[0]
        })
    # Create a dummy object with .landmark attribute for compatibility
    class DummyLandmarks:
        pass
    dummy = DummyLandmarks()
    dummy.landmark = [type('obj', (object,), d) for d in landmark_dicts]
    return dummy

# # Improve later
def robust_face_detection_pipeline(image):
    # Step 1: Try Mediaface
    landmarks = get_landmarks_mediapipe(image)

    if landmarks is not None:
        return landmarks, "mediapipe"

    # Step 2: Milti-scale mediapipe
    landmarks = multi_scale_detection(image)
    if landmarks is not None:
        return landmarks, "mediapipe_enhanced"

    # Step 3: OpenCV Haar Cascade fallback
    opencv_faces = opencv_face_detection(image)
    if opencv_faces is not None:
        face_region, bbox = extract_face_region(image, opencv_faces[0])
        landmarks = get_landmarks_mediapipe(face_region)
        if landmarks is not None:
            # Adjust landmarks to original image coordinates
            x, y, w, h = bbox
            scale_x = w / 512
            scale_y = h / 512
            landmark_dicts = []
            for landmark in landmarks:
                landmark_dicts.append({
                    'x': (landmark[0] * 512 * scale_x + x) / image.shape[1],
                    'y': (landmark[1] * 512 * scale_y + y) / image.shape[0]
                })
            class DummyLandmarks:
                pass
            dummy = DummyLandmarks()
            dummy.landmark = [type('obj', (object,), d) for d in landmark_dicts]
            return dummy, "opencv_region"
    
    # Improve later
    # (Optional) Add RetinaFace/YOLO-Face fallback here
    print("Failed to detect face")
    return None, "failed"

# Main function to run to get landmarks of face and the method used to get landmarks
def get_landmarks(image):
    landmarks, method = robust_face_detection_pipeline(image)
    if landmarks is not None:
        if method == "opencv_region":
            # OpenCV landmarks have only x,y coordinates
            landmarks = np.array([
                [lm.x, lm.y, 0.0]  # Add 0.0 for z coordinate
                for lm in landmarks.landmark
            ])
        elif hasattr(landmarks, 'landmark'):
            # MediaPipe landmarks have x,y,z coordinates
            landmarks = np.array([
                [lm.x, lm.y, lm.z]
                for lm in landmarks.landmark
            ])
        else:
            # Already numpy array format
            landmarks = landmarks
        return landmarks, method
    return None, method

def get_landmark_slices_478(num_landmarks):
    # Define the slices for MediaPipe landmarks
    left_eye_slice = np.arange(145, 159)
    right_eye_slice = np.arange(374, 386)
    nose_slice = np.arange(312, 318)
    left_mouth_slice = np.arange(78, 85)
    right_mouth_slice = np.arange(291, 296)
    
    return left_eye_slice, right_eye_slice, nose_slice, left_mouth_slice, right_mouth_slice

def process_landmarks(images):
    """
    Convert MediaPipe landmarks to RetinaFace's 5-point format
    """
    landmarks_list = []
    filenames_list = []
    methods_list = []
    
    for filename, img, landmarks, method, _ in images:
        if landmarks is not None:
            # Convert MediaPipe landmarks to RetinaFace's 5-point format
            h, w = img.shape[:2]
            
            # Extract coordinates (normalized)
            right_eye_outer = landmarks[33, :2]  # Right eye outer corner
            right_eye_inner = landmarks[133, :2]   # Right eye inner corner
            left_eye_outer = landmarks[263, :2]   # Left eye outer corner
            left_eye_inner = landmarks[362, :2]   # Left eye inner corner
            nose_tip = landmarks[1, :2]           # Nose tip
            right_mouth = landmarks[291, :2]     # Right mouth corner
            left_mouth = landmarks[61, :2]       # Left mouth corner
            
            # Calculate eye centers by averaging outer and inner corners
            right_eye_center = (right_eye_outer + right_eye_inner) / 2
            left_eye_center = (left_eye_outer + left_eye_inner) / 2
            
            # Assemble RetinaFace format landmarks (normalized coordinates)
            retinaface_landmarks = np.array([
                right_eye_center,  # Right eye center
                left_eye_center,   # Left eye center
                nose_tip,          # Nose tip
                right_mouth,       # Right mouth corner
                left_mouth         # Left mouth corner
            ])
            
            # Convert to pixel coordinates
            landmarks_pixel = retinaface_landmarks * np.array([w, h])
            landmarks_list.append(landmarks_pixel)
            filenames_list.append(filename)
            methods_list.append(method)
    
    if landmarks_list:
        landmarks_array = np.stack(landmarks_list).astype(np.float32)
        filenames_array = np.array(filenames_list, dtype=np.str_)
        methods_array = np.array(methods_list, dtype=np.str_)
        return landmarks_array, filenames_array, methods_array
    return None, None, None

def prepare_for_cropper(landmarks_array, filenames_array):
    """
    Prepare landmarks for face-crop-plus cropper
    """
    landmarks_for_cropper = (landmarks_array, filenames_array)
    return landmarks_for_cropper

def remove_image_background(image: Image.Image, model_name: str = "birefnet-portrait", background_style: str = 'Transparent', output_format: str = 'PNG') -> Image.Image:
    """
    Remove background from image and apply specified background style
    
    Args:
        image: PIL Image object
        model_name: rembg model to use for background removal
        background_style: 'Transparent', 'White', 'Black', 'Light Gray', or 'Blue'
        output_format: 'JPEG' or 'PNG'
    
    Returns:
        PIL Image object with background removed/replaced
    """
    # Create session with specified model
    session = new_session(model_name)

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_data = img_byte_arr.getvalue()

    # Use specified model to remove background
    output_data = remove(img_data, session=session)
    output_image = Image.open(io.BytesIO(output_data)).convert("RGBA")
    
    # Handle background style
    if background_style == "Transparent" and output_format == "PNG":
        # Keep transparent background for PNG
        return output_image
    else:
        # Create a colored background
        background_colors = {
            'White': (255, 255, 255),
            'Black': (0, 0, 0),
            'Light Gray': (200, 200, 200),
            'Blue': (135, 206, 235)  # Sky blue
        }
        
        bg_color = background_colors.get(background_style, (255, 255, 255))
        
        # Create background image
        new_background = Image.new('RGB', output_image.size, bg_color)
        
        # Composite the result onto the background
        if output_image.mode == 'RGBA':
            new_background.paste(output_image, mask=output_image.split()[3])
        else:
            new_background.paste(output_image)
        
        return new_background

def get_output_size(aspect_ratio):
    """Convert aspect ratio to output size tuple"""
    aspect_ratios = {
        "4:3": (600, 800),   # Portrait (3:4 rotated)
        "3:4": (800, 600),   # Landscape
        "1:1": (600, 600)    # Square
    }
    return aspect_ratios.get(aspect_ratio, (600, 800))

def process_cropped_image(pil_img, remove_bg, output_format, bg_model="birefnet-portrait", background_style="Transparent"):
    """Process the cropped image based on user settings"""
    if remove_bg:
        final_img = remove_image_background(pil_img, bg_model, background_style, output_format)
    else:
        final_img = pil_img.copy()
    
    # Convert to RGB if saving as JPEG and image is still RGBA
    if output_format.lower() == "jpeg" and final_img.mode == "RGBA":
        rgb_img = Image.new('RGB', final_img.size, (255, 255, 255))
        rgb_img.paste(final_img, mask=final_img.split()[3] if final_img.mode == 'RGBA' else None)
        final_img = rgb_img
    
    return final_img

# Streamlit App
st.title("🎭 Face Crop Tool")
st.markdown("Upload images and automatically crop faces with customizable settings.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Face Size Slider
face_size = st.sidebar.slider(
    "Face Size", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.9, 
    step=0.1,
    help="Ratio of the face to the entire image"
)

# Image Aspect Ratio
aspect_ratio = st.sidebar.selectbox(
    "Image Aspect Ratio",
    options=["4:3", "3:4", "1:1"],
    index=0,
    help="Choose the aspect ratio for output images"
)

# Remove Background Toggle
remove_background = st.sidebar.toggle(
    "Remove Background",
    value=True,
    help="Remove background and replace with white"
)

# Only show background-related options when background removal is enabled
if remove_background:
    # Background Removal Model
    background_model = st.sidebar.selectbox(
        "Background Removal Model",
        options=[
            "birefnet-portrait",
            "birefnet-general-lite",
            "isnet-general-use", 
            "u2net_human_seg", 
            "u2net", 
            "silueta"
        ],
        index=0,
        help="Choose the AI model for background removal"
    )

    # Output Format
    output_format = st.sidebar.selectbox(
        "Output Format",
        options=["PNG", "JPEG"],
        index=0,
        help="Choose the output image format"
    )

    # Background Style (based on output format)
    if output_format == "PNG":
        # For PNG, offer transparent or colored background
        background_style = st.sidebar.selectbox(
            "Background Style",
            options=["Transparent", "White", "Black", "Light Gray", "Blue"],
            index=0,
            help="Choose background style for PNG output"
        )
    else:
        # For JPEG, only offer colored backgrounds (no transparency)
        background_style = st.sidebar.selectbox(
            "Background Color",
            options=["White", "Black", "Light Gray", "Blue"],
            index=0,
            help="Choose background color for JPEG output"
        )
else:
    # Set default values when background removal is disabled
    background_model = "birefnet-portrait"
    output_format = "JPEG"
    background_style = None

# File uploader
uploaded_files = st.file_uploader(
    "Choose images to process", 
    type=["jpg", "jpeg", "png", "avif"], 
    accept_multiple_files=True
)

# Initialize session state for processed images
if 'processed_images_list' not in st.session_state:
    st.session_state.processed_images_list = []

if 'crop_completed' not in st.session_state:
    st.session_state.crop_completed = False

# Create a dictionary to store processed images
processed_images = {}

if uploaded_files:
    st.write("### Original Images")
    
    # Process and display original images first
    processed_images = {}
    
    # Display original images
    cols = st.columns(3)  # Show 3 images per row
    for idx, uploaded_file in enumerate(uploaded_files):
        with cols[idx % 3]:
            # Read file once and store in memory
            file_bytes = uploaded_file.read()
            if not file_bytes:
                st.warning(f"{uploaded_file.name}: Empty file")
                continue
                
            # Store in processed_images dict
            processed_images[uploaded_file.name] = file_bytes
            
            try:
                # Try OpenCV first
                img_cv = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img_cv is None:
                    # Try PIL fallback
                    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                # Display original image
                st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), 
                        caption=f"Original: {uploaded_file.name}", 
                        use_container_width=True)
                
            except Exception as e:
                st.warning(f"{uploaded_file.name}: Could not decode image. {e}")
                continue

# Sidebar buttons
st.sidebar.markdown("---")
generate_crops = st.sidebar.button("🎨 Generate Crops", type="primary", use_container_width=True)

if generate_crops and uploaded_files:
    with st.spinner("Processing images..."):
        # Initialize progress tracking
        total_files = len(uploaded_files)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process all images and collect landmarks
        images = []
        detection_stats = {
            'mediapipe': [],
            'mediapipe_enhanced': [],
            'opencv_region': [],
            'failed': [],
            'decode_errors': []
        }
        
        for idx, uploaded_file in enumerate(uploaded_files):
            # Update progress bar for face detection phase
            progress = (idx + 1) / (total_files * 2)  # Face detection takes first half
            progress_bar.progress(progress)
            status_text.text(f"🔍 Detecting faces... ({idx + 1}/{total_files})")
            
            file_bytes = processed_images[uploaded_file.name]
            
            try:
                # Try OpenCV first
                img_cv = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                if img_cv is None:
                    # Try PIL fallback
                    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                # Process the image
                landmarks, method = get_landmarks(img_cv)
                
                if landmarks is not None:
                    # Save to temporary directory
                    file_path = join(temp_input_dir.name, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(file_bytes)
                    
                    # Add to images list and track detection method
                    images.append((uploaded_file.name, img_cv, landmarks, method, file_bytes))
                    detection_stats[method].append(uploaded_file.name)
                else:
                    detection_stats['failed'].append(uploaded_file.name)
                    
            except Exception as e:
                detection_stats['decode_errors'].append(f"{uploaded_file.name}: {str(e)}")
                continue

        # Process all landmarks at once
        if images:
            # Update progress for landmark processing
            progress_bar.progress(0.5)
            status_text.text("🎯 Processing landmarks...")
            
            landmarks_array, filenames_array, methods_array = process_landmarks(images)
            
            if landmarks_array is not None:
                # Prepare landmarks for cropper
                landmarks_for_cropper = prepare_for_cropper(landmarks_array, filenames_array)
                
                # Get output size based on aspect ratio
                output_size = get_output_size(aspect_ratio)
                
                # Create cropper instance with user settings
                cropper = Cropper(
                    output_size=output_size,
                    output_format=output_format.lower(),
                    face_factor=face_size,
                    padding="replicate",
                    strategy="all",
                    landmarks=landmarks_for_cropper,
                    device="cuda:0" if is_available() else "cpu",
                    batch_size=1,
                    enh_threshold=enh_threshold,
                )
                
                try:
                    # Update progress for cropping phase
                    progress_bar.progress(0.6)
                    status_text.text("✂️ Cropping faces...")
                    
                    # Process all images
                    cropper.process_dir(temp_input_dir.name, temp_output_dir.name)
                    
                    # Process and store cropped images
                    st.session_state.processed_images_list = []
                    
                    # Update progress for post-processing
                    progress_bar.progress(0.7)
                    status_text.text("🎨 Post-processing images...")
                    
                    st.write("### Processed Images")
                    
                    # Display cropped images
                    cols = st.columns(3)  # Show 3 images per row
                    col_idx = 0
                    total_processed = len(filenames_array)
                    
                    for process_idx, filename in enumerate(filenames_array):
                        # Update progress for each processed image
                        progress = 0.7 + (0.3 * (process_idx + 1) / total_processed)
                        progress_bar.progress(progress)
                        status_text.text(f"🖼️ Processing image {process_idx + 1}/{total_processed}...")
                        
                        # Try different possible extensions that the cropper might create
                        possible_extensions = [".jpeg", ".jpg", ".png"]
                        cropped_img_path = None
                        cropped_filename = None
                        
                        for ext in possible_extensions:
                            potential_filename = os.path.splitext(filename)[0] + "_0" + ext
                            potential_path = os.path.join(temp_output_dir.name, potential_filename)
                            if os.path.exists(potential_path):
                                cropped_img_path = potential_path
                                cropped_filename = potential_filename
                                break
                        
                        if cropped_img_path and os.path.exists(cropped_img_path):
                            # Read the cropped image and convert to PIL format
                            cropped_img = cv2.imread(cropped_img_path)
                            if cropped_img is not None:
                                # Convert to PIL Image
                                pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                                
                                # Process based on user settings
                                final_img = process_cropped_image(pil_img, remove_background, output_format, background_model, background_style)
                                
                                # Convert back to RGB numpy array for display
                                final_array = np.array(final_img)
                                
                                # Save the processed image with correct extension based on user preference
                                file_extension = ".png" if output_format.upper() == "PNG" else ".jpg"
                                processed_filename = f"processed_{os.path.splitext(filename)[0]}_0{file_extension}"
                                processed_path = os.path.join(temp_output_dir.name, processed_filename)
                                final_img.save(processed_path)
                                
                                # Add to list of processed images for download
                                st.session_state.processed_images_list.append(processed_filename)
                                
                                # Display in columns
                                with cols[col_idx % 3]:
                                    method_used = methods_array[np.where(filenames_array == filename)[0][0]]
                                    st.image(final_array, 
                                            caption=f"Cropped: {filename}\nMethod: {method_used}",
                                            use_container_width=True)
                                
                                col_idx += 1
                            else:
                                st.warning(f"Could not read cropped image: {cropped_filename}")
                        else:
                            # Debug: Show what files actually exist
                            existing_files = os.listdir(temp_output_dir.name)
                            st.warning(f"Could not find cropped image for {filename}. Available files: {existing_files[:3]}...")
                    
                    # Complete progress
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing completed!")
                    
                    st.session_state.crop_completed = True
                    
                    # Display comprehensive processing statistics
                    st.write("## 📊 Processing Summary")
                    
                    # Create columns for statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="✅ Successfully Processed",
                            value=len(st.session_state.processed_images_list),
                            delta=f"out of {total_files} images"
                        )
                    
                    with col2:
                        total_detected = len(detection_stats['mediapipe']) + len(detection_stats['mediapipe_enhanced']) + len(detection_stats['opencv_region'])
                        detection_rate = (total_detected / total_files * 100) if total_files > 0 else 0
                        st.metric(
                            label="� Face Detection Rate",
                            value=f"{detection_rate:.1f}%",
                            delta=f"{total_detected}/{total_files} faces found"
                        )
                    
                    with col3:
                        if remove_background:
                            st.metric(
                                label="🤖 Background Model Used",
                                value=background_model,
                                delta="Applied to all images"
                            )
                        else:
                            st.metric(
                                label="🖼️ Processing Mode",
                                value="Crop Only",
                                delta="No background removal"
                            )
                    
                    # Detailed detection method breakdown
                    if total_detected > 0:
                        st.write("### 🔍 Face Detection Methods Used")
                        
                        method_data = []
                        if detection_stats['mediapipe']:
                            method_data.append({
                                'Method': '🥇 MediaPipe (Standard)',
                                'Count': len(detection_stats['mediapipe']),
                                'Percentage': f"{len(detection_stats['mediapipe'])/total_detected*100:.1f}%",
                                'Description': 'Best quality detection'
                            })
                        
                        if detection_stats['mediapipe_enhanced']:
                            method_data.append({
                                'Method': '⚡ MediaPipe (Enhanced)',
                                'Count': len(detection_stats['mediapipe_enhanced']),
                                'Percentage': f"{len(detection_stats['mediapipe_enhanced'])/total_detected*100:.1f}%",
                                'Description': 'Multi-scale detection for difficult images'
                            })
                        
                        if detection_stats['opencv_region']:
                            method_data.append({
                                'Method': '🔄 OpenCV + MediaPipe',
                                'Count': len(detection_stats['opencv_region']),
                                'Percentage': f"{len(detection_stats['opencv_region'])/total_detected*100:.1f}%",
                                'Description': 'Fallback method for challenging cases'
                            })
                        
                        # Display as a nice table using Streamlit columns
                        for method_info in method_data:
                            col_method, col_count, col_percent, col_desc = st.columns([3, 1, 1, 3])
                            with col_method:
                                st.write(method_info['Method'])
                            with col_count:
                                st.write(f"**{method_info['Count']}**")
                            with col_percent:
                                st.write(method_info['Percentage'])
                            with col_desc:
                                st.write(method_info['Description'])
                    
                    # Show failures if any
                    if detection_stats['failed'] or detection_stats['decode_errors']:
                        st.write("### ⚠️ Processing Issues")
                        
                        if detection_stats['failed']:
                            with st.expander(f"❌ Face Detection Failed ({len(detection_stats['failed'])} images)"):
                                for filename in detection_stats['failed']:
                                    st.text(f"• {filename}")
                                st.info("💡 Try adjusting image quality or ensuring faces are clearly visible")
                        
                        if detection_stats['decode_errors']:
                            with st.expander(f"🔧 File Decode Errors ({len(detection_stats['decode_errors'])} images)"):
                                for error in detection_stats['decode_errors']:
                                    st.text(f"• {error}")
                                st.info("💡 Check file format and ensure images are not corrupted")
                    
                    # Final success message
                    if len(st.session_state.processed_images_list) > 0:
                        st.success(f"�🎉 Successfully processed {len(st.session_state.processed_images_list)} images!")
                    else:
                        st.error("❌ No images were successfully processed. Please check your images and try again.")
                    
                except Exception as e:
                    st.error(f"❌ Error during face processing: {str(e)}")
        else:
            # Show summary even when no faces detected
            st.write("## 📊 Processing Summary")
            st.error("⚠️ No faces were detected in any of the uploaded images.")
            
            # Show detailed breakdown of failures
            total_failed = len(detection_stats['failed'])
            total_errors = len(detection_stats['decode_errors'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("❌ Face Detection Failed", total_failed)
            with col2:
                st.metric("🔧 File Decode Errors", total_errors)
            
            if detection_stats['failed']:
                with st.expander("View failed detections"):
                    for filename in detection_stats['failed']:
                        st.text(f"• {filename}")
            
            if detection_stats['decode_errors']:
                with st.expander("View decode errors"):
                    for error in detection_stats['decode_errors']:
                        st.text(f"• {error}")
            
            st.info("💡 **Tips to improve detection:**\n- Ensure faces are clearly visible\n- Use well-lit images\n- Avoid heavily cropped or rotated faces\n- Check image quality and format")

# Download button in sidebar
if st.session_state.crop_completed and st.session_state.processed_images_list:
    with st.sidebar:
        st.markdown("---")
        
        # Create download data
        with io.BytesIO() as buffer:
            with zipfile.ZipFile(buffer, "w") as zipf:
                for img_name in st.session_state.processed_images_list:
                    img_path = os.path.join(temp_output_dir.name, img_name)
                    if os.path.exists(img_path):
                        zipf.write(img_path, arcname=img_name)
            buffer.seek(0)
            
            st.download_button(
                label="📥 Download Processed Images",
                data=buffer.getvalue(),
                file_name="processed_images.zip",
                mime="application/zip",
                type="secondary",
                use_container_width=True
            )

# Information section    
with st.expander("ℹ️ How to use"):
    st.markdown("""
    ## 🚀 **Quick Start Guide**
    1. **Configure Settings**: Use the sidebar to adjust face size, aspect ratio, and background removal options
    2. **Upload Images**: Choose one or more image files (JPG, JPEG, PNG, AVIF)
    3. **Generate Crops**: Click the "🎨 Generate Crops" button to process images
    4. **Download Results**: Use the "📥 Download Processed Images" button to get your results as a ZIP file
    
    ## ⚙️ **Settings Explanation**
    
    ### **Core Settings (Always Available)**
    - **Face Size**: Controls how much of the face is included in the crop
      - `0.1` = Small face in image (more background/shoulders visible)
      - `0.9` = Large face in image (face takes up most of the crop)
      - `1.0` = Maximum face size (face fills almost entire image)
    - **Aspect Ratio**: Choose the dimensions for output images
      - `4:3` = Portrait orientation (600×800px)
      - `3:4` = Landscape orientation (800×600px) 
      - `1:1` = Square format (600×600px)
    
    ### **Background Removal Settings (When Enabled)**
    - **Remove Background**: Toggle to enable AI-powered background removal
    - **Background Removal Model**: Choose the AI model for processing
      - 🥇 **`birefnet-portrait`** - **Best quality** for profile pictures (slower)
      - ⚡ **`birefnet-general-lite`** - **Recommended** best balance of speed & accuracy
      - 🚀 **`u2net_human_seg`** - Fast processing for human subjects (fast but bad accuray and less accurate around hairs)
      - ⚖️ **`isnet-general-use`** - Good general purpose model (not recommended)
      - 💨 **`silueta`** - Fastest but basic quality
      - 📦 **`u2net`** - Basic model for general use
    
    - **Output Format**: Choose file format based on your needs
      - **PNG** - Best quality, supports transparency, larger file size
      - **JPEG** - Smaller file size, no transparency support
    
    - **Background Style/Color**: 
      - **For PNG**: Transparent, White, Black, Light Gray, or Blue backgrounds
      - **For JPEG**: White, Black, Light Gray, or Blue backgrounds (no transparency)
    
    ## 🎯 **Model Recommendations**
    
    ### **For Best Quality (Profile Pictures)**
    - Use **`birefnet-portrait`** - Specifically trained for human portraits
    - Excellent for hair details and clothing accuracy
    - Best for final production use
    
    ### **For Speed & Quality Balance** 
    - Use **`birefnet-general-lite`** - 4x faster than birefnet-portrait
    - 90-95% quality of the full model
    - **Recommended for large batches (50+ images)**
    
    ### **For Maximum Speed**
    - Use **`u2net_human_seg`** or **`silueta`** for fastest processing
    - Good for testing or when speed is critical
    
    ## 💡 **Pro Tips**
    - **Face Size Recommendation**: Use **0.9 for profile pictures** to maximize face content while maintaining natural proportions. Values below 0.8 tend to create excessive padding around the subject.
    - For **transparent backgrounds**: Choose PNG format with "Transparent" style
    - For **colored backgrounds**: Choose your preferred color (works with both PNG & JPEG)
    - **Japanese portraits**: birefnet-portrait handles Asian hair textures excellently
    - **Batch processing**: The app shows progress and estimated time for large batches
    """)

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: For best results, use high-quality images with clearly visible faces.")