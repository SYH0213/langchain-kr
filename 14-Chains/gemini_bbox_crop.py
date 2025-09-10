import os
import json
from pathlib import Path
from PIL import Image
import supervision as sv
from google import genai
from google.genai import types
import tempfile
import shutil
# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()


# Try to import PyMuPDF for PDF support
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyMuPDF not installed. PDF support disabled.")
    print("Install with: pip install PyMuPDF")

# Configuration
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0
DEFAULT_TARGET_WIDTH = 1280

# DEFAULT_PROMPT = (
#     "Give the 2d bounding box chart & chart title. "
#     + 'Output a JSON list of the 2D bounding box in the key "box_2d", '
#     + 'the text label in the key "label". Use descriptive labels.'
# )

DEFAULT_PROMPT = (
    "Give me the 2d bounding boxes for graphs and charts in this document, but not tables."
    + 'Output a JSON list of the 2D bounding box in the key "box_2d", '
    + 'the text label in the key "label". Use descriptive labels.'
)

# Safety settings
DEFAULT_SAFETY_SETTINGS = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"
    ),
]


def initialize_client(api_key=None):
    """Initialize and return Gemini client."""
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


def load_and_resize_image(image_path, target_width=DEFAULT_TARGET_WIDTH):
    """
    Load an image from file and resize it maintaining aspect ratio.

    Args:
        image_path: Path to the image file
        target_width: Target width for resizing (default: 1024)

    Returns:
        tuple: (original_image, resized_image)
    """
    image = Image.open(image_path)
    width, height = image.size

    # Calculate target height maintaining aspect ratio
    target_height = int(target_width * height / width)
    resized_image = image.resize(
        (target_width, target_height), Image.Resampling.LANCZOS
    )

    return image, resized_image


def detect_bounding_boxes(
    client,
    image,
    prompt=DEFAULT_PROMPT,
    model_name=MODEL_NAME,
    temperature=TEMPERATURE,
    safety_settings=DEFAULT_SAFETY_SETTINGS,
):
    """
    Use Gemini API to detect bounding boxes in an image.

    Args:
        client: Gemini client instance
        image: PIL Image object to analyze
        prompt: Prompt for the model
        model_name: Name of the Gemini model to use
        temperature: Temperature setting for generation
        safety_settings: Safety settings for the API call

    Returns:
        str: Response text from the model
    """
    response = client.models.generate_content(
        model=model_name,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            temperature=temperature,
            safety_settings=safety_settings,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text


def parse_labels_from_response(response_text, num_detections):
    """
    Parse labels from the Gemini response.

    Args:
        response_text: JSON response from Gemini model
        num_detections: Number of detected objects

    Returns:
        list: List of labels for each detection
    """
    try:
        response_data = json.loads(response_text)
        if isinstance(response_data, list):
            labels = [
                item.get("label", f"object_{i}") for i, item in enumerate(response_data)
            ]
        else:
            labels = [f"object_{i}" for i in range(num_detections)]
    except:
        # If JSON parsing fails, use generic labels
        labels = [f"object_{i}" for i in range(num_detections)]
        print("Warning: Could not parse labels from response, using generic labels")

    return labels


def sanitize_filename(label):
    """
    Clean a label string to make it suitable for use as a filename.

    Args:
        label: Label string to clean

    Returns:
        str: Cleaned label suitable for filename
    """
    clean_label = "".join(
        c for c in label if c.isalnum() or c in (" ", "-", "_")
    ).rstrip()
    return clean_label.replace(" ", "_")


def crop_single_bbox(image, bbox, label, index, output_dir, base_filename):
    """
    Crop and save a single bounding box from an image.

    Args:
        image: PIL Image object
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        label: Label for the bounding box
        index: Index of the bounding box
        output_dir: Directory to save the cropped image
        base_filename: Base filename of the original image (without extension)

    Returns:
        tuple: (cropped_image, output_path)
    """
    # Get bounding box coordinates
    x1, y1, x2, y2 = map(int, bbox)

    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)

    # Crop the image
    cropped = image.crop((x1, y1, x2, y2))

    # Clean label to use in filename
    clean_label = sanitize_filename(label)

    output_filename = f"{base_filename}_{clean_label}.png"
    output_path = os.path.join(output_dir, output_filename)

    # Save the cropped image
    cropped.save(output_path)

    # Print information
    print(f"Crop {index+1}:")
    print(f"  Label: {label}")
    print(f"  Coordinates: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"  Size: {x2-x1} x {y2-y1} pixels")
    print(f"  Saved as: {output_filename}")
    print()

    return cropped, output_path


def crop_and_save_bounding_boxes(
    image, response_text, output_dir="./output/crops", base_filename="image"
):
    """
    Crop bounding box regions from an image and save them as separate files.

    Args:
        image: PIL Image object
        response_text: JSON response from Gemini model
        output_dir: Directory to save cropped images
        base_filename: Base filename of the original image (without extension)

    Returns:
        tuple: (cropped_images, detections)
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get image resolution
    resolution_wh = image.size

    # Parse detections from Gemini response
    detections = sv.Detections.from_vlm(
        vlm=sv.VLM.GOOGLE_GEMINI_2_5, result=response_text, resolution_wh=resolution_wh
    )

    # Parse labels from response
    labels = parse_labels_from_response(response_text, len(detections))

    print(f"\nFound {len(detections)} bounding boxes to crop")
    print("-" * 50)

    # Crop and save each bounding box
    cropped_images = []
    for i, bbox in enumerate(detections.xyxy):
        label = labels[i] if i < len(labels) else f"object_{i}"
        cropped, _ = crop_single_bbox(image, bbox, label, i, output_dir, base_filename)
        cropped_images.append(cropped)

    print("-" * 50)
    print(f"✓ Successfully saved {len(detections)} cropped images to {output_dir}")

    return cropped_images, detections


def visualize_detections(image, response_text, show_plot=True, save_path=None):
    """
    Visualize detected bounding boxes on the image.

    Args:
        image: PIL Image object
        response_text: JSON response from Gemini model
        show_plot: Whether to display the plot (default: True)
        save_path: Path to save the annotated image (optional)

    Returns:
        Image: Annotated image with bounding boxes
    """
    resolution_wh = image.size

    # Parse detections
    detections = sv.Detections.from_vlm(
        vlm=sv.VLM.GOOGLE_GEMINI_2_5, result=response_text, resolution_wh=resolution_wh
    )

    # Calculate optimal annotation parameters
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    # Initialize annotators
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        smart_position=True,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_position=sv.Position.CENTER,
    )
    masks_annotator = sv.MaskAnnotator()

    # Apply annotations
    annotated = image
    for annotator in (box_annotator, label_annotator, masks_annotator):
        annotated = annotator.annotate(scene=annotated, detections=detections)

    if show_plot:
        sv.plot_image(annotated)
    
    # Save annotated image if path is provided
    if save_path:
        annotated.save(save_path)
        print(f"  Debug: Annotated image saved to {save_path}")

    return annotated


def process_image(
    image_path,
    client=None,
    prompt=DEFAULT_PROMPT,
    output_dir="./output/crops",
    visualize=True,
    debug=False,
):
    """
    Complete pipeline to process an image: detect, crop, and visualize bounding boxes.

    Args:
        image_path: Path to the image file
        client: Gemini client instance (will be initialized if None)
        prompt: Prompt for the model
        output_dir: Directory to save cropped images
        visualize: Whether to visualize the results
        debug: Whether to save annotated debug images

    Returns:
        dict: Processing results including cropped images and detections
    """
    # Initialize client if not provided
    if client is None:
        client = initialize_client()

    # Extract base filename without extension
    base_filename = Path(image_path).stem

    # Load and resize image
    original_image, resized_image = load_and_resize_image(image_path)

    # Detect bounding boxes
    response_text = detect_bounding_boxes(client, resized_image, prompt)

    # Crop and save bounding boxes with original filename
    cropped_images, detections = crop_and_save_bounding_boxes(
        original_image, response_text, output_dir, base_filename
    )

    # Visualize if requested or debug mode is on
    annotated_image = None
    debug_image_path = None
    
    if visualize or debug:
        # Only save debug image if detections exist and debug mode is enabled
        if debug and len(detections) > 0:
            debug_dir = Path(output_dir).parent / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_image_path = debug_dir / f"{base_filename}_annotated.png"
        elif debug and len(detections) == 0:
            print(f"  Debug: No detections found, skipping annotated image save")
        
        # Only visualize if there are detections or visualize is explicitly requested
        if len(detections) > 0 or visualize:
            annotated_image = visualize_detections(
                original_image, 
                response_text,
                show_plot=visualize,
                save_path=debug_image_path
            )

    return {
        "original_image": original_image,
        "resized_image": resized_image,
        "cropped_images": cropped_images,
        "detections": detections,
        "response_text": response_text,
        "annotated_image": annotated_image,
    }


def pdf_to_png(pdf_path, output_dir, dpi=150):
    """
    Convert all pages of a PDF file to PNG images.

    Args:
        pdf_path: Path to the PDF file to convert
        output_dir: Directory to save the PNG files
        dpi: Image resolution (DPI)

    Returns:
        list: List of paths to the generated PNG files
    """
    if not PDF_SUPPORT:
        raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install PyMuPDF")
    
    png_files = []
    
    try:
        # Extract base filename without extension
        base_filename = Path(pdf_path).stem
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Open PDF document
        pdf_document = fitz.open(pdf_path)
        
        print(f"Converting PDF to PNG: {pdf_path}")
        print(f"Total pages: {len(pdf_document)}")
        
        # Process all pages
        for page_num in range(len(pdf_document)):
            # Load current page
            page = pdf_document.load_page(page_num)
            
            # Set resolution
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            
            # Convert page to image
            pix = page.get_pixmap(matrix=mat)
            
            # Create output file path
            output_path = Path(output_dir) / f"{base_filename}_{page_num + 1}.png"
            
            # Save as PNG
            pix.save(str(output_path))
            png_files.append(output_path)
            
            print(f"  Page {page_num + 1}/{len(pdf_document)} converted: {output_path.name}")
        
        # Clean up
        pdf_document.close()
        
        print(f"✓ PDF conversion complete: {len(png_files)} pages converted")
        return png_files
        
    except Exception as e:
        print(f"✗ PDF conversion error: {e}")
        raise


def process_pdf(
    pdf_path,
    output_dir="./output/crops",
    prompt=DEFAULT_PROMPT,
    visualize=False,
    client=None,
    debug=False,
    keep_temp_files=False,
    dpi=150,
):
    """
    Process a PDF file by converting it to PNG images and then processing each page.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save cropped images
        prompt: Prompt for the model
        visualize: Whether to visualize the results for each image
        client: Gemini client instance (will be initialized if None)
        debug: Whether to save annotated debug images
        keep_temp_files: Whether to keep the temporary PNG files after processing
        dpi: Resolution for PDF to PNG conversion

    Returns:
        dict: Summary of processing results
    """
    # Validate PDF path
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise ValueError(f"PDF file does not exist: {pdf_path}")
    if not pdf_file.suffix.lower() == ".pdf":
        raise ValueError(f"File is not a PDF: {pdf_path}")
    
    # Create temporary directory for PNG files
    if keep_temp_files:
        temp_dir = Path(output_dir).parent / "pdf_pages"
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="pdf_to_png_"))
    
    try:
        # Convert PDF to PNG images
        print("=" * 60)
        print("STEP 1: Converting PDF to PNG images")
        print("=" * 60)
        png_files = pdf_to_png(pdf_path, str(temp_dir), dpi=dpi)
        
        if not png_files:
            print("No pages were converted from the PDF")
            return {
                "processed_count": 0,
                "failed_count": 0,
                "total_crops": 0,
                "results": [],
            }
        
        # Process the generated PNG files
        print("\n" + "=" * 60)
        print("STEP 2: Processing converted PNG images")
        print("=" * 60)
        
        # Use process_folder on the temporary directory
        results = process_folder(
            folder_path=str(temp_dir),
            output_dir=output_dir,
            prompt=prompt,
            visualize=visualize,
            client=client,
            debug=debug,
        )
        
        return results
        
    finally:
        # Clean up temporary files if requested
        if not keep_temp_files and temp_dir != Path(output_dir).parent / "pdf_pages":
            try:
                shutil.rmtree(temp_dir)
                print(f"\n✓ Temporary PNG files cleaned up")
            except Exception as e:
                print(f"\n⚠ Warning: Could not clean up temporary files: {e}")


def process_folder(
    folder_path,
    output_dir="./output/crops",
    prompt=DEFAULT_PROMPT,
    visualize=False,
    client=None,
    debug=False,
):
    """
    Process all PNG files in a specified folder.

    Args:
        folder_path: Path to the folder containing PNG files
        output_dir: Directory to save cropped images
        prompt: Prompt for the model
        visualize: Whether to visualize the results for each image
        client: Gemini client instance (will be initialized if None)
        debug: Whether to save annotated debug images for each processed file

    Returns:
        dict: Summary of processing results
    """
    # Validate folder path
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    # Initialize client once for all images
    if client is None:
        client = initialize_client()

    # Get all PNG files in the folder
    png_files = sorted(folder.glob("*.png"))

    if not png_files:
        print(f"No PNG files found in {folder_path}")
        return {
            "processed_count": 0,
            "failed_count": 0,
            "total_crops": 0,
            "results": [],
        }

    print(f"Found {len(png_files)} PNG files in {folder_path}")
    print("=" * 60)

    # Process each PNG file
    results = []
    failed_files = []
    total_crops = 0

    for idx, png_file in enumerate(png_files, 1):
        print(f"\n[{idx}/{len(png_files)}] Processing: {png_file.name}")
        print("-" * 40)

        try:
            # Process individual image
            result = process_image(
                image_path=str(png_file),
                client=client,
                prompt=prompt,
                output_dir=output_dir,
                visualize=visualize,
                debug=debug,
            )

            # Store results
            results.append(
                {
                    "file": png_file.name,
                    "status": "success",
                    "crops_count": len(result["cropped_images"]),
                }
            )

            total_crops += len(result["cropped_images"])
            print(f"✓ Successfully processed {png_file.name}")

        except Exception as e:
            print(f"✗ Failed to process {png_file.name}: {str(e)}")
            failed_files.append(png_file.name)
            results.append({"file": png_file.name, "status": "failed", "error": str(e)})

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total files processed: {len(png_files)}")
    print(f"Successful: {len(png_files) - len(failed_files)}")
    print(f"Failed: {len(failed_files)}")
    print(f"Total crops created: {total_crops}")

    if failed_files:
        print("\nFailed files:")
        for file in failed_files:
            print(f"  - {file}")

    return {
        "processed_count": len(png_files) - len(failed_files),
        "failed_count": len(failed_files),
        "total_crops": total_crops,
        "results": results,
    }


# Main execution
if __name__ == "__main__":
    import sys

    # Check if input path is provided as command line argument
    if len(sys.argv) > 1:
        INPUT_PATH = sys.argv[1]
        OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "./output/crops"
        DEBUG_MODE = "--debug" in sys.argv or "-d" in sys.argv
        KEEP_TEMP = "--keep-temp" in sys.argv
        
        input_path = Path(INPUT_PATH)
        
        # Check if input is a PDF file
        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            print(f"PDF processing mode")
            print(f"Input PDF: {INPUT_PATH}")
            print(f"Output directory: {OUTPUT_DIR}")
            if DEBUG_MODE:
                print(f"Debug mode: ENABLED (annotated images will be saved)")
            if KEEP_TEMP:
                print(f"Keep temp files: ENABLED (PNG pages will be kept)")
            
            # Process PDF file
            summary = process_pdf(
                pdf_path=INPUT_PATH,
                output_dir=OUTPUT_DIR,
                prompt=DEFAULT_PROMPT,
                visualize=False,
                debug=DEBUG_MODE,
                keep_temp_files=KEEP_TEMP,
            )
        
        # Check if input is a folder
        elif input_path.is_dir():
            print(f"Batch processing mode")
            print(f"Input folder: {INPUT_PATH}")
            print(f"Output directory: {OUTPUT_DIR}")
            if DEBUG_MODE:
                print(f"Debug mode: ENABLED (annotated images will be saved)")

            # Process all PNG files in the folder
            summary = process_folder(
                folder_path=INPUT_PATH,
                output_dir=OUTPUT_DIR,
                prompt=DEFAULT_PROMPT,
                visualize=False,
                debug=DEBUG_MODE,
            )
        
        # Check if input is a PNG file
        elif input_path.is_file() and input_path.suffix.lower() == ".png":
            print(f"Single file mode")
            print(f"Processing: {INPUT_PATH}")
            if DEBUG_MODE:
                print(f"Debug mode: ENABLED (annotated image will be saved)")
            
            # Process single PNG file
            results = process_image(
                image_path=INPUT_PATH,
                prompt=DEFAULT_PROMPT,
                output_dir=OUTPUT_DIR,
                visualize=False,
                debug=DEBUG_MODE,
            )
            
            print(f"\nProcessing complete!")
            print(f"- Original image size: {results['original_image'].size}")
            print(f"- Number of objects detected: {len(results['cropped_images'])}")
        
        else:
            print(f"Error: Invalid input path or unsupported file type")
            print(f"Supported inputs:")
            print(f"  - PDF file (.pdf)")
            print(f"  - PNG file (.png)")
            print(f"  - Folder containing PNG files")
            sys.exit(1)

    else:
        print("Usage:")
        print("  python gemini_bbox_crop.py <input> [output_dir] [options]")
        print("\nInput can be:")
        print("  - PDF file path")
        print("  - PNG file path")
        print("  - Folder path containing PNG files")
        print("\nOptions:")
        print("  --debug, -d     Save annotated debug images")
        print("  --keep-temp     Keep temporary PNG files (for PDF input)")
        print("\nExamples:")
        print("  python gemini_bbox_crop.py document.pdf")
        print("  python gemini_bbox_crop.py ./images/")
        print("  python gemini_bbox_crop.py image.png ./output --debug")
        sys.exit(0)
