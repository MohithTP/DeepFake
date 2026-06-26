import os
import cv2
import numpy as np

def calculate_face_quality(image_path: str) -> dict:
    """
    Computes the Variance of the Laplacian to measure image/face sharpness.
    This acts as the 'Lightweight Scout' to discard extremely blurry frames.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        dict: A dictionary containing the variance score and a boolean passing status.
    """
    if not os.path.exists(image_path):
        return {"error": f"Image not found at {image_path}"}
        
    try:
        image = cv2.imread(image_path)
        if image is None:
             return {"error": "Could not read image."}
             
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Laplacian operator and calculate variance
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Threshold for blurriness (can be adjusted based on camera quality)
        threshold = 50.0 
        is_sharp = variance > threshold
        
        return {
            "variance_score": float(variance),
            "is_sharp": bool(is_sharp),
            "status": "PASS" if is_sharp else "REJECT_BLURRY"
        }
    except Exception as e:
        return {"error": str(e)}

def detect_adversarial_noise(image_path: str) -> dict:
    """
    Checks for high-frequency perturbations or noise maps indicative of adversarial attacks.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        dict: A dictionary containing the estimated noise level and whether an attack is suspected.
    """
    if not os.path.exists(image_path):
        return {"error": f"Image not found at {image_path}"}
        
    try:
        image = cv2.imread(image_path)
        if image is None:
             return {"error": "Could not read image."}
             
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply a median blur
        median = cv2.medianBlur(gray, 3)
        
        # Subtract median from original to get the noise/high-frequency residual
        noise = cv2.absdiff(gray, median)
        
        # Calculate the mean of the noise
        noise_mean = np.mean(noise)
        
        # If the average pixel difference is unusually high, it might be adversarial static.
        threshold = 15.0
        is_attack_suspected = noise_mean > threshold
        
        return {
            "noise_level": float(noise_mean),
            "attack_suspected": bool(is_attack_suspected),
            "recommendation": "FLAG" if is_attack_suspected else "SAFE"
        }
    except Exception as e:
        return {"error": str(e)}

def detect_text_density(image_path: str) -> dict:
    """
    Estimates the density of text elements in an image.
    Uses Sobel gradients and morphological closing to group text-like components.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        dict: A dictionary containing the density score, whether it is likely a document, and a recommendation.
    """
    if not os.path.exists(image_path):
        return {"error": f"Image not found at {image_path}"}
        
    try:
        image = cv2.imread(image_path)
        if image is None:
             return {"error": "Could not read image."}
             
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate horizontal gradients using Sobel operator
        grad_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        
        # Otsu thresholding to binarize the gradient map
        _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological closing with a horizontal rectangular kernel to group characters/words
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Calculate text density as the ratio of candidate text pixels to total pixels
        white_pixels = np.sum(closed == 255)
        total_pixels = closed.size
        density = float(white_pixels / total_pixels)
        
        # A document/text-heavy image typically has a density > 12% in the closed gradient space
        is_document = density > 0.12
        
        return {
            "text_density": density,
            "is_document": bool(is_document),
            "status": "TEXT_HEAVY" if is_document else "STANDARD_IMAGE"
        }
    except Exception as e:
        return {"error": str(e)}

def extract_metadata(file_path: str) -> dict:
    """
    Extracts basic metadata (size, extension, dimensions) to help the agent route the file.
    
    Args:
        file_path (str): The path to the file.
        
    Returns:
        dict: Metadata dictionary.
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found at {file_path}"}
        
    file_size = os.path.getsize(file_path)
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        modality = "video"
    elif ext in ['.txt', '.csv', '.json', '.md', '.pdf']:
        modality = "text"
    else:
        modality = "image"
    
    meta = {
        "file_size_bytes": file_size,
        "extension": ext,
        "modality": modality
    }
    
    if meta['modality'] == 'image':
        try:
             image = cv2.imread(file_path)
             if image is not None:
                 meta['dimensions'] = {"height": image.shape[0], "width": image.shape[1]}
        except Exception:
             pass
             
    return meta
