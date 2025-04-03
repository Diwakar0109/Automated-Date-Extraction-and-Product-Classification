from paddleocr import PaddleOCR

def ocr(image_path):
    # Paths to the exported inference model
    det_model_dir = r"tuned_rec_det_model\best_accuracy_det"  # Path to detection model
    rec_model_dir = r"tuned_rec_det_model\best_accuracy_rec"  # Path to recognition model

    # Initialize PaddleOCR with the inference models
    ocr = PaddleOCR(
        det_model_dir=det_model_dir,
        rec_model_dir=rec_model_dir,
        use_gpu=True,  # Enable GPU (set to False if running on CPU)
        download_enabled=False,  # Disable downloading
        lang='en'  # Set language to English
    )

    # Perform OCR on the image
    result = ocr.ocr(image_path, det=True, rec=True)

    # Extract text as a string
    extracted_text = ""
    for line in result[0]:  # Extract text from OCR results
        extracted_text += f"{line[1][0]}\n"
    return extracted_text
