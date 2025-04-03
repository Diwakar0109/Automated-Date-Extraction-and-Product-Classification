Automated Product Analysis System

🚀 Advancing AI & Robotics: From Product Categorization to Robotic Grasping! 🤖

I am excited to share my journey in AI, Computer Vision, and Robotics through two impactful projects:

🛒 Smart Product Categorization & OCR-Based Shelf-Life Estimation
Built a CNN-based model with ResNet-50 to classify products into Fruits/Vegetables (for freshness detection) and Packaged Items (for OCR-based shelf-life calculation).

🔹 Trained PaddleOCR on ~200K images for high-accuracy text extraction.
🔹 Extracted manufacturing & expiry dates, MRP, and quantities from packaged goods.
🔹 Implemented NLP-based date parsing for automated shelf-life estimation.

🌍 Why is this important?
✅ Prevents expired products from reaching consumers.
✅ Reduces food waste by detecting freshness.
✅ Automates inventory management for retailers.

🤖 Robotic Grasping with YOLO 11
My latest project focuses on 6-axis robotic grasping using a fine-tuned YOLO 11 model to detect hand-object interactions and guide robotic arms with precise grasping coordinates.

🔹 Enhanced YOLO 11 for real-time object detection & localization.
🔹 Designed an oriented bounding box system to calculate accurate grasp points.
🔹 Integrated with computer vision algorithms for smooth robotic movement.

🌍 Why is this important?
✅ Enables human-robot collaboration for automation tasks.
✅ Enhances precision & efficiency in robotic grasping.
✅ Bridges the gap between AI & industrial robotics.


INPUT:

  ![test1](https://github.com/user-attachments/assets/2c01be9d-35c5-4a92-856b-33c671d02b1d)
  ![test2](https://github.com/user-attachments/assets/8b4b07d9-1588-48d2-a370-8d2c8e131fb3)
  ![test3](https://github.com/user-attachments/assets/5a9d67bc-5f62-4267-a7cc-5a4e4eaceeb4)
  ![test4](https://github.com/user-attachments/assets/89ae3c6b-a130-4e14-aaea-40be21653f10)
  ![test5](https://github.com/user-attachments/assets/a2274640-09e1-4003-b933-ee6b460554b8)

  
OUTPUT:
  ![Screenshot1](https://github.com/user-attachments/assets/d6472349-c7be-4a3c-8bcf-5a8166b78ad8)
  ![Screenshot2](https://github.com/user-attachments/assets/a969c226-9262-4eec-8582-d062cb65343d)

  
EXPLANATION:

1. Introduction
    The Automated Product Analysis System is designed to process product images and extract meaningful information using advanced deep learning and natural language processing (NLP) techniques. This system integrates multiple components, including object classification, optical character recognition (OCR), price (MRP) recognition, and freshness detection, to automate product evaluation and data processing efficiently.

2. System Components and Techniques
2.1. Object Classification
  Script: object_classification.py
  
  Purpose: Identifies objects in images using a deep learning model.
  
  Key Libraries: PyTorch, torchvision, PIL
  
  Functions:
  
  object_classification(): Loads the classification model and predicts the category of an object in an image.
  
  predict_image(): Processes an input image and provides classification results.
  
  Machine Learning Model: A PyTorch-based deep learning model trained on labeled product images.

2.2. Optical Character Recognition (OCR) for Text Extraction
Script: text_extraction_ocr.py

Purpose: Extracts text from images using OCR.

Key Library: PaddleOCR

Functions:

ocr(): Detects and extracts text from an image using OCR.

Enhancements: The PaddleOCR model has been fine-tuned with 200,000 images to enhance text recognition accuracy.

2.3. Product Information Extraction
Script: extract_info.py

Purpose: Processes extracted OCR text to extract structured data, such as dates, quantities, and expiry information.

Key Libraries: SpaCy, dateparser, spellchecker

Functions:

preprocess_ocr_text(): Cleans and normalizes the extracted text.

extract_dates(): Identifies and extracts date information.

calculate_expiry_date(): Computes the expiry date based on product details.

Techniques Used: NLP-based text processing, entity recognition, and spell correction.

2.4. Maximum Retail Price (MRP) Recognition
Script: extract_mrp.py

Purpose: Identifies and extracts MRP details from the text.

Key Libraries: SpaCy, regex

Functions:

filter_mrp(): Filters and identifies MRP-related text.

extract_mrp(): Extracts and formats the MRP value.

Techniques Used: Rule-based NLP and entity extraction.

2.5. Freshness Detection
Script: freshness_test.py

Purpose: Determines the freshness of a product using an image classification model.

Key Library: TensorFlow Keras

Functions:

testing_freshness(): Utilizes a trained model to classify a product as fresh or stale.

Machine Learning Model: A Convolutional Neural Network (CNN) for image classification.

2.6. Training Modules
Freshness Detection Model Training
Script: Train/freshness_detection_train.py

Purpose: Trains a CNN-based deep learning model for freshness detection.

Key Libraries: TensorFlow, scikit-learn

Techniques Used: Image augmentation, dropout regularization, and transfer learning.

MRP Recognition Model Training
Script: Train/train_mrp_recognition.py

Purpose: Trains an NLP-based model for recognizing MRP from OCR-extracted text.

Key Library: SpaCy

Techniques Used: Named Entity Recognition (NER) for price extraction.

2.7. Automated Workflow Integration
Script: automated_workflow.py

Purpose: Integrates all components into an end-to-end pipeline.

Key Library: openpyxl (for Excel data handling)

Functions:

initialize_workbook(): Creates a structured Excel file to store extracted data.

Workflow Overview: The script orchestrates the object classification, OCR processing, MRP extraction, and freshness detection in a sequential manner.

3. Workflow Overview
Image Processing & OCR

text_extraction_ocr.py: Extracts text from images using OCR.

extract_info.py: Processes extracted text for relevant product details.

extract_mrp.py: Identifies and extracts MRP values.

Object Classification

object_classification.py: Classifies the product type using a deep learning model.

Freshness Detection

freshness_test.py: Predicts whether the product is fresh or stale using an image-based classification model.

Train/freshness_detection_train.py: Trains the model with labeled datasets.

Automated Data Processing

automated_workflow.py: Integrates all steps into a structured processing pipeline.

Extracted data is stored in an Excel sheet for further analysis.

4. Conclusion
This system leverages deep learning, natural language processing (NLP), and OCR to automate product analysis. It provides a structured approach to processing product images and extracting key attributes such as classification, pricing, and freshness status. The integration of TensorFlow, PyTorch, and SpaCy enables efficient automation of these tasks.

The system can be extended to additional applications, such as:
  
  Quality assessment
  
  Brand recognition
  
  Fraud detection in product packaging

By combining multiple AI-driven components into a single pipeline, the Automated Product Analysis System enhances efficiency in product evaluation and data extraction.
