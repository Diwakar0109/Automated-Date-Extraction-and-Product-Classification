import spacy  
from spacy.training import Example  
import os  # Import the os module  

# Prepare training data  
TRAIN_DATA = [  
    ("The MRP is ₹2,000.00", {"entities": [(11, 20, "MRP")]}),  
    ("MRP: Rs. 500 only", {"entities": [(5, 12, "MRP")]}),  
    ("Price: ₹1,O00. Incorrect OCR", {"entities": [(7, 13, "MRP")]}),  
    ("Manufacturing details. No MRP here.", {"entities": []}),  
    ("Produced on 2023-07-15. Best before 6 months. MRP ₹2,000. Contains 500 grams of product.",  
     {"entities": [(88, 95, "MRP")]}),  
    ("The MRP is ₹2,000.00", {"entities": [(11, 20, "MRP")]}),  
    ("Price: ₹1,O00. Incorrect OCR", {"entities": [(7, 13, "MRP")]}),  
    ("Manufactured on 2023.07.15. Best before 6 months. MRP ₹2,000. Contains 500 grams of product.",  
     {"entities": [(48, 55, "MRP")]}),  
    ("10 JUN 2024. Best before 6 months. MRP ₹2,O00. Contains 500 gramz of product.",  
     {"entities": [(33, 40, "MRP")]}) ,
    ("No.of Servings per packAbout7 MRP.R 25.00 ",
     {"entities": [(30, 41, "MRP")]}) ,
    ("74.25 g ET WEIGHT: 67.5 g+6.75 g Extra* BATCH No.: 14:2814B08 PKD.: 16/09/24 14/03/25 USEBY", {"entities": []}),  
    ("Date of Packaging SEP2024 238 Use By MAY2025 MRP in INR 26.00 Rs.Per Gram 0.52", {"entities": [(45,61,"MRP")]}),  
    ("ADDED ORANGEFLAVOUT BISCUNTS MRP Rs. 5.00 incl. of all taxes (Rs. 0.20 per g)", {"entities": [(29,41,"MRP")]}),  

]  

# Create a blank English model  
nlp = spacy.blank("en")  

# Add a NER pipeline with a new label "MRP"  
ner = nlp.add_pipe("ner")  

# Add labels to the NER model  
for _, annotations in TRAIN_DATA:  
    for ent in annotations.get("entities"):  
        ner.add_label(ent[2])  

# Start training  
optimizer = nlp.begin_training()  

for epoch in range(20):  
    for text, annotations in TRAIN_DATA:  
        doc = nlp.make_doc(text)  
        example = Example.from_dict(doc, annotations)  
        nlp.update([example], drop=0.1, sgd=optimizer)  

# Define the output directory  
output_dir = r"mrp_extract_tained_model"  

# Create the directory if it doesn't exist  
os.makedirs(output_dir, exist_ok=True)  

# Save the trained model to disk  
nlp.to_disk(output_dir)  

# Load the model for inference  
nlp = spacy.load(output_dir)  

# Test the model with new text  
test_text = "No.of Servings per pack-About7 MRP.R 25.00"  
doc = nlp(test_text)  

for ent in doc.ents:  
    print(ent.text, ent.label_)