import spacy  
import re

def filter_mrp(text):
    mrp_pattern = r"\b(?:MRP\s*)?(?:R|Rs|₹|INR|rupees)?\s*([\d.,]+)\b"
    mrps = re.findall(mrp_pattern, text, re.IGNORECASE)
    clean_mrps = []
    for price in mrps:
        price_clean = price.replace(',', '').replace(' ', '')
        try:
            mrp = float(price_clean)
            clean_mrps.append(mrp)
        except ValueError:
            continue
    return clean_mrps

def extract_mrp(text):
    text=text.replace("\n", " ")
    str=""
    output_dir = r"mrp_extract_tained_model" 
    nlp = spacy.load(output_dir)  
    doc = nlp(text)  
    #print("Recognized Entities:",)  
    for ent in doc.ents:  
     #   print(ent.text)
        str+=(ent.text)

    mrp=filter_mrp(str)
    if mrp:
        return max(mrp)
    else:
        return None
