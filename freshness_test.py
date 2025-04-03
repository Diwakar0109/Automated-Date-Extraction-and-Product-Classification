import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def testing_freshness(image_path):
    # Load the trained model
    model = load_model(r'model_resnet50_6_classes.h5')

    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))  # Resize to match model input
    image_array = img_to_array(image)  # Convert to numpy array
    image_array = image_array / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict the class probabilities
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get index of highest probability
    confidence_score = predictions[0][predicted_class]  # Confidence of the predicted class

    # Map the predicted class index to the class name
    class_indices = {
        'freshapples': 0,
        'freshbanana': 1,
        'freshoranges': 2,
        'rottenapples': 3,
        'rottenbanana': 4,
        'rottenoranges': 5
    }
    class_names = list(class_indices.keys())
    predicted_class_name = class_names[predicted_class]
    print(predicted_class_name)
    # Strategic shelf life mapping based on fruit type and confidence
    if 'fresh' in predicted_class_name:
        # Base shelf life for fresh items
        base_shelf_life = {
            'freshapples': 7,
            'freshbanana': 5,
            'freshoranges': 4
        }.get(predicted_class_name, 0)

        
        if confidence_score > 0.8:  
            shelf_life = base_shelf_life
        elif 0.5 < confidence_score <= 0.8: 
            shelf_life = int(base_shelf_life * 0.7)
        elif 0.2 < confidence_score <= 0.5: 
            shelf_life = int(base_shelf_life * 0.4)
        else:  
            shelf_life = 1  
    else:  
        shelf_life = 0

    return predicted_class_name, shelf_life