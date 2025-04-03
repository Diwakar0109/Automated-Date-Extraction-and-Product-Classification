import torch
from torchvision import transforms, models
from PIL import Image
def object_classification(img_path):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model architecture
    model = models.resnet18(pretrained=True)  # Use pretrained weights
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 9)  # Adjusting for 9 classes
    model = model.to(device)

    # Load the saved model weights (assuming it was trained for 9 classes)
    model.load_state_dict(torch.load(r"classification_model.pth"))
    model.eval()

    # Define the transformation for the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Function to preprocess the image and make predictions
    def predict_image(image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            class_index = predicted.item()

        # Map index to class label (for 9 classes)
        class_labels = {
            0: 'Banana',
            1: 'Lemon',
            2: 'Mango',
            3: 'Orange',
            4: 'Pineapple',
            5: 'Tomato',
            6: 'Watermelon',
            7: 'Apple',
            8: 'Images_Packed'
        }
        return class_labels[class_index]
    result = predict_image(img_path)
    return result

