import torch
from torchvision import transforms
from PIL import Image
from ResidualNetwork import RMN
from torchvision import datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
model = RMN(num_classes=6).to(device)
model.load_state_dict(torch.load('ResidualNet.pth', map_location=device))
model.eval()
# train_dataset = datasets.ImageFolder('images/train', transform=transform)
# class_names = train_dataset.classes
class_names=['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# print(class_names)


def get_class(image):
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, predicted_class = torch.max(probabilities, 1)
    predicted_class = predicted_class.item()
    predicted_class_name = class_names[predicted_class]
    top_prob = top_prob.item()
    return predicted_class_name, top_prob
