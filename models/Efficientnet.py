import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class FaceRecognitionModel(nn.Module):
    def __init__(self, embedding_size=512, num_classes=256):
        super(FaceRecognitionModel, self).__init__()
        # Initialize EfficientNet-B3 without pre-trained weights
        self.efficientnet = EfficientNet.from_name('efficientnet-b3', include_top=False)

        # Create a custom classifier
        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Identity()  # Remove the original classifier

        # New fully connected layers for embedding and classification
        self.embedding_layer = nn.Linear(in_features, embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        # Extract features
        x = self.efficientnet(x)

        # Generate embeddings
        embeddings = self.embedding_layer(x)

        # Classify (useful during training)
        logits = self.classifier(embeddings)

        return embeddings, logits

# Initialize the model
model = FaceRecognitionModel()

# Example input tensor
input_tensor = torch.rand(1, 3, 300, 300)  # Adjust size as needed for your dataset

# Forward pass
embeddings, logits = model(input_tensor)
