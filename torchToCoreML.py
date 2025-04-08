import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import coremltools as ct


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()

        # Load pretrained EfficientNet from torchvision
        self.backbone = models.efficientnet_b0(pretrained=True)

        # Get the feature dimension from EfficientNet
        self.feature_dim = self.backbone.classifier[1].in_features

        # Remove the original classifier
        self.backbone.classifier = nn.Identity()

        # Add custom layers for face recognition
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.feature_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)

        # Classification head
        x = self.dropout(features)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    


if __name__ == "__main__":
    
    model = FaceRecognitionModel(105)
    model.load_state_dict(torch.load("best_face_recognition_model.pth", map_location='cpu'))
    
    model.eval()
    
    dummy_input = torch.randn((3, 3, 244, 244))
    
    traced_model = torch.jit.trace(model, dummy_input)
    
    traced_out = traced_model(dummy_input)
    
    print("output dummy shape:", model(dummy_input).shape)
    
    coreml_model = ct.convert(traced_model, convert_to='mlprogram', inputs=[ct.TensorType(shape=dummy_input.shape)])
    
    coreml_model.save('FaceRecCoreML.mlpackage')