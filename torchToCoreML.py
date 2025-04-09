import torch
import torch.nn as nn
import torchvision.models as models
import coremltools as ct


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()

        # Load pretrained EfficientNet
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Remove classifier and get feature size
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # Custom classification head
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(self.feature_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    num_classes = 105
    model = FaceRecognitionModel(num_classes)
    model.load_state_dict(torch.load("best_face_recognition_model.pth", map_location='cpu'))
    model.eval()

    # CoreML requires input shape: (batch_size, channels, height, width)
    example_input = torch.rand(1, 3, 224, 224)  # EfficientNet expects 224x224, not 244

    # Trace model
    traced_model = torch.jit.trace(model, example_input)

    # Convert to Core ML Program (iOS 17+)
    coreml_model = ct.convert(
        traced_model,
        convert_to='mlprogram',  # Required for CoreML 8.3+
        inputs=[ct.TensorType(shape=example_input.shape, name="input_image")]
    )

    # Save the model in new .mlpackage format
    coreml_model.save("FaceRecognition.mlpackage")

    print("✅ Conversion complete! Saved as FaceRecognition.mlpackage")
