import os
import torch
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from io import BytesIO
import signal
import numpy as np
import cv2
import base64
import subprocess

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

import torch.nn as nn
import torchvision.models as models

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_prob=0.3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ReflectionRemovalNet(nn.Module):
    def __init__(self):
        super(ReflectionRemovalNet, self).__init__()

        self.encoder = nn.Sequential(
            ResidualBlock(3, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=2),  
        )

        self.bottleneck = nn.Sequential(
            ResidualBlock(512, 512, stride=1)
        )

        self.decoder = nn.Sequential(
            ResidualBlock(512, 256, stride=1),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(256, 128, stride=1),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(128, 64, stride=1),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64, stride=1),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  
        )

    def forward(self, x):
        enc = self.encoder(x)
        bottleneck = self.bottleneck(enc)
        dec = self.decoder(bottleneck)
        return dec

model = ReflectionRemovalNet()
model.load_state_dict(torch.load(r'D:\MANIPAL\Research\website\anti_reflection\models\model_epoch_30.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

process = None 

@app.route('/start-video', methods=['POST'])
def start_video():
    global process
    if process is None or process.poll() is not None:  # Check if no active process
        process = subprocess.Popen(['python', 'realtimevid.py'])
        return jsonify({'message': 'Real-time video processing started successfully'}), 200
    else:
        return jsonify({'message': 'Process is already running'}), 400

@app.route('/stop-video', methods=['POST'])
def stop_video():
    global process
    if process is not None and process.poll() is None:  # Check if process is running
        os.kill(process.pid, signal.SIGTERM)
        process = None
        return jsonify({'message': 'Processing stopped successfully'}), 200
    else:
        return jsonify({'message': 'No process is running'}), 400

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image = Image.open(file.stream)
    original_size = image.size 
    image = transform(image).unsqueeze(0) 

    with torch.no_grad():
        output = model(image)

    output_image = output.squeeze().cpu().numpy()
    output_image = (output_image * 255).astype('uint8')  
    output_image = Image.fromarray(output_image.transpose(1, 2, 0))  

    output_image = output_image.resize(original_size, Image.LANCZOS)

    buffered = BytesIO()  
    output_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({'message': 'Image processed successfully', 'image_data': img_str})

if __name__ == '__main__':
    app.run(debug=True)