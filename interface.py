import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QLabel, QVBoxLayout, QWidget, 
                            QHBoxLayout, QFrame, QPushButton, QFileDialog,
                            QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QSize
import torchvision.transforms as transforms

# Load the trained model
from main import HybridSwinCNN

model_path = 'model/model_epoch_7.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    model = HybridSwinCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
except Exception as e:
    QMessageBox.critical(None, "Error", f"Failed to load model: {str(e)}")
    sys.exit(1)

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class ImageDropLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText("\n\n Drag & Drop Brain MRI Scan Here \n\n (Supports: JPG, PNG, JPEG)")
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #5d5d5d;
                border-radius: 10px;
                font-size: 16px;
                color: #666;
                background-color: #f8f9fa;
                padding: 20px;
            }
        """)
        self.setMinimumSize(300, 300)

    def is_valid_mri(self, image_path):
        """4-stage strict validation for brain MRI scans"""
        try:
            # Stage 1: Basic file validation
            if not os.path.isfile(image_path):
                return False
            if os.path.getsize(image_path) < 10_000:  # Reject tiny files
                return False

            # Stage 2: Image format validation
            img = Image.open(image_path)
            if img.mode not in ('L', 'RGB'):
                return False

            # Stage 3: Dimensional validation (MRI specific)
            width, height = img.size
            if not (180 <= width <= 512 and 180 <= height <= 512):
                return False

            # Stage 4: Anatomical validation
            img_array = np.array(img.convert('L'))
            
            # Texture check (rejects solid colors)
            if img_array.std() < 15:
                return False
                
            # Edge detection (rejects non-anatomical images)
            edges = cv2.Canny(img_array, 50, 150)
            if edges.mean() < 5:
                return False

            return True
        except Exception:
            return False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.browse_image()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        if os.path.isfile(file_path):
            self.process_image(file_path)

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent(), 
            "Open MRI Scan", 
            "", 
            "Image Files (*.jpg *.png *.jpeg)"
        )
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        if not self.is_valid_mri(file_path):
            self.show_error("Invalid MRI Scan", 
                          "This is not a valid brain MRI image.\n"
                          "Please upload a T1-weighted axial MRI scan.")
            return
            
        self.show_image(file_path)
        self.predict_image(file_path)

    def show_image(self, file_path):
        image = QImage(file_path)
        if image.isNull():
            self.setText("Failed to load image.")
        else:
            pixmap = QPixmap.fromImage(image)
            pixmap = pixmap.scaled(256, 256, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(pixmap)

    def predict_image(self, file_path):
        try:
            image = Image.open(file_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                probs = F.softmax(outputs, dim=1)[0] * 100
                _, predicted = torch.max(outputs, 1)
                confidence = probs[predicted.item()].item()
                class_name = class_names[predicted.item()]

                parent = self.parent()
                if hasattr(parent, 'result_label'):
                    result_text = (
                        f"<b>Prediction:</b> {class_name.capitalize()}<br>"
                        f"<b>Confidence:</b> {confidence:.1f}%"
                    )
                    parent.result_label.setText(result_text)
                    
                    if class_name == 'notumor':
                        parent.result_label.setStyleSheet("color: #28a745; font-size: 14px;")
                    else:
                        parent.result_label.setStyleSheet("color: #dc3545; font-size: 14px;")
                        
        except Exception as e:
            self.show_error("Analysis Error", f"Failed to analyze image: {str(e)}")

    def show_error(self, title, message):
        QMessageBox.warning(self.parent(), title, message)

    def clear_display(self):
        self.setPixmap(QPixmap())
        self.setText("\n\n Drag & Drop Brain MRI Scan Here \n\n (Supports: JPG, PNG, JPEG)")

class BrainTumorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Tumor Detection System")
        self.setGeometry(300, 300, 500, 600)
        
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel#title {
                font-size: 24px;
                font-weight: bold;
                color: #343a40;
                padding: 10px;
            }
            QLabel#subtitle {
                font-size: 14px;
                color: #6c757d;
                padding-bottom: 15px;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0069d9;
            }
            QFrame#result_frame {
                background-color: #ffffff;
                border-radius: 8px;
                border: 1px solid #dee2e6;
                padding: 15px;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        title_label = QLabel("Brain Tumor Detection System")
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignCenter)

        subtitle_label = QLabel("Upload T1-weighted axial MRI scans for tumor classification")
        subtitle_label.setObjectName("subtitle")
        subtitle_label.setAlignment(Qt.AlignCenter)

        self.image_label = ImageDropLabel(self)

        result_frame = QFrame()
        result_frame.setObjectName("result_frame")
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(10, 10, 10, 10)

        result_title = QLabel("Analysis Result:")
        result_title.setStyleSheet("font-weight: bold; font-size: 16px; color: #495057;")

        self.result_label = QLabel("No scan analyzed yet")
        self.result_label.setStyleSheet("font-size: 14px; color: #6c757d;")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)

        result_layout.addWidget(result_title)
        result_layout.addWidget(self.result_label)
        result_frame.setLayout(result_layout)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        scan_btn = QPushButton("Scan New MRI")
        scan_btn.setIconSize(QSize(16, 16))
        scan_btn.clicked.connect(self.reset_scan)

        info_btn = QPushButton("About")
        info_btn.setIconSize(QSize(16, 16))
        info_btn.clicked.connect(self.show_about)

        exit_btn = QPushButton("Exit")
        exit_btn.setIconSize(QSize(16, 16))
        exit_btn.clicked.connect(self.close)

        button_layout.addWidget(scan_btn)
        button_layout.addStretch()
        button_layout.addWidget(info_btn)
        button_layout.addWidget(exit_btn)

        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(result_frame)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def reset_scan(self):
        """Full reset for new scan cycle"""
        self.image_label.clear_display()
        self.result_label.setText("No scan analyzed yet")
        self.result_label.setStyleSheet("font-size: 14px; color: #6c757d;")

    def show_about(self):
        about_text = (
            "Brain Tumor Detection System\n\n"
            "Uses deep learning to classify MRI scans into:\n"
            "- Glioma\n- Meningioma\n- Pituitary tumor\n- No tumor\n\n"
            "For accurate results, only use T1-weighted axial MRI scans.\n\n"
            "Made by-\n"
                 "--R Jerome FelixRaj\n"
                 "--D Joel Gunaseelan"
        )
        QMessageBox.information(self, "About", about_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = BrainTumorApp()
    window.show()
    sys.exit(app.exec_())
