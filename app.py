import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import zipfile
import random
import shutil
import yaml
import subprocess
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from PyQt5.QtWidgets import QApplication,QGroupBox,QGridLayout,QScrollArea,QTabWidget, QMainWindow,QLineEdit,QSlider, QTextEdit, QWidget, QVBoxLayout,QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QIcon, QTransform, QImage

class TrainingThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def __init__(self,yaml_path,epochs,save_dir,model_name):
        super().__init__()
        self.yaml_path = yaml_path
        self.epochs = epochs
        self.save_dir = save_dir
        self.model_name = model_name
        
    def run(self):
        command=[
            "yolo",
            "task=obb",
            "mode=train",
            f"data={self.yaml_path}",
            f"epochs={self.epochs}",
            f"project={self.save_dir}",
            f"name={self.model_name}",
            "exist_ok=True",
        ]
        
        process=subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,bufsize=1)
        
        for line in process.stdout:
            self.log_signal.emit(line)
            
        process.wait()
        self.finished_signal.emit() 
        
class YOLOTrainerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Model Trainer")
        #self.setWindowIcon(QIcon("logo.png"))
        self.setGeometry(300,200,800,600)
        self.resize(900,700)
        
        
        self.source_path = ""
        self.destination_path = ""
        self.image_rotation_angles={}
        self.current_enlarged_image_path= None
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.tabs=QTabWidget()
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.addWidget(self.tabs)
        
        self.train_tab=QWidget()
        self.train_layout=QVBoxLayout(self.train_tab)
        
        #model name ka
        model_group = QGroupBox("Model Name")
        model_layout= QVBoxLayout()
        self.model_name_input=QLineEdit()
        self.model_name_input.setPlaceholderText("Enter your model name (default: custom_train)")
        model_layout.addWidget(QLabel("Model Name:"))
        model_layout.addWidget(self.model_name_input)
        model_group.setLayout(model_layout)
        
        #zip uploading ka 
        zip_group=QGroupBox("📦 Upload ZIP Dataset")
        zip_layout = QVBoxLayout()
        self.select_source_button=QPushButton("Select ZIP File")
        self.select_source_button.clicked.connect(self.select_zip_file)
        self.source_path_label = QLabel("🔍 Source ZIP Path: Not selected")
        zip_layout.addWidget(self.select_source_button)
        zip_layout.addWidget(self.source_path_label)
        zip_group.setLayout(zip_layout)
        
        #Destination ka 
        dest_group = QGroupBox("Output Folder")
        dest_layout= QVBoxLayout()
        self.select_dest_button=QPushButton("Select Destination Folder")
        self.select_dest_button.clicked.connect(self.select_destination_folder)
        self.dest_path_label = QLabel("💾 Destination Folder: Not set")
        dest_layout.addWidget(self.select_dest_button)
        dest_layout.addWidget(self.dest_path_label)
        dest_group.setLayout(dest_layout)
        
        #submit ka
        self.submit_button = QPushButton ("Submit and Prepare Dataset")
        self.submit_button.setEnabled(False)
        self.submit_button.clicked.connect(self.submit_processing)
        
        #epochs
        epoch_group=QGroupBox("🔁 Training Settings")
        epoch_layout = QVBoxLayout()
        self.epoch_label = QLabel("Number of Epochs: 60")
        self.epoch_slider= QSlider(Qt.Horizontal)
        self.epoch_slider.setMinimum(1)
        self.epoch_slider.setMaximum(100)
        self.epoch_slider.setValue(60)
        self.epoch_slider.valueChanged.connect(
            lambda val: self.epoch_label.setText(f"Number of epochs: {val}")
        )
        epoch_layout.addWidget(self.epoch_label)
        epoch_layout.addWidget(self.epoch_slider)
        epoch_group.setLayout(epoch_layout)
        
        #training butt-onnn hehe
        self.train_button=QPushButton("Start training")
        self.train_button.setEnabled(False)
        self.train_button.clicked.connect(self.start_training)
        
        #console stfs
        console_group=QGroupBox("Console output")
        console_layout = QVBoxLayout()
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        console_layout.addWidget(self.console_output)
        console_group.setLayout(console_layout)
        
        self.train_layout.addWidget(model_group)
        self.train_layout.addWidget(zip_group)
        self.train_layout.addWidget(dest_group)
        self.train_layout.addWidget(self.submit_button)
        self.train_layout.addWidget(epoch_group)
        self.train_layout.addWidget(self.train_button)
        self.train_layout.addWidget(console_group)
        
        self.tabs.addTab(self.train_tab,"Train")
        
        #Validation tab
        self.validate_tab = QWidget()
        self.validate_layout=QVBoxLayout(self.validate_tab)
        
        #Enlarging the imaeg
        self.image_display_label = QLabel("🔍 Click an image from the list below to preview.")
        self.image_display_label.setAlignment(Qt.AlignCenter)
        self.image_display_label.setFixedHeight(600)
        self.image_display_label.setStyleSheet("border: 1px solid gray; background-color: white;")
        
        #rotate button
        self.rotate_button=QPushButton("Rotate 90")
        self.rotate_button.setEnabled(False)
        self.rotate_button.clicked.connect(self.rotate_current_image)
        
        #thumbnail preview
        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_container = QWidget()
        self.thumbnail_layout = QGridLayout(self.thumbnail_container)
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_scroll.setWidget(self.thumbnail_container)
        
        #buttton
        button_layout = QVBoxLayout()
        button_layout.setSpacing(5)
        
        self.valset_button = QPushButton("📂 Validate with Val Set")
        self.valset_button.clicked.connect(self.load_val_set_images)
        
        self.import_button = QPushButton("📥 Import to Validate")
        self.import_button.clicked.connect(self.import_images_to_validate)
        
        button_layout.addWidget(self.valset_button)
        button_layout.addWidget(self.import_button)
        
        buttons_widget = QWidget()
        buttons_widget.setLayout(button_layout)
        
        #run inference button
        self.inference_button = QPushButton("Run Inference")
        self.inference_button.setEnabled(False)
        self.inference_button.clicked.connect(self.run_inference_on_image)
        
        #button and thumbnail side by side
        controls_and_thumbnails_layout = QHBoxLayout()
        controls_and_thumbnails_layout.addWidget(buttons_widget, alignment= Qt.AlignTop)
        controls_and_thumbnails_layout.addWidget(self.thumbnail_scroll)
        
        #Final Layout 
        self.validate_layout.addWidget(self.image_display_label)
        self.validate_layout.addWidget(self.rotate_button)
        self.validate_layout.addWidget(self.inference_button)
        self.validate_layout.addLayout(controls_and_thumbnails_layout)
        
        
        self.tabs.addTab(self.validate_tab,"Validate")
        self.tabs.setTabEnabled(1,False)
        
        self.central_widget.setLayout(main_layout)
        
        
    def log(self, message):
        print(message)
        self.console_output.append(message)
    
    def start_training(self): 
        model_name = self.model_name_input.text().strip()
        if not model_name:
            model_name="custom_train"
            
        yaml_path=os.path.join(self.destination_path, "dataset", "data.yaml")
        if not os.path.exists(yaml_path):
            QMessageBox.critical(self, "Error", "YAML file not found. Please prepare the dataset first.")
            return
        self.console_output.clear()
        self.console_output.append("Starting training...")
        
        epochs = self.epoch_slider.value()
        self.train_thread = TrainingThread(yaml_path,epochs, self.destination_path,model_name=model_name)
        self.train_thread.log_signal.connect(self.console_output.append)
        self.train_thread.finished_signal.connect(self.on_training_complete)
        self.train_thread.start()
    
    def on_training_complete(self):
        self.console_output.append("Training Completed!")
        self.tabs.setTabEnabled(1,True)
    
    def load_val_set_images(self):
        val_dir = os.path.join(self.destination_path,"dataset","images","val")
        if not os.path.exists(val_dir):
            QMessageBox.critical(self,"Error","Validation images directory not found")
            return
        
        images= [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        while self.thumbnail_layout.count():
            child = self.thumbnail_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        for i, image_file in enumerate(images):
            image_path = os.path.join(val_dir, image_file)
            pixmap = QPixmap(image_path).scaled(120,120,Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            thumb_button = QPushButton()
            thumb_button.setIcon(QIcon(pixmap))
            thumb_button.setIconSize(QSize(120,120))
            thumb_button.setFixedSize(130,130)
            thumb_button.clicked.connect(lambda _, p=image_path: self.show_enlarged_image(p))
            
            row,col = divmod(i,6)
            self.thumbnail_layout.addWidget(thumb_button,row,col)
        
    def import_images_to_validate(self):
        file_dialog = QFileDialog()
        files,_ =file_dialog.getOpenFileNames(self,"Select Images","","Image Files(*.png *.jpg *jpeg)")
        if files:
            folder = os.path.dirname(files[0])
            image_files = [os.path.basename(f) for f in files]
            self.display_validation_image_list(image_files,folder)
    
    def display_validation_image_list(self, image_files, folder_path):
        # Clear previous thumbnails
        while self.thumbnail_layout.count():
            child = self.thumbnail_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add new thumbnails
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(folder_path, image_file)
            pixmap = QPixmap(image_path).scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            thumb_button = QPushButton()
            thumb_button.setIcon(QIcon(pixmap))
            thumb_button.setIconSize(QSize(120, 120))
            thumb_button.setFixedSize(130, 130)
            thumb_button.clicked.connect(lambda _, p=image_path: self.show_enlarged_image(p))
            
            row, col = divmod(i, 6)
            self.thumbnail_layout.addWidget(thumb_button, row, col)

            
    def show_enlarged_image(self, image_path):
        if not os.path.exists(image_path):
            QMessageBox.warning(self, "Error", "Image not found")
            return 

        self.current_enlarged_image_path = image_path

        if image_path not in self.image_rotation_angles:
            self.image_rotation_angles[image_path] = 0

        self.display_image_with_rotation(image_path)
        self.rotate_button.setEnabled(True)
        self.inference_button.setEnabled(True)
        
    def run_inference_on_image(self):
        if not self.current_enlarged_image_path:
            QMessageBox.warning(self,"Error","No image selected for inference.")
            return
        
        try:
            model_name = self.model_name_input.text().strip()
            if not model_name:
                model_name="custom_train"
            angle = self.image_rotation_angles.get(self.current_enlarged_image_path,0)
            pil_image = Image.open(self.current_enlarged_image_path).convert("RGB")
            if angle!=0:
                pil_image = pil_image.rotate(-angle,expand=True)
            
            temp_path=os.path.join(os.getcwd(),"temp_rotated_input.jpg")
            pil_image.save(temp_path)
            
            model_path = os.path.join(self.destination_path,model_name,"weights","best.pt")

            model = YOLO(model_path)
            results = model(temp_path)
            
            # if not results or results[0].boxes is None or len(results[0].boxes)==0:
            #     QMessageBox.information(self,"Inference","No Objects detected in the image")
            #     self.display_image_with_rotation(self.current_enlarged_image_path)
            #     return
            
            result_image = results[0].plot()
            
            height,width,channel= result_image.shape
            bytes_per_line=3*width
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            qimage=QImage(result_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage).scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.image_display_label.setPixmap(pixmap)
            self.log("Inference completed")
            os.remove(temp_path)
        
        except Exception as e:
            QMessageBox.critical(self,"Inference Error",f"An error occurred during inference:\n{str(e)}")
        
    def display_image_with_rotation(self, image_path):
        try:
            angle = self.image_rotation_angles.get(image_path, 0)
            pixmap = QPixmap(image_path)
            if angle != 0:
                transform = QTransform().rotate(angle)
                pixmap = pixmap.transformed(transform, Qt.SmoothTransformation)

            self.image_display_label.setPixmap(
                pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display image:\n{str(e)}")

    
    def rotate_current_image(self):
        if not self.current_enlarged_image_path:
            return
        
        prev_angle = self.image_rotation_angles.get(self.current_enlarged_image_path,0)
        new_angle = (prev_angle+90)%360
        self.image_rotation_angles[self.current_enlarged_image_path]=new_angle
        
        self.display_image_with_rotation(self.current_enlarged_image_path)
            

                     
    def normalize_quad_coords(Self,coords,img_w=1280,img_h=720):
        normalized=[]
        for i in range(0, len(coords), 2):
            x = coords[i] / img_w
            y = coords[i + 1] / img_h
            normalized.extend([round(x, 6), round(y, 6)])
        return normalized
    
    def select_zip_file(self):
        zip_path, _ = QFileDialog.getOpenFileName(self, "Select ZIP File", "", "ZIP Files (*.zip)")
        if zip_path:
            self.source_path = zip_path
            self.source_path_label.setText(f"🔍Source ZIP Path: {zip_path}")
            self.log(f"Selected ZIP file: {zip_path}")
            self.check_ready_to_submit()
            
        
    def select_destination_folder(self):
        folder = QFileDialog.getExistingDirectory(self,"Select Destination Folder")
        if folder:
            self.destination_path = folder
            self.dest_path_label.setText(f"💾Destination Folder: {folder}")
            self.log(f"Selected destination folder: {folder}")
            self.check_ready_to_submit()
            
        
    
    def check_ready_to_submit(self):
        if self.source_path and self.destination_path:
            self.submit_button.setEnabled(True)
            self.log("Ready to submit: Source and destination paths are set.")
            
    
    def submit_processing(self):
        working_dir = os.path.join(self.destination_path, "dataset")
        try:
            if os.path.exists(working_dir):
                shutil.rmtree(working_dir)
            os.makedirs(working_dir)
            
            with zipfile.ZipFile(self.source_path, 'r') as zip_ref:
                zip_ref.extractall(working_dir)
                
            self.prepare_dataset(working_dir)
            self.train_button.setEnabled(True)
            QMessageBox.information(self, "Success", "Dataset prepared successfully!")
            self.log("Dataset prepared successfully!")
        except Exception as e:
            self.log(f"Error preparing dataset: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to prepare dataset: {str(e)}")
        
            
    def prepare_dataset(self, working_dir):
        images_dir = os.path.join(working_dir, "images")
        labels_dir = os.path.join(working_dir, "labels")
        classes_path= os.path.join(working_dir, "classes.txt")
        assert os.path.exists(images_dir), "Images directory not found"
        assert os.path.exists(labels_dir), "Labels directory not found"
        assert os.path.exists(classes_path), "Classes file not found"
        
        #Read classes
        with open(classes_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines() if line.strip()]
        nc=len(class_names)
        print(nc)
        
        train_images_dir = os.path.join(working_dir,"images/train")
        val_images_dir = os.path.join(working_dir, "images/val")
        train_labels_dir = os.path.join(working_dir, "labels/train")
        val_labels_dir = os.path.join(working_dir, "labels/val")
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)
        
        images_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print("Total images:", len(images_files))
        self.log(f"Total images found: {len(images_files)}")
        self.log(f"Total classes: {nc}")
        
        random.shuffle(images_files)
        split_idx = int(0.85 * len(images_files))
        
        train_files = images_files[:split_idx]
        val_files = images_files[split_idx:]
        def move_files_normalized(file_list,dest_img,dest_lbl):
            for img_file in file_list:
                base = os.path.splitext(img_file)[0]
                label_file = base + ".txt"
                
                src_img = os.path.join(images_dir, img_file)
                dest_img_path = os.path.join(dest_img, img_file)
                shutil.move(src_img, dest_img_path)
                
                #normalizing nw
                src_label = os.path.join(labels_dir, label_file)
                dest_label_path = os.path.join(dest_lbl, label_file)
                
                if os.path.exists(src_label):
                    with open(src_label, 'r') as f:
                        lines=f.readlines()
                    
                    new_lines = []
                    for line in lines:
                        parts=line.strip().split()
                        if len(parts) == 1:
                            dot_split = line.strip().split()
                            if len(dot_split) > 1 and dot_split[0].isdigit():
                                class_id = dot_split[0]
                                rest = ".".join(dot_split[1:]).strip().split()
                                parts = [class_id] + rest
                                
                        if len(parts) == 9:
                            class_id = int(parts[0])
                            coords = list(map(float, parts[1:]))
                            normalized = self.normalize_quad_coords(coords)
                            line_str = f"{class_id} " + " ".join(f"{v:.6f}" for v in normalized)
                            new_lines.append(line_str)
                        else:
                            print(f"Skipping line with unexpected format: {line.strip()}")
                    
                    with open(dest_label_path, 'w') as f:
                        f.write("\n".join(new_lines) + "\n")
                    
                    os.remove(src_label)
                else:
                    open(dest_label_path, 'w').close()
        
        move_files_normalized(train_files, train_images_dir, train_labels_dir)
        move_files_normalized(val_files, val_images_dir, val_labels_dir)
        
        yaml_path = os.path.join(working_dir, "data.yaml")
        with open(yaml_path, 'w') as f:
            f.write(f"path: {os.path.abspath(working_dir)}\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/val\n\n")
            f.write(f"nc: {nc}\n")
            f.write(f"names: {class_names}")
        
        with open(yaml_path, 'r') as f:
            yaml_content = yaml.safe_load(f)
            self.log("YAML Configuration:")
            self.log(yaml.dump(yaml_content, default_flow_style=False))

        print("✅ Dataset ready at:", working_dir)

            
if __name__ == "__main__":
    app = QApplication([])
    window = YOLOTrainerApp()
    window.showMaximized()
    app.exec_()
        