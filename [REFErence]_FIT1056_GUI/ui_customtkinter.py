from tkinter import *
import customtkinter as ctk
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.io as io

import matplotlib.pyplot as plt

class ModelFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        # Create a tab view
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        tabview = ctk.CTkTabview(self)
        tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Add tabs
        tab_front = tabview.add("Front")
        tab_side = tabview.add("Side")
        tab_both = tabview.add("Front + Side")

        ##### TAB FRONT #####
        tab_front.grid_columnconfigure(0, weight=3)
        tab_front.grid_columnconfigure(1, weight=1)
        tab_front.grid_columnconfigure(2, weight=12)
        tab_front.grid_columnconfigure(3, weight=1)
        tab_front.grid_columnconfigure(4, weight=3)

        for i in range(7):
            tab_front.grid_rowconfigure(i, weight=1)

        # Select file text
        ctk.CTkLabel(tab_front, text="Select file:", font=("Microsoft Yahei UI Light", 20)).grid(row=1, column=1, columnspan=2, sticky="sw")

        # Front text
        ctk.CTkLabel(tab_front, text="Front:", font=("Microsoft Yahei UI Light", 13)).grid(row=2, column=1, padx=5, pady=3, sticky="sew")

        # File label text
        self.file_label_front = ctk.CTkLabel(tab_front, text="No file selected", font=("Arial", 11), text_color="gray", bg_color="white")
        self.file_label_front.grid(row=2, column=2, padx=5, pady=5, sticky="sew")

        # Browse button
        browse_button = ctk.CTkButton(tab_front, text="Browse", command=lambda: self.browse_file("Front", "Front"),
                                           font=("Microsoft Yahei UI Light", 11), text_color="white")
        browse_button.grid(row=2, column=3, padx=5, pady=5, sticky="s")

        # Predict Button
        predict_button_front = ctk.CTkButton(tab_front, text="Predict", command=lambda: predict("Front"))
        predict_button_front.grid(row=5, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        ##### TAB SIDE #####
        tab_side.grid_columnconfigure(0, weight=3)
        tab_side.grid_columnconfigure(1, weight=1)
        tab_side.grid_columnconfigure(2, weight=12)
        tab_side.grid_columnconfigure(3, weight=1)
        tab_side.grid_columnconfigure(4, weight=3)

        for i in range(7):
            tab_side.grid_rowconfigure(i, weight=1)

        # Select file text
        ctk.CTkLabel(tab_side, text="Select file:", font=("Microsoft Yahei UI Light", 20)).grid(row=1, column=1, columnspan=2, sticky="sw")

        # Side text
        ctk.CTkLabel(tab_side, text="Side:  ", font=("Microsoft Yahei UI Light", 13)).grid(row=2, column=1, padx=5, pady=3, sticky="sew")
        
        # File label text
        self.file_label_side = ctk.CTkLabel(tab_side, text="No file selected", font=("Arial", 11), text_color="gray", bg_color="white")
        self.file_label_side.grid(row=2, column=2, padx=5, pady=5, sticky="sew")

        # Browse button
        browse_button = ctk.CTkButton(tab_side, text="Browse", command=lambda: self.browse_file("Side", "Side"),
                                           font=("Microsoft Yahei UI Light", 11), text_color="white")
        browse_button.grid(row=2, column=3, padx=5, pady=5, sticky="s")

        # Predict Button
        predict_button_side = ctk.CTkButton(tab_side, text="Predict", command=lambda: predict("Side"))
        predict_button_side.grid(row=5, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        ###### TAB BOTH ######
        tab_both.grid_columnconfigure(0, weight=3)
        tab_both.grid_columnconfigure(1, weight=1)
        tab_both.grid_columnconfigure(2, weight=12)
        tab_both.grid_columnconfigure(3, weight=1)
        tab_both.grid_columnconfigure(4, weight=3)

        for i in range(7):
            tab_both.grid_rowconfigure(i, weight=1)

        # Select files text
        ctk.CTkLabel(tab_both, text="Select files:", font=("Microsoft Yahei UI Light", 20)).grid(row=1, column=1, columnspan=2, sticky="sw")

        # Front text
        ctk.CTkLabel(tab_both, text="Front:", font=("Microsoft Yahei UI Light", 13)).grid(row=2, column=1, padx=5, pady=3, sticky="sew")
        
        # File label text
        self.file_label_both_front = ctk.CTkLabel(tab_both, text="No file selected", font=("Arial", 11), text_color="gray", bg_color="white")
        self.file_label_both_front.grid(row=2, column=2, padx=5, pady=5, sticky="sew")

        # Browse button
        browse_button = ctk.CTkButton(tab_both, text="Browse", command=lambda: self.browse_file("Both", "Front"),
                                           font=("Microsoft Yahei UI Light", 11), text_color="white")
        browse_button.grid(row=2, column=3, padx=5, pady=5, sticky="s")

        # Side text
        ctk.CTkLabel(tab_both, text="Side:", font=("Microsoft Yahei UI Light", 13)).grid(row=3, column=1, padx=5, pady=3, sticky="new")
        
        # File label text
        self.file_label_both_side = ctk.CTkLabel(tab_both, text="No file selected", font=("Arial", 11), text_color="gray", bg_color="white")
        self.file_label_both_side.grid(row=3, column=2, padx=5, pady=5, sticky="new")

        # Browse button
        browse_button = ctk.CTkButton(tab_both, text="Browse", command=lambda: self.browse_file("Both", "Side"),
                                           font=("Microsoft Yahei UI Light", 11), text_color="white")
        browse_button.grid(row=3, column=3, padx=5, pady=5, sticky="n")

        # Predict Button
        predict_button_both = ctk.CTkButton(tab_both, text="Predict", command=lambda: predict("Both"))
        predict_button_both.grid(row=5, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

    def browse_file(self, tab, orientation):
        """
        Open the CSV file (raw CSI data)
        :return: None
        """
        file_path = ctk.filedialog.askopenfilename(title="Select a File",
                                               filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if file_path:
            file_name = os.path.basename(file_path)  # Extract just the filename

            if tab == "Front":
                self.file_label_front.configure(text=file_name, text_color="black")
                self.front_data = pd.read_csv(file_path).values
            elif tab == "Side":
                self.file_label_side.configure(text=file_name, text_color="black")
                self.side_data = pd.read_csv(file_path).values
            elif tab == "Both":
                if orientation == "Front":
                    self.file_label_both_front.configure(text=file_name, text_color="black")
                    self.both_data_front = pd.read_csv(file_path).values
                elif orientation == "Side":
                    self.file_label_both_side.configure(text=file_name, text_color="black")
                    self.both_data_side = pd.read_csv(file_path).values

def predict(tab):
    """
    Run CNN model to make prediction, then open the output frame, showing the prediction result.

    :return: None
    """
    ### IN PROGRESS ###
    ### CALL DIFFERENT MODELS BASED ON TAB SELECTED ###
    ### AND FRONT + SIDE WILL HAVE TO BE INTERLEAVED FIRST ###
    if tab == "Front":
        print("Predicted for front")
        tensor = toTensor(model_frame.front_data).unsqueeze(0)
        print(tensor)
        return run_CNN_front(model_front, tensor)
    elif tab == "Side":
        print("Predicted for side")
        tensor = toTensor(model_frame.side_data).unsqueeze(0)
        print(tensor)
    elif tab == "Both":
        print("Predicted for both")
        print(model_frame.both_data_front)
        print(model_frame.both_data_side)
    return

def toTensor(data):
        """
        Convert CSI data to Tensor

        :return: None
        """
        image_shape = (300, 166)
        # target_size=(256, 256) # no need this, model is trained on 300x166 images (166 for 166 subcarriers)

        # Ensure 2D shape (handle different input sizes)
        if len(data.shape) == 1:  # If flat, assume it's a row vector
            data = data.reshape(1, -1)  # Convert to (1, N)

        # Convert to tensor and resize
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, H, W) for grayscale
        data = F.interpolate(data.unsqueeze(0), size=image_shape, mode="bilinear", align_corners=False).squeeze(0)
        image_np = data.squeeze(0).numpy()
        # data =  ((data - data.min()) / (data.max() - data.min())) * 255
        plt.imsave(os.path.join(script_dir, "output_image.png"), image_np, cmap='gray') # save as image
        data = io.read_image(os.path.join(script_dir, "output_image.png"), mode=io.image.ImageReadMode.GRAY).type(torch.float32) # load image
        # resize_transform = transforms.Resize(image_shape)  # Resize to fixed (H, W)
        # data = resize_transform(data)  # Resize tensor
        return data

def run_CNN_front(model, tensor):
    probs = [{}, None]

    with torch.no_grad(): # To prevent weights from getting updated
        logits = model(tensor) # raw logits
    _, output = torch.max(logits, 1) # get index of greatest value
    output_prob = torch.nn.functional.softmax(logits, dim=1) # normalise logits values to add up to 1
    for index in range(len(activities)): # print probability of each activity
        probs[0][activities[index]] = float(output_prob[0][index])
        print(f"{activities[index]}: {output_prob[0][index]}")
    print(activities[output]) # activity with highest probability :D
    probs[1] = activities[output]

    # Sort outcome based on probabilities and save as dictionary
    sorted_probs = dict(sorted(probs[0].items(), key=lambda item: item[1]))
    probs[0] = sorted_probs
    return output

class CNNModel(nn.Module):
    """
    A class to for the CNN model.

    """
    def __init__(self, num_classes=10):  # Adjust num_classes as needed
        super(CNNModel, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces to (150, 83)

        # Block 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces to (75, 41)

        # Block 3
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces to (37, 20)

        # Compute the flattened feature size
        self.flatten_dim = self._get_flatten_dim()

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _get_flatten_dim(self):
        """Helper function to compute feature map size after convolutions."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 300, 166)  # Batch size 1, single channel
            x = self.pool1(F.relu(self.bn1(self.conv2(F.relu(self.conv1(dummy_input))))))
            x = self.pool2(F.relu(self.bn2(self.conv4(F.relu(self.conv3(x))))))
            x = self.pool3(F.relu(self.bn3(self.conv6(F.relu(self.conv5(x))))))
            return x.numel()

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv2(F.relu(self.conv1(x))))))
        x = self.pool2(F.relu(self.bn2(self.conv4(F.relu(self.conv3(x))))))
        x = self.pool3(F.relu(self.bn3(self.conv6(F.relu(self.conv5(x))))))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation (CrossEntropyLoss applies softmax)
        return x

class ResultsFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        # Add content to Results Frame
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.output = None
        self.output_probs = None


        label = ctk.CTkLabel(self, text="Results will be displayed here")
        label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        

def loadModel(model_class, weight):
        """
        Load weight into the CNN model

        :return: CNNModel object which have been loaded with the weights
        """
        # set the device we will be using to train the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the saved weights into the model
        loaded_model = model_class(10)  # Recreate the model architecture
        loaded_model.load_state_dict(torch.load(os.path.join(script_dir, weight), map_location=torch.device('cpu')))
        loaded_model.to(device)
        loaded_model.eval()  # Set to evaluation mode

        return loaded_model

if __name__ == "__main__":
    root = ctk.CTk()
    ctk.set_appearance_mode("dark")  # Options: "light", "dark", "system"
    ctk.set_default_color_theme("blue")  # Can customize this if desired
    root.title("Human Activity Recognition UI")
    root.geometry("1200x700")
    # Set the master attribute to the master parameter, which is our main interface/window
    script_dir = os.path.dirname(os.path.abspath(__file__))
    activities = ['waving', 'twist', 'standing', 'squatting', 'rubhand', 'pushpull', 'punching', 'nopeople', 'jump', 'clap']
    model_front = loadModel(CNNModel, "cnn_model_weights_mini_vgg_25_epochs_front.pth")
    # root.resizable(width=False, height=False)

    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=2)
    root.grid_columnconfigure(1, weight=3)

    model_frame = ModelFrame(root)
    results_frame = ResultsFrame(root)
    model_frame.grid(row=0, column=0, sticky="nsew", padx=3, pady=10)
    results_frame.grid(row=0, column=1, sticky="nsew", padx=3, pady=10)

    root.mainloop()