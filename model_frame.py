from tkinter import *
import customtkinter as ctk
import os
from PIL import Image

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.io as io

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from model import CNNModel

class ModelFrame(ctk.CTkFrame):
    def __init__(self, master, results_frame, **kwargs):
        super().__init__(master, **kwargs)
        # Create a tab view
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.result_frame = results_frame
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.activities_front = ['waving', 'twist', 'standing', 'squatting', 'rubhand', 'pushpull', 'punching', 'nopeople', 'jump', 'clap']
        self.activities_side = ['jump', 'clap', 'pushpull', 'nopeople', 'punching', 'rubhand', 'standing', 'waving', 'twist', 'squatting']
        self.activities_both = ['jump', 'nopeople', 'twist', 'rubhand', 'clap', 'squatting', 'standing', 'waving', 'pushpull', 'punching']


        # Logo image on top
        image_path = os.path.join(self.script_dir, "Logo", "batman_logo_transparent.png")
        logo_image = Image.open(image_path)
        self.login_logo = ctk.CTkImage(light_image=logo_image, size=(200, 105))
        image_label = ctk.CTkLabel(self, image=self.login_logo, text="")
        image_label.grid(row=0, column=0, pady=(0, 0), sticky="n")

        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Add tabs
        tab_front = self.tabview.add("Front")
        tab_side = self.tabview.add("Side")
        tab_both = self.tabview.add("Front + Side")

        # Set the model for the initial tab (front)
        self.load_all_model()



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
        predict_button_front = ctk.CTkButton(tab_front, text="Predict", command=lambda: self.predict("Front"))
        predict_button_front.grid(row=5, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        # Button to open About
        FAQ_button = ctk.CTkButton(tab_front, text="About", text_color='#ffffff',
                                   command=self.open_About,
                                   font=("Microsoft Yahei UI Light", 12, "bold","underline"), fg_color="transparent", 
                                    bg_color="transparent", hover = None, anchor="w", width=5
                                    )
        FAQ_button.grid(row=6, column=1, sticky="new")

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
        predict_button_side = ctk.CTkButton(tab_side, text="Predict", command=lambda: self.predict("Side"))
        predict_button_side.grid(row=5, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        # Button to open About
        FAQ_button = ctk.CTkButton(tab_side, text="About", text_color='#ffffff',
                                   command=self.open_About,
                                   font=("Microsoft Yahei UI Light", 12, "bold","underline"), fg_color="transparent", 
                                    bg_color="transparent", hover = None, anchor="w", width=5
                                    )
        FAQ_button.grid(row=6, column=1, sticky="new")

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
        predict_button_both = ctk.CTkButton(tab_both, text="Predict", command=lambda: self.predict("Both"))
        predict_button_both.grid(row=5, column=1, columnspan=3, padx=5, pady=5, sticky="ew")


        # Button to open About
        FAQ_button = ctk.CTkButton(tab_both, text="About", text_color='#ffffff',
                                   command=self.open_About,
                                   font=("Microsoft Yahei UI Light", 12, "bold","underline"), fg_color="transparent", 
                                    bg_color="transparent", hover = None, anchor="w", width=5
                                    )
        FAQ_button.grid(row=6, column=1, sticky="new")

    def open_About(self):
        """
        To trigger the funtion that will open and close the About section.
        :return: None
        """
        self.result_frame.toggle_text_visibility()


    def load_all_model(self):
        """
        To load the front, side, and both orientation model all at once.
        :return: None
        """
        self.model_front = self.loadWeight("Front")
        self.model_side = self.loadWeight("Side")
        self.model_both = self.loadWeight("Both")


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

    def predict(self, tab):
        """
        Run CNN model to make prediction, then open the output frame, showing the prediction result.
        If an error occurs, call a fallback or error handler function.
        """
        try:
            result = None
            if tab == "Front":
                tensor = self.toTensor(self.front_data).unsqueeze(0)
                activities = self.activities_front
                result = self.run_CNN(tensor, activities, self.model_front)
                self.result_frame.newOutput(result[0])
            elif tab == "Side":
                tensor = self.toTensor(self.side_data).unsqueeze(0)
                activities = self.activities_side
                result = self.run_CNN(tensor, activities, self.model_side)
                self.result_frame.newOutput(result[0])
            elif tab == "Both":
                tensor = self.toTensorInterleave().unsqueeze(0)
                activities = self.activities_both
                result = self.run_CNN(tensor, activities, self.model_both)
                self.result_frame.newOutput(result[0])
            else:
                raise ValueError(f"Invalid tab: {tab}")


        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            self.result_frame.handle_prediction_error(tab, e)  # call fallback/error function

        return
    
    def loadWeight(self, tab):
        """
        Load weight into the CNN model object.

        :return: CNNModel object which have been loaded with the weights
        """
        # set the device we will be using to train the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the saved weights into the model
        loaded_model = CNNModel(10)  # Recreate the model architecture

        if tab == "Front":
            loaded_model.load_state_dict(torch.load(os.path.join(self.script_dir, "Models", "front_model.pth"), map_location=torch.device('cpu')))
        elif tab == "Side":
            loaded_model.load_state_dict(torch.load(os.path.join(self.script_dir, "Models", "side_model.pth"), map_location=torch.device('cpu')))
        elif tab == "Both":
            loaded_model.load_state_dict(torch.load(os.path.join(self.script_dir, "Models", "interleaving_model.pth"), map_location=torch.device('cpu')))

        # loaded_model.load_state_dict(torch.load(r"C:\Users\USER\Downloads\cnn_model_weights_mini_vgg.pth")) # change to file path
        loaded_model.to(device)
        loaded_model.eval()  # Set to evaluation mode

        return loaded_model
    
    def run_CNN(self, tensor, activities, model):
        """
        Run CNN model to make prediction, then open the output frame, showing the prediction result.

        :return: None
        """
        probs = [{}, None]

        with torch.no_grad(): # To prevent weights from getting updated
            logits = model(tensor) # raw logits
        _, output = torch.max(logits, 1) # get index of greatest value
        output_prob = torch.nn.functional.softmax(logits, dim=1) * 100 # normalise logits values to add up to 1
        for index in range(len(activities)): # print probability of each activity
            probs[0][activities[index]] = float(output_prob[0][index])
            print(f"{activities[index]}: {output_prob[0][index]}")
        print(activities[output]) # activity with highest probability :D
        probs[1] = activities[output]

        # Sort outcome based on probabilities and save as dictionary
        sorted_probs = dict(sorted(probs[0].items(), key=lambda item: item[1], reverse=True))
        probs[0] = sorted_probs

        return probs

    def toTensor(self, data):
        """
        Convert CSI data to Tensor

        :return: None
        """
        image_shape = (300, 166)

        # Ensure 2D shape (handle different input sizes)
        if len(data.shape) == 1:  # If flat, assume it's a row vector
            data = data.reshape(1, -1)  # Convert to (1, N)
        else:
            data = data
        h, w = data.shape

        # Convert to tensor and resize
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, H, W) for grayscale
        data = F.interpolate(data.unsqueeze(0), size=image_shape, mode="bilinear", align_corners=False).squeeze(0)
        image_np = data.squeeze(0).numpy()

        plt.imsave(os.path.join(self.script_dir, "output_image.png"), image_np, cmap='gray') # save as image
        data = io.read_image(os.path.join(self.script_dir, "output_image.png"), mode=io.image.ImageReadMode.GRAY).type(torch.float32) # load image

        return data
    

    def toTensorInterleave(self, target_packets=450):
        """
        Convert interleaved CSI data (front + side) into a resized grayscale tensor.

        :param target_packets: Desired number of packets (rows) after interpolation/truncation.
        :return: PyTorch tensor of shape (1, H, W) ready for CNN input.
        """
        def resize_packets(data, target_packets):
            num_packets = data.shape[0]
            if num_packets < target_packets:
                new_index = np.linspace(0, num_packets - 1, target_packets)
                interp_func = interp1d(np.arange(num_packets), data, axis=0, kind='linear')
                return interp_func(new_index)
            else:
                return data[:target_packets, :]

        # Step 1: Stretch or truncate front & side
        front_resized = resize_packets(self.both_data_front, target_packets)
        side_resized = resize_packets(self.both_data_side, target_packets)

        # Step 2: Interleave front and side columns
        if front_resized.shape != side_resized.shape:
            raise ValueError("Front and side data must have the same shape after resizing.")

        num_rows, num_cols = front_resized.shape
        interleaved = np.empty((num_rows, num_cols * 2), dtype=front_resized.dtype)
        interleaved[:, 0::2] = front_resized
        interleaved[:, 1::2] = side_resized

        # Step 3: Convert to PyTorch tensor and resize to (300, 166)
        data = torch.tensor(interleaved, dtype=torch.float32).unsqueeze(0)  # shape (1, H, W)
        data = F.interpolate(data.unsqueeze(0), size=(300, 166), mode="bilinear", align_corners=False).squeeze(0)

        # Step 4: Save and reload as image for model compatibility
        image_np = data.squeeze(0).numpy()
        image_path = os.path.join(self.script_dir, "output_image.png")
        plt.imsave(image_path, image_np, cmap='gray')
        data = io.read_image(image_path, mode=io.image.ImageReadMode.GRAY).type(torch.float32)

        print("FINISH MERGING/INTERLEAVING 2 DATA")

        return data