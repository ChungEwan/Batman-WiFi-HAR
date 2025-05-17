from tkinter import *
import customtkinter as ctk
import os
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageTk


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.io as io

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# for interleaving
from scipy.interpolate import interp1d

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
                print("Predicted for front")
            elif tab == "Side":
                tensor = self.toTensor(self.side_data).unsqueeze(0)
                activities = self.activities_side
                result = self.run_CNN(tensor, activities, self.model_side)
                print("Predicted for side")
            elif tab == "Both":
                tensor = self.toTensorInterleave().unsqueeze(0)
                activities = self.activities_both
                result = self.run_CNN(tensor, activities, self.model_both)
                print("Predicted for both")
            else:
                raise ValueError(f"Invalid tab: {tab}")

            self.result_frame.newOutput(result[0])

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


class ResultsFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        # Add content to Results Frame
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.output = None
        self.output_probs = None
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        # Placeholder for the chart
        self.chart_canvas = None
        self.showing_about = False  # Toggle state

        # Frame for original content
        self.main_content_frame = ctk.CTkFrame(self)
        self.main_content_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.original_content()

        # Frame for ABOUT content
        self.about_frame = ctk.CTkScrollableFrame(self, width=500)
        self.about_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.about_frame.grid_remove()  # Initially hidden
        self.about_content()

    def original_content(self):
        """
        Fill the main content frame with the output of the model.
        """
        self.main_content_frame.grid_rowconfigure(0, weight=0)  # Logo row
        self.main_content_frame.grid_rowconfigure(1, weight=1)  # Spacer
        self.main_content_frame.grid_rowconfigure(2, weight=1)  # Label row
        self.main_content_frame.grid_rowconfigure(3, weight=1)  # Spacer (for bottom padding)
        self.main_content_frame.grid_columnconfigure(0, weight=1)  # Center items horizontally

        # Results label in the center
        output_label = ctk.CTkLabel(self.main_content_frame, text="OUTPUT",font=("Arial", 26,"bold"), anchor="center", justify="center" )
        output_label.grid(row=0, column=0, columnspan=2, pady=10, sticky="n")

        # Results label in the center
        self.label = ctk.CTkLabel(self.main_content_frame, text="Results will be displayed here",font=("Microsoft Yahei UI Light", 26,"bold"))
        self.label.grid(row=2, column=0, columnspan=2, pady=10, sticky="n")

    def about_content(self):
        """
        Fill the main content frame with the "About" of the system.
        """
        self.about_frame.grid_rowconfigure(0, weight=0)  # Title (ABOUT)
        self.about_frame.grid_rowconfigure(1, weight=0)  # Orientation
        self.about_frame.grid_rowconfigure(2, weight=0)  # Image
        self.about_frame.grid_rowconfigure(3, weight=0)  # Text for oreintation
        self.about_frame.grid_rowconfigure(4, weight=0)  # CSI Data Format
        self.about_frame.grid_rowconfigure(5, weight=0)  # Text for data format
        self.about_frame.grid_columnconfigure(0, weight=1)  
        self.about_frame.grid_columnconfigure(1, weight=1)  
        self.about_frame.grid_columnconfigure(2, weight=1)  
    
        ctk.CTkLabel(
            self.about_frame,
            text="ABOUT",
            font=("Arial", 26, "bold"),
            anchor="center",
            justify="center"
        ).grid(row=0, column=0, columnspan=3, pady=(10, 30))

        ctk.CTkLabel(self.about_frame, text="Orientation", font=("Arial", 20, "bold")).grid(row=1, column=0, columnspan=3, sticky="nw", padx=30)

        # Oreintation Image
        oreintation_image_path = os.path.join(self.script_dir, "Logo", "Orientation_image.png")
        orientation_image = Image.open(oreintation_image_path)
        orientation_image_ctk = ctk.CTkImage(light_image=orientation_image, size=(320, 220))
        orientation_image_label = ctk.CTkLabel(self.about_frame, image=orientation_image_ctk, text="", wraplength=600)
        orientation_image_label.grid(row=2, column=0, pady=20, columnspan = 3)

        ctk.CTkLabel(self.about_frame, text="This system predicts human activities using Channel State Information (CSI) " \
        "data captured from WiFi signals. Users can upload a CSV file containing amplitude CSI data,"
        "and the system will analyze the signal patterns to determine the most likely activity", 
        font=("Arial", 14), justify="left", wraplength=600).grid(row=3, column=0, columnspan=3, sticky="nw", padx=30, pady=10)
        ctk.CTkLabel(self.about_frame, text="Orientation Modes Supported:", font=("Arial", 14)).grid(row=4, column=0, columnspan=3, sticky="nw", padx=30)
        ctk.CTkLabel(self.about_frame, text="⮞ Front Only – CSI captured from a frontal angle", font=("Arial", 14)).grid(row=5, column=0, columnspan=3, sticky="nw", padx=30)
        ctk.CTkLabel(self.about_frame, text="⮞ Side Only – CSI captured from a side angle", font=("Arial", 14)).grid(row=6, column=0, columnspan=3, sticky="nw", padx=30)
        ctk.CTkLabel(self.about_frame, text="⮞ Front + Side – Combines both orientations through interleaving", font=("Arial", 14)).grid(row=7, column=0, columnspan=3, sticky="nw", padx=30)


        ctk.CTkLabel(self.about_frame, text="CSI Data Format", font=("Arial", 20, "bold")).grid(row=8, column=0, sticky="nw", padx=30, pady=(10, 0),columnspan=3)
        ctk.CTkLabel(self.about_frame, text="⮞ No header or index: Data only (no column names or row numbers)", font=("Arial", 14)).grid(row=9, column=0, columnspan=3, sticky="nw", padx=30)
        ctk.CTkLabel(self.about_frame, text="⮞ Columns: Exactly 166 (each = 1 subcarrier)", font=("Arial", 14)).grid(row=10, column=0, columnspan=3, sticky="nw", padx=30)
        ctk.CTkLabel(self.about_frame, text="⮞ Rows: Flexible, ideally 350-450 (each = 1 CSI packet)", font=("Arial", 14)).grid(row=11, column=0, columnspan=3, sticky="nw", padx=30)
        ctk.CTkLabel(self.about_frame, text="⮞ Cell values: Numbers only (e.g., float or int), no text or missing values", font=("Arial", 14)).grid(row=12, column=0, columnspan=3, sticky="nw", padx=30)

    def toggle_text_visibility(self):
        """
        To open and close the about frame which depicts the "About" information.
        """
        if self.showing_about:
            self.about_frame.grid_remove()
            self.main_content_frame.grid()
        else:
            self.main_content_frame.grid_remove()
            self.about_frame.grid()
        self.showing_about = not self.showing_about

    def newOutput(self, output):
        """
        Update the ResultsFrame with the highest activity and display the chart.
        """
        # Find the activity with the highest score
        highest_activity = max(output, key=output.get)
        self.output = highest_activity
        self.output_probs = output

        # Update the label with the highest activity
        self.label.configure(text=f"Predicted Activity: {self.output}")

        # Generate and display the chart
        self.display_chart(self.output_probs)

        # Remove about frame if it is open
        self.showing_about = False
        self.about_frame.grid_remove()
        self.main_content_frame.grid()

    def display_chart(self, data, bg_color="transparent"):
        """
        Function to create and display the chart.
        """
        sorted_data = sorted(data.items(), key=lambda item: item[1], reverse=True)
        highest = sorted_data[0]
        top_3 = sorted_data[1:4]

        
        # create container
        self.top_container = ctk.CTkFrame(self.main_content_frame, fg_color="transparent")
        self.top_container.grid(row=3, column=0, columnspan=2, pady=5, sticky="n")
        self.top_container.grid_columnconfigure(0, weight=1)
        self.top_container.grid_columnconfigure(1, weight=1)

        # Donut chart
        fig_donut, ax_donut = plt.subplots(figsize=(3, 3))
        fig_donut.patch.set_facecolor('#2b2b2b')
        ax_donut.set_facecolor('#2b2b2b')

        highest_label, highest_value = highest
        remaining = 100 - highest_value
        ax_donut.pie([highest_value, remaining],
                    labels=[f"{highest_label} ({highest_value:.1f}%)", ""],
                    colors=['#71D191', '#EAEAEA'],
                    startangle=90,
                    counterclock=True,
                    wedgeprops={'width': 0.3},
                    textprops={'color': 'white'})
        ax_donut.set_title("Highest Activity Probability", fontsize=14, color="white")

        # move to left
        self.chart_canvas = FigureCanvasTkAgg(fig_donut, master=self.top_container)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().grid(row=0, column=1, padx=5, pady=5, sticky="n")

        # Action image to left
        action_image_path = os.path.join(self.script_dir, "Logo", f"{self.output}.png")
        if os.path.exists(action_image_path):
            img = Image.open(action_image_path)
            # Convert to CTkImage
            ctk_image = ctk.CTkImage(light_image=img, size=(120, 120))
            self.image_label = ctk.CTkLabel(self.top_container, image=ctk_image, text="")
            self.image_label.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        # Bar chart
        fig_bar, ax_bar = plt.subplots(figsize=(7, 3))
        fig_bar.patch.set_facecolor('#2b2b2b')
        ax_bar.set_facecolor('#2b2b2b')
        if top_3:
            labels = [item[0] for item in top_3][::-1]
            values = [item[1] for item in top_3][::-1]
            ax_bar.barh(labels, values, color=['#ADD8E6', '#FFFFE0', '#DDA0DD'])
            ax_bar.set_xlim(0, 100)
            ax_bar.set_xlabel("Percentage", color="white")
            ax_bar.set_title("The Next Top 3 Activities", fontsize=14, color="white")
            ax_bar.tick_params(colors="white")

        self.bar_canvas = FigureCanvasTkAgg(fig_bar, master=self.main_content_frame)
        self.bar_canvas.draw()
        self.bar_canvas.get_tk_widget().grid(row=4, column=0, columnspan=2, pady=5, sticky="n")

    def handle_prediction_error(self, tab, error):
        self.label.configure(text=f"Please insert the compatible CSI data!")
        
class CNNModel(nn.Module):
    """
    A class for the CNN model.

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

def on_close():
    # Disable event loop updates or UI elements here if needed
    root.quit()  # or app.quit() if using CTk
    root.destroy()


if __name__ == "__main__":
    root = ctk.CTk()
    ctk.set_appearance_mode("dark")  # Options: "light", "dark", "system"
    ctk.set_default_color_theme("blue")  # Can customize this if desired
    root.title("Human Activity Recognition UI")
    root.geometry("1200x700")    
    # root.resizable(width=False, height=False)

    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=2)
    root.grid_columnconfigure(1, weight=3)

    results_frame = ResultsFrame(root, fg_color="transparent")
    model_frame = ModelFrame(root, results_frame, fg_color="transparent")
    model_frame.grid(row=0, column=0, sticky="nsew", padx=3, pady=10)
    results_frame.grid(row=0, column=1, sticky="nsew", padx=3, pady=10)

    root.protocol("WM_DELETE_WINDOW", on_close) # prevent GUI from running infinitely in background after closing

    root.mainloop()