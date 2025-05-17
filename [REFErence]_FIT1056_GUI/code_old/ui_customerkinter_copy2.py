from tkinter import *
import customtkinter as ctk
import os
import pandas as pd
import numpy as np
from PIL import Image


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
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.result_frame = results_frame
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.activities_front = ['waving', 'twist', 'standing', 'squatting', 'rubhand', 'pushpull', 'punching', 'nopeople', 'jump', 'clap']
        self.activities_side = ['jump', 'clap', 'pushpull', 'nopeople', 'punching', 'rubhand', 'standing', 'waving', 'twist', 'squatting']
        self.activities_both = ['jump', 'nopeople', 'twist', 'rubhand', 'clap', 'squatting', 'standing', 'waving', 'pushpull', 'punching']


        self.tabview = ctk.CTkTabview(self, command=self.on_tab_change)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Add tabs
        tab_front = self.tabview.add("Front")
        tab_side = self.tabview.add("Side")
        tab_both = self.tabview.add("Front + Side")

        # Set the model for the initial tab (front)
        self.on_tab_change()


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



    def on_tab_change(self):
        """
        To load the corresponding model everytime user change tab
        :return: None
        """
        tab_name = self.tabview.get()

        print(f"Tab changed to: {tab_name}")  # This line is just for debugging, showing the tab name

        if tab_name == "Front":
            self.model = self.loadWeight("Front")
            print("load for front")
        elif tab_name == "Side":
            self.model = self.loadWeight("Side")
            print("load for side")
        elif tab_name == "Front + Side":
            self.model = self.loadWeight("Both")
            print("load for both")


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

            :return: None
            """
            ### IN PROGRESS ###
            ### CALL DIFFERENT MODELS BASED ON TAB SELECTED ###
            ### AND FRONT + SIDE WILL HAVE TO BE INTERLEAVED FIRST ###
            if tab == "Front":
                tensor = self.toTensor(self.front_data).unsqueeze(0)
                activities = self.activities_front
                print("Predicted for front")
                # print(self.front_data)
            elif tab == "Side":
                tensor = self.toTensor(self.side_data).unsqueeze(0)
                activities = self.activities_side
                print("Predicted for side")
                # print(self.side_data)
            elif tab == "Both":
                tensor = self.toTensorInterleave().unsqueeze(0)
                activities = self.activities_both
                print("Predicted for both")
                # print(self.both_data_front)
                # print(self.both_data_side)

            result = self.run_CNN(tensor, activities)
                # Example data
            #example_data = {"Clap": 60, "Jump": 90, "No People": 45, "Punching": 75, "Rub Hand": 30}
            self.result_frame.newOutput(result[0])

            # Update the output frame/pass result to output frame
           #self.result_frame.newOutput(example_data)
            #return
    
    def loadWeight(self, tab):
        """
        Load weight into the CNN model

        :return: CNNModel object which have been loaded with the weights
        """
        # set the device we will be using to train the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the saved weights into the model
        loaded_model = CNNModel(10)  # Recreate the model architecture

        if tab == "Front":
            loaded_model.load_state_dict(torch.load(os.path.join(self.script_dir, "cnn_model_weights_mini_vgg_25_epochs_front.pth"), map_location=torch.device('cpu')))
        elif tab == "Side":
            loaded_model.load_state_dict(torch.load(os.path.join(self.script_dir, "cnn_model_15_epochs_side_no_annotation.pth"), map_location=torch.device('cpu')))
        elif tab == "Both":
            loaded_model.load_state_dict(torch.load(os.path.join(self.script_dir, "cnn_model_weights_mini_vgg_20_epochs_interleave.pth"), map_location=torch.device('cpu')))

        # loaded_model.load_state_dict(torch.load(r"C:\Users\USER\Downloads\cnn_model_weights_mini_vgg.pth")) # change to file path
        loaded_model.to(device)
        loaded_model.eval()  # Set to evaluation mode

        return loaded_model
    
    def run_CNN(self, tensor, activities):
        """
        Run CNN model to make prediction, then open the output frame, showing the prediction result.

        :return: None
        """
        # Convert data to tensors
        
        # print(tensor)
        # print(tensor.shape)

        probs = [{}, None]

        with torch.no_grad(): # To prevent weights from getting updated
            logits = self.model(tensor) # raw logits
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

        :param front: 2D NumPy array of shape (packets, subcarriers) for front view.
        :param side: 2D NumPy array of shape (packets, subcarriers) for side view.
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

        self.grid_rowconfigure(0, weight=0)  # Logo row
        self.grid_rowconfigure(1, weight=1)  # Spacer
        self.grid_rowconfigure(2, weight=0)  # Label row
        self.grid_rowconfigure(3, weight=1)  # Spacer (for bottom padding)
        self.grid_columnconfigure(0, weight=1)  # Center items horizontally

        # Logo image on top
        image_path = os.path.join(self.script_dir, "Logo", "batman_logo_transparent.png")
        logo_image = Image.open(image_path)
        self.login_logo = ctk.CTkImage(light_image=logo_image, size=(200, 105))
        image_label = ctk.CTkLabel(self, image=self.login_logo, text="")
        image_label.grid(row=0, column=0, pady=(10, 0), sticky="n")

        # Results label in the center
        self.label = ctk.CTkLabel(self, text="Results will be displayed here",font=("Microsoft Yahei UI Light", 26,"bold"))
        self.label.grid(row=2, column=0, pady=10, sticky="n")

        # Placeholder for the chart
        self.chart_canvas = None

    def newOutput(self, output):
        """
        Update the ResultsFrame with the highest activity and display the chart.
        :param output: Dictionary with activity names as keys and values as scores.
        """
        # Find the activity with the highest score
        highest_activity = max(output, key=output.get)
        self.output = highest_activity
        self.output_probs = output

        # Update the label with the highest activity
        self.label.configure(text=f"Highest Activity: {self.output}")

        # Generate and display the chart
        self.display_chart(self.output_probs)

    def display_chart(self, data, bg_color="transparent"):
        """
        Generate and display a progress chart with a circular plot for the highest value
        and a horizontal bar graph for the next top 3 values.
        :param data: Dictionary with activity names as keys and values as scores.
        :param bg_color: Background color for the plots (e.g., "transparent", "#FFFFFF").
        """
        # Sort data in descending order
        sorted_data = sorted(data.items(), key=lambda item: item[1], reverse=True)

        # Extract the highest value and the next top 3 values
        highest = sorted_data[0]
        top_3 = sorted_data[1:4]

        # Create a matplotlib figure
        fig, axes = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(hspace=0.5)

        fig.patch.set_alpha(0.0) 

        for ax in axes:
            ax.set_facecolor((0, 0, 0, 0))
            ax.title.set_color("white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.tick_params(colors="white")

        # Set background color for the figure
        if bg_color == "transparent":
            fig.patch.set_alpha(0.0)  # Transparent background
        else:
            fig.patch.set_facecolor(bg_color)  # Set to specified color

        # Set background color for the axes
        for ax in axes:
            if bg_color == "transparent":
                ax.set_facecolor((0, 0, 0, 0))  # Transparent background
            else:
                ax.set_facecolor(bg_color)  # Set to specified color

        # Circular plot for the highest value
        highest_label, highest_value = highest
        remaining = 100 - highest_value
        axes[0].pie([highest_value, remaining],
                    colors=['#71D191', '#EAEAEA'],
                    startangle=90,
                    counterclock=False,
                    wedgeprops={'width': 0.3},
                    autopct='%.1f%%',
                    textprops={'color': 'white'})

        axes[0].set_title("Highest Activity", fontsize=14, color='white')

# 额外外部标题
        axes[0].text(0, 1.2, f"{highest_label} ({highest_value:.1f}%)", ha='center', fontsize=10, color='white')

        # Horizontal bar graph for the next top 3 values
        if top_3:
            labels = [item[0] for item in top_3][::-1]  # Reverse the order of labels
            values = [item[1] for item in top_3][::-1]  # Reverse the order of values
            axes[1].barh(labels, values, color=['#ADD8E6', '#FFFFE0', '#DDA0DD'])
            axes[1].set_xlim(0, 100)
            axes[1].set_xlabel("Percentage")
            axes[1].set_title("Top 3 Activities", fontsize=14)

        # Embed the matplotlib figure into the customtkinter frame
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()  # Remove the old chart if it exists

        self.chart_canvas = FigureCanvasTkAgg(fig, master=self)
        self.chart_canvas.draw()
        widget = self.chart_canvas.get_tk_widget()
        widget.configure(bg='#212121', highlightthickness=0, borderwidth=0)  # 你UI的深色背景
        widget.grid(row=3, column=0, pady=10, sticky="n")
        
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

    root.mainloop()