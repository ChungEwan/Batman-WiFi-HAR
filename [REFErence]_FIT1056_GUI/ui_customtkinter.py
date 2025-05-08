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
            print(model_frame.front_data)
        elif tab == "Side":
            print("Predicted for side")
            print(model_frame.side_data)
        elif tab == "Both":
            print("Predicted for both")
            print(model_frame.both_data_front)
            print(model_frame.both_data_side)
        return
        # Convert data to tensors
        tensor = self.toTensor().unsqueeze(0)
        # print(tensor)
        # print(tensor.shape)

        probs = [{}, None]

        with torch.no_grad(): # To prevent weights from getting updated
            logits = self.model(tensor) # raw logits
        _, output = torch.max(logits, 1) # get index of greatest value
        output_prob = torch.nn.functional.softmax(logits, dim=1) # normalise logits values to add up to 1
        for index in range(len(self.activities)): # print probability of each activity
            probs[0][self.activities[index]] = float(output_prob[0][index])
            print(f"{self.activities[index]}: {output_prob[0][index]}")
        print(self.activities[output]) # activity with highest probability :D
        probs[1] = self.activities[output]

        # Sort outcome based on probabilities and save as dictionary
        sorted_probs = dict(sorted(probs[0].items(), key=lambda item: item[1]))
        probs[0] = sorted_probs

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

    model_frame = ModelFrame(root)
    results_frame = ResultsFrame(root)
    model_frame.grid(row=0, column=0, sticky="nsew", padx=3, pady=10)
    results_frame.grid(row=0, column=1, sticky="nsew", padx=3, pady=10)

    root.mainloop()