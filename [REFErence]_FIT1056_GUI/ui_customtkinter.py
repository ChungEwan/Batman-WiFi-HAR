from tkinter import *
import customtkinter as ctk
import os
import pandas as pd
import numpy as np

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

        # Choose file text
        ctk.CTkLabel(tab_front, text="Choose file:", font=("Microsoft Yahei UI Light", 20)).grid(row=1, column=1, columnspan=2, sticky="nsw")

        # Front text
        ctk.CTkLabel(tab_front, text="Front:", font=("Microsoft Yahei UI Light", 13)).grid(row=2, column=1, padx=5, pady=3, sticky="new")

        # File label text
        file_label_front = ctk.CTkLabel(tab_front, text="No file selected", font=("Arial", 11), text_color="gray", bg_color="white")
        file_label_front.grid(row=2, column=2, padx=5, pady=5, sticky="new")

        # Browse button
        browse_button = ctk.CTkButton(tab_front, text="Browse", command=browse_file,
                                           font=("Microsoft Yahei UI Light", 11), text_color="white")
        browse_button.grid(row=2, column=3, padx=5, pady=5, sticky="n")

        # Predict Button
        predict_button_front = ctk.CTkButton(tab_front, text="Predict", command=lambda: print("Predicting..."))
        predict_button_front.grid(row=5, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        ##### TAB SIDE #####
        tab_side.grid_columnconfigure(0, weight=3)
        tab_side.grid_columnconfigure(1, weight=1)
        tab_side.grid_columnconfigure(2, weight=12)
        tab_side.grid_columnconfigure(3, weight=1)
        tab_side.grid_columnconfigure(4, weight=3)

        for i in range(7):
            tab_side.grid_rowconfigure(i, weight=1)

        # Choose file text
        ctk.CTkLabel(tab_side, text="Choose file:", font=("Microsoft Yahei UI Light", 20)).grid(row=1, column=1, columnspan=2, sticky="nsw")

        # Side text
        ctk.CTkLabel(tab_side, text="Side:  ", font=("Microsoft Yahei UI Light", 13)).grid(row=2, column=1, padx=5, pady=3, sticky="new")
        
        # File label text
        file_label_side = ctk.CTkLabel(tab_side, text="No file selected", font=("Arial", 11), text_color="gray", bg_color="white")
        file_label_side.grid(row=2, column=2, padx=5, pady=5, sticky="new")

        # Browse button
        browse_button = ctk.CTkButton(tab_side, text="Browse", command=browse_file,
                                           font=("Microsoft Yahei UI Light", 11), text_color="white")
        browse_button.grid(row=2, column=3, padx=5, pady=5, sticky="n")

        # Predict Button
        predict_button_side = ctk.CTkButton(tab_side, text="Predict", command=lambda: print("Predicting..."))
        predict_button_side.grid(row=5, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        ###### TAB BOTH ######
        tab_both.grid_columnconfigure(0, weight=3)
        tab_both.grid_columnconfigure(1, weight=1)
        tab_both.grid_columnconfigure(2, weight=12)
        tab_both.grid_columnconfigure(3, weight=1)
        tab_both.grid_columnconfigure(4, weight=3)

        for i in range(7):
            tab_both.grid_rowconfigure(i, weight=1)

        # Choose file text
        ctk.CTkLabel(tab_both, text="Choose file:", font=("Microsoft Yahei UI Light", 20)).grid(row=1, column=1, columnspan=2, sticky="nsw")

        # Front text
        ctk.CTkLabel(tab_both, text="Front:", font=("Microsoft Yahei UI Light", 13)).grid(row=2, column=1, padx=5, pady=3, sticky="new")
        
        # File label text
        file_label_both = ctk.CTkLabel(tab_both, text="No file selected", font=("Arial", 11), text_color="gray", bg_color="white")
        file_label_both.grid(row=2, column=2, padx=5, pady=5, sticky="new")

        # Browse button
        browse_button = ctk.CTkButton(tab_both, text="Browse", command=browse_file,
                                           font=("Microsoft Yahei UI Light", 11), text_color="white")
        browse_button.grid(row=2, column=3, padx=5, pady=5, sticky="n")

        # Side text
        ctk.CTkLabel(tab_both, text="Side:", font=("Microsoft Yahei UI Light", 13)).grid(row=3, column=1, padx=5, pady=3, sticky="new")
        
        # File label text
        file_label_both = ctk.CTkLabel(tab_both, text="No file selected", font=("Arial", 11), text_color="gray", bg_color="white")
        file_label_both.grid(row=3, column=2, padx=5, pady=5, sticky="new")

        # Browse button
        browse_button = ctk.CTkButton(tab_both, text="Browse", command=browse_file,
                                           font=("Microsoft Yahei UI Light", 11), text_color="white")
        browse_button.grid(row=3, column=3, padx=5, pady=5, sticky="n")

        # Predict Button
        predict_button_both = ctk.CTkButton(tab_both, text="Predict", command=lambda: print("Predicting..."))
        predict_button_both.grid(row=5, column=1, columnspan=3, padx=5, pady=5, sticky="ew")


class ResultsFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        # Add content to Results Frame
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        label = ctk.CTkLabel(self, text="Results will be displayed here")
        label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
def browse_file(self, tab):
        """
        Open the CSV file (raw CSI data)
        :return: None
        """
        file_path = ctk.filedialog.askopenfilename(title="Select a File",
                                               filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if file_path:
            file_name = os.path.basename(file_path)  # Extract just the filename

            # Read and store file content
            self.data = pd.read_csv(file_path).values

        # Hide entry field, show file name
        self.file_entry.grid_forget()
        self.file_label.configure(text=file_name, text_color="black")  # Show only filename
        self.file_label.grid(row=1, column=2, columnspan=2, padx=30, pady=5, sticky="SEW")

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