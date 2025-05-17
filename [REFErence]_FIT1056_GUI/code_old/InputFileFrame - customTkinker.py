# Third party imports
import os
import customtkinter as ctk
from PIL import Image
from tkinter import filedialog
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Local application imports

class InputFrame(ctk.CTkFrame):
    """
    A InputFrame class for user to insert the CSI data.
    """

    def __init__(self, master):
        """
        Constructor for the InputFrame class.
        :param master: CTk object; the main window that the
                       input frame is to be contained.
        
        :syntax:
        In the context of graphical user interfaces (GUIs), "FocusIn" and "FocusOut" are events related to the focus of a user interface element, such as an input field or a button.
        """
        # Call the constructor from the parent class, which is ctk.CTkFrame
        super().__init__(master=master)
        # Set the master attribute to the master parameter, which is our main interface/window
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.master = master
        self.master.configure(fg_color="#ffffff")
        self.configure(fg_color="#ffffff")

        self.model = self.loadWeight()
        self.activities = ['waving', 'twist', 'standing', 'squatting', 'rubhand', 'pushpull', 'punching', 'nopeople', 'jump', 'clap']
        
        # Configure grid weights to prevent shifting
        self.grid_rowconfigure(0, weight=1)  # Prioritize top row
        self.grid_columnconfigure(0, weight=1)  # Allow centering horizontally
        
        # Label containing the welcome heading
        title = ctk.CTkLabel(master=self, text_color="#347f8c",
                             text="Human Activity Recognition with "
                                  "wifi signal",
                             font=("Microsoft Yahei UI Light", 25, "bold"), fg_color="#ffffff")
        title.grid(row=0, column=2, columnspan=2, padx=10, pady=50, sticky="n")

        # Variable to store file path
        self.file_path = ctk.StringVar()

        # Entry field (read-only) for file display (hidden later)
        self.file_entry = ctk.CTkEntry(self, textvariable=self.file_path, font=("Microsoft Yahei UI Light", 11),
                                       text_color="#3d95a5", state="readonly", width=30)
        self.file_entry.grid(row=1, column=2, columnspan=2, padx=30, pady=5, sticky="SEW")

        # File name label
        self.file_label = ctk.CTkLabel(self, text="No file selected", font=("Arial", 11), text_color="gray")
        self.file_label.grid(row=1, column=2, columnspan=2, padx=30, pady=5, sticky="SEW")

        # Browse Button
        self.browse_button = ctk.CTkButton(self, text="Browse", command=self.browse_file,
                                           font=("Microsoft Yahei UI Light", 11), text_color="white", fg_color="#347f8c")
        self.browse_button.grid(row=1, column=4, padx=5, pady=5, sticky="W")

        # Button to predict activity
        predict_button = ctk.CTkButton(master=self, text="Predict Activity",
                                       command=self.run_CNN, font=("Microsoft Yahei UI Light", 11), text_color='#fff',
                                       fg_color="#347f8c", border_width=0, width=30)
        predict_button.grid(row=5, column=2, columnspan=2, padx=50, pady=30, sticky="EW")

        # Button to open FAQ
        FAQ_button = ctk.CTkButton(fg_color="#ffffff", cursor='hand2', master=self, text="FAQ", text_color='#347f8c',
                                   command=self.open_FAQ, border_width=0,
                                   font=("Microsoft Yahei UI Light", 10, "underline"))
        FAQ_button.grid(row=6, column=2, padx=50, sticky="W")

    def browse_file(self):
        """
        Open the CSV file (raw CSI data)
        :return: None
        """
        file_path = filedialog.askopenfilename(title="Select a File",
                                               filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if file_path:
            file_name = os.path.basename(file_path)  # Extract just the filename

            # Read and store file content
            self.data = pd.read_csv(file_path).values

        # Hide entry field, show file name
        self.file_entry.grid_forget()
        self.file_label.configure(text=file_name, text_color="black")  # Show only filename
        self.file_label.grid(row=1, column=2, columnspan=2, padx=30, pady=5, sticky="SEW")





    def toTensor(self):
        """
        Convert CSI data to Tensor

        :return: None
        """
        image_shape = (300, 166)
        # target_size=(256, 256) # no need this, model is trained on 300x166 images (166 for 166 subcarriers)

        # Ensure 2D shape (handle different input sizes)
        if len(self.data.shape) == 1:  # If flat, assume it's a row vector
            data = self.data.reshape(1, -1)  # Convert to (1, N)
        else:
            data = self.data
        h, w = data.shape

        # Convert to tensor and resize
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, H, W) for grayscale
        data = F.interpolate(data.unsqueeze(0), size=image_shape, mode="bilinear", align_corners=False).squeeze(0)
        image_np = data.squeeze(0).numpy()
        # data =  ((data - data.min()) / (data.max() - data.min())) * 255
        plt.imsave(os.path.join(self.script_dir, "output_image.png"), image_np, cmap='gray') # save as image
        data = io.read_image(os.path.join(self.script_dir, "output_image.png"), mode=io.image.ImageReadMode.GRAY).type(torch.float32) # load image
        # resize_transform = transforms.Resize(image_shape)  # Resize to fixed (H, W)
        # data = resize_transform(data)  # Resize tensor
        # print(data.shape)


        return data

    def run_CNN(self):
        """
        Run CNN model to make prediction, then open the output frame, showing the prediction result.

        :return: None
        """
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

        # Print the sorted dictionary
        # print(sorted_probs)

        self.place_forget()  # Hide the current frame properly

        # Create and display the Patient login frame
        OutputFrame(self.master, self, probs).place(relx=0.5, rely=0.5, anchor=ctk.CENTER)  # Place OutputFrame correctly
        


    def loadWeight(self):
        """
        Load weight into the CNN model

        :return: CNNModel object which have been loaded with the weights
        """
        # set the device we will be using to train the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the saved weights into the model
        loaded_model = CNNModel(10)  # Recreate the model architecture
        # loaded_model.load_state_dict(torch.load(r"C:\Users\USER\Downloads\cnn_model_weights_mini_vgg.pth")) # change to file path
        loaded_model.load_state_dict(torch.load(os.path.join(self.script_dir, "cnn_model_weights_mini_vgg_25_epochs_front.pth"), map_location=torch.device('cpu')))
        loaded_model.to(device)
        loaded_model.eval()  # Set to evaluation mode

        return loaded_model






    def open_FAQ(self):
        """

        :return: None
        """
        # self.repopulate_password_entry()
        # self.repopulate_username_entry()
        # self.place_forget()
        # # Create and display the password reset page, where the user can reset their password.
        # reset_frame = ResetPW(self, self.master)
        # reset_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def clear_username_entry(self, event):
        """
        Clears the username entry field when it receives focus.
        :param event: Event object
        :return: None
        """
        # self.username_entry.delete(0, tk.END)
        pass


    def sign_up(self):
        """
        Allows the user to initiate the sign-up process. 
        This function prepares the user interface for registration by clearing any previous entries 
        in the username and password fields, and then navigates to the role selection page.
        
        :return: None
        """
        # self.repopulate_password_entry()
        # self.repopulate_username_entry()
        # self.place_forget()
        # # Create and display the Role selection page for the user to choose their role (e.g., student, teacher, parent).
        # Role(self, self.master).place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        pass




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









class OutputFrame(ctk.CTkFrame):
    """
    An OutputFrame class to show prediction outcome/output.

        :param master: Tk object; the main window that the
                       login frame is to be contained.
        :param previous_frame: the login frame.
        :param output: A list, where the 0th index is a sorted dictionary of the prediction outcome, and the 1st index is the activity having the highest probability
    """
    def __init__(self, master, previous_frame, output):
        super().__init__(master)
        self.master = master
        self.previous_frame = previous_frame  # Store the previous frame (login page)
        self.output = output  # Store the output
        self.toggle_state = False  # Track if it's expanded or collapsed

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)  

        # Label to display the Title
        Title_label = ctk.CTkLabel(self, text="Output", font=ctk.CTkFont("Microsoft Yahei UI Light", 20, "bold"), text_color="#347f8c")
        Title_label.grid(row=0, column=0, padx=20, pady=5, columnspan=4)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "Logo", str(output[1]) + "-100.png")
        logo_image = Image.open(image_path)  # Load the image using PIL
        self.login_logo = ctk.CTkImage(light_image=logo_image, size=(100, 100))
        image_label = ctk.CTkLabel(self, image=self.login_logo, text="")
        image_label.grid(row=1, column=0, rowspan=2, padx=10, pady=10)

        prediction_text = f"Predicted Action: {output[1]}\n"
        output_label = ctk.CTkLabel(self, text=prediction_text, font=ctk.CTkFont("Microsoft Yahei UI Light", 12))
        output_label.grid(row=1, column=1, padx=20, pady=5)

        self.probabilities = output[0] 

        self.Stats_button = ctk.CTkButton(self, text="Show Statistics", command=self.open_Stats,
                                          font=ctk.CTkFont("Microsoft Yahei UI Light", 11, weight="normal", underline=True), 
                                          fg_color="transparent", text_color="#347f8c", hover_color="#f0f0f0")
        self.Stats_button.grid(row=2, column=1, sticky="ew")

        self.back_button = ctk.CTkButton(self, text="Make new prediction", command=self.go_back,
                                         font=ctk.CTkFont("Microsoft Yahei UI Light", 11, weight="normal"),
                                         fg_color="#347f8c", 
                                         text_color="#fff", width=250)
        self.back_button.grid(row=3, column=0, pady=20, padx=20, columnspan=4)

        output_image_path = os.path.join(script_dir, "output_image.png")
        output_image = Image.open(output_image_path)  # Load the image using PIL
        self.output_image = ctk.CTkImage(light_image=output_image, size=(100, 100))
        self.output_image_holder = ctk.CTkLabel(self, image=self.output_image, text="")

        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.barh(list(self.probabilities.keys()), self.probabilities.values())
        ax.set_title("Probabilities")
        ax.set_xlabel("Probabilities")
        ax.set_ylabel("Actions")
        ax.set_xscale('log')
        ax.bar_label(bars, fontsize=9)
        fig.tight_layout()

        graph = FigureCanvasTkAgg(fig, self)
        graph.draw()
        self.graph = graph.get_tk_widget()

    def go_back(self):
        self.place_forget()
        self.previous_frame.place(relx=0.5, rely=0.5, anchor="center")

    def open_Stats(self):
        if self.toggle_state:
            self.output_image_holder.grid_remove()
            self.graph.grid_remove()
            self.Stats_button.configure(text="Show Statistics")
            self.back_button.configure(width=250)
        else:
            self.back_button.configure(width=500)
            self.output_image_holder.grid(row=1, column=2, padx=10, pady=10)
            self.graph.grid(row=1, column=3, padx=10, pady=10)
            self.Stats_button.configure(text="Hide Statistics")

        self.toggle_state = not self.toggle_state





if __name__ == "__main__":
    ctk.set_appearance_mode("light")  # Options: "light", "dark", "system"
    ctk.set_default_color_theme("blue")  # Can customize this if desired

    root = ctk.CTk()
    root.title("Human Activity Recognition")
    root.geometry("1200x700")    
    root.resizable(width=False, height=False)
    
    input_frame = InputFrame(root)  # Ensure this class inherits from ctk.CTkFrame
    input_frame.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)

    root.mainloop()