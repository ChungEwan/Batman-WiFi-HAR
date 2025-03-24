# Third party imports
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Local application imports


class InputFrame(tk.Frame):
    """
    A InputFrame class for user to insert the CSI data.
    """

    def __init__(self, master):
        """
        Constructor for the InputFrame class.
        :param master: Tk object; the main window that the
                       login frame is to be contained.
        
        :syntax:
        In the context of graphical user interfaces (GUIs), "FocusIn" and "FocusOut" are events related to the focus of a user interface element, such as an input field or a button. Here's what each event means:

        FocusIn Event: This event occurs when a GUI element gains focus, which typically happens when a user interacts with it. For example, 
        when a user clicks on an input field or navigates to it using the keyboard, the input field gains focus, and the "FocusIn" event is triggered. 
        This event is often used to perform actions when an element becomes the active focus, 
        such as clearing placeholder text or changing the appearance of the focused element.

        FocusOut Event: This event occurs when a GUI element loses focus, which happens when the user clicks outside the element or navigates away from it. 
        For example, if a user clicks on an input field, enters some text, and then clicks somewhere else or presses the "Tab" key to move to the next input field, 
        the first input field loses focus, and the "FocusOut" event is triggered. This event is often used to perform actions when an element is no longer the active focus, 
        such as restoring placeholder text or validating user input.
        """
        # call the constructor from the parent class, which is tk.Frame
        super().__init__(master=master)
        # set the master attribute to the master parameter
        # which is our main interface / window
        self.master = master
        self.master.configure(bg="#ffffff")
        self.configure(bg="#ffffff")

        self.model = self.loadWeight()

        # Image obtained from:
        # https://www.veryicon.com/icons/healthcate-medical/medical-icon-two-color-icon/ico-health-clinic.html
        # set the image path to variable
        # image_path = "./images/changed.png"
        # # create a PhotoImage object from the image path, set the PhotoImage object to a class attribute
        # self.login_logo = tk.PhotoImage(file=image_path)

        # Configure grid weights to prevent shifting
        self.grid_rowconfigure(0, weight=1)  # Prioritize top row
        self.grid_columnconfigure(0, weight=1)  # Allow centering horizontally


        # tk.Label(master=self, 
        #          #image=self.login_logo
        #          bg="#ffffff", border=0).grid(row=0, column=0, rowspan=6,
                                                                                #   columnspan=2, padx=10, pady=10)

        # Label containing the welcome heading
        title = tk.Label(master=self, fg="#347f8c",
                               text="Human Activity Recognition with "
                                    "wifi signal",
                               font=("Microsoft Yahei UI Light", 25, "bold"), bg="#ffffff")
        title.grid(row=0, column=2, columnspan=2, padx=10, pady=50, sticky="n")



        # Variable to store file path
        self.file_path = tk.StringVar()

        # Entry field (read-only) for file display (hidden later)
        self.file_entry = tk.Entry(self, textvariable=self.file_path, font=("Microsoft Yahei UI Light", 11),
                                foreground="#3d95a5", state="readonly", width=30)
        self.file_entry.grid(row=1, column=2, columnspan=2, padx=30, pady=5, sticky="SEW")

        # File name label 
        self.file_label = tk.Label(self, text="No file selected", font=("Arial", 11), 
                                 fg="gray")  
        self.file_label.grid(row=1, column=2, columnspan=2, padx=30, pady=5, sticky="SEW")

        # Browse Button
        self.browse_button = tk.Button(self, text="Browse", command=self.browse_file,
                                    font=("Microsoft Yahei UI Light", 11), bg="#347f8c", fg="white")
        self.browse_button.grid(row=1, column=4, padx=5, pady=5, sticky="W")




        # Button to predict activity
        predict_button = tk.Button(master=self, text="Predict Activity",
                                 command=self.run_CNN, font=("Microsoft Yahei UI Light", 11), fg='#fff',
                                 bg="#347f8c", border=0, width=30)
        predict_button.grid(row=5, column=2, columnspan=2, padx=50, pady=30, sticky=tk.EW)



        # Button to open FAQ?
        FAQ_button = tk.Button(bg="#ffffff", cursor='hand2', master=self, text="FAQ",fg='#347f8c',
                                           command=self.open_FAQ, relief="flat", borderwidth=0,
                                           font=("Microsoft Yahei UI Light", 10, "underline"))
        FAQ_button.grid(row=6, column=2, padx=50, sticky=tk.W)

        # # Button to sign up
        # sign_up_button = tk.Button(bg="#ffffff", master=self, text="Sign Up", fg='#347f8c',relief="flat", cursor='hand2'
        #                            , font=("Microsoft Yahei UI Light", 10, 'underline'), borderwidth=0, command=self.sign_up)
        # sign_up_button.grid(row=6, column=3, padx=50, sticky=tk.E)

        # Variable and label to inform user of login outcome
        # self.login_text = tk.StringVar()
        # login_message = tk.Label(bg="#ffffff", fg='red', master=self, textvariable=self.login_text)

        # login_message.grid(row=7, column=2, columnspan=2, padx=10, pady=10)

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
        self.file_label.config(text=file_name, fg="black")  # Show only filename
        self.file_label.grid(row=1, column=2, columnspan=2, padx=30, pady=5, sticky="SEW")




    def toTensor(self):
        image_shape = (300, 166)

        # Ensure 2D shape (handle different input sizes)
        if len(self.data.shape) == 1:  # If flat, assume it's a row vector
            data = self.data.reshape(1, -1)  # Convert to (1, N)
        h, w = data.shape

        # Convert to tensor and resize
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, H, W) for grayscale
        resize_transform = transforms.Resize(self.target_size)  # Resize to fixed (H, W)
        data = resize_transform(data)  # Resize tensor

        return data

    def run_CNN(self):
        """
        Run CNN model to make prediction, then open the output frame, showing the prediction result.

        :return: None
        """
        # Convert data to tensors
        tensor = self.toTensor()

        output = self.model(tensor)
        _, output = torch.max(output, 1)

        output = None

        self.place_forget()  # Hide the current frame properly

        # Create and display the Patient login frame
        OutputFrame(self.master, self, output).place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Place OutputFrame correctly
        


    def loadWeight(self):
        # set the device we will be using to train the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the saved weights into the model
        loaded_model = CNNModel(10)  # Recreate the model architecture
        loaded_model.load_state_dict(torch.load("cnn_model_weights.pth")) # change to file path
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









class OutputFrame(tk.Frame):
    def __init__(self, master, previous_frame, output):
        super().__init__(master)
        self.master = master
        self.previous_frame = previous_frame  # Store the previous frame (login page)
        self.output = output  # Store authenticated user details

        # Add a Back Button to return to login
        back_button = tk.Button(master=self, text="Make new prediction",
                                 command=self.go_back, font=("Microsoft Yahei UI Light", 11), fg='#fff',
                                 bg="#347f8c", border=0, width=30)
        back_button.pack()

    def go_back(self):
        self.place_forget()  # Hide the OutputFrame properly
        self.previous_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Show InputFrame again





if __name__ == "__main__":
    root = tk.Tk()
    root.title("Login")
    root.geometry("900x540")
    root.resizable(width=False, height=False)
    root.configure(bg="#ffffff")
    InputFrame(root).place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    root.mainloop()
