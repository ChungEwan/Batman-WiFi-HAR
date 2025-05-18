from tkinter import *
import customtkinter as ctk
import os
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import PercentFormatter

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
        self.top_container = None
        self.chart_widget = None
        self.bar_widget = None
        self.image_label = None

        self.showing_about = False  # Toggle state

        # Frame for original content
        self.main_content_frame = ctk.CTkFrame(self)
        self.main_content_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.original_content()

        # Frame for ABOUT content
        self.about_frame = ctk.CTkScrollableFrame(self, width=500)
        self.about_frame.grid(row=0, column=0, sticky="nsew", padx=(0,10), pady=10)
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
        self.label.grid(row=2, column=0, columnspan=2, pady=(10, 2), sticky="n")

    def about_content(self):
        """
        Fill the main content frame with the "About" of the system.
        """
        self.about_frame.grid_rowconfigure(0, weight=0)  # Title (ABOUT)
        self.about_frame.grid_rowconfigure(1, weight=0)  # Orientation
        self.about_frame.grid_rowconfigure(2, weight=0)  # Image
        self.about_frame.grid_rowconfigure(3, weight=0)  # Text for orientation
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
        self.label.configure(text=f"Predicted Activity: {self.proper_name(self.output)}")
        self.label.grid(row=1, column=0, pady=0, sticky="s")

        # Generate and display the chart
        self.display_chart(self.output_probs)

        # Remove about frame if it is open
        if self.showing_about:
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
        self.top_container.grid(row=3, column=0, columnspan=2, pady=1, sticky="n")
        self.top_container.grid_columnconfigure(0, weight=1)
        self.top_container.grid_columnconfigure(1, weight=1)

        # Donut chart
        fig_donut, ax_donut = plt.subplots(figsize=(3, 3))
        fig_donut.patch.set_facecolor("#2b2b2b")
        ax_donut.set_facecolor('#2b2b2b')

        highest_label, highest_value = highest
        remaining = 100 - highest_value
        ax_donut.pie([highest_value, remaining],
                    colors=['#71D191', '#EAEAEA'],
                    startangle=90,
                    counterclock=True,
                    wedgeprops={'width': 0.3},
                    textprops={'color': 'white'})
        # ax_donut.set_title(f"Confidence: {highest_value:.1f}%", fontsize=14, color="white")
        ax_donut.text(0, 0, f"{highest_value:.1f}%", ha='center', va='center', fontsize=16, color='white', fontweight='bold')

        # move to left
        chart_canvas = FigureCanvasTkAgg(fig_donut, master=self.top_container)
        chart_canvas.draw()
        self.chart_widget = chart_canvas.get_tk_widget()
        self.chart_widget.grid(row=0, column=1, padx=5, pady=(0, 5), sticky="n")

        # Action image to left
        action_image_path = os.path.join(self.script_dir, "Logo", f"{self.output}.png")
        if os.path.exists(action_image_path):
            img = Image.open(action_image_path)
            # Convert to CTkImage
            ctk_image = ctk.CTkImage(light_image=img, size=(180, 180))
            self.image_label = ctk.CTkLabel(self.top_container, image=ctk_image, text="")
            self.image_label.grid(row=0, column=0, padx=6, pady=1, sticky="nsew")

        # Bar chart
        fig_bar, ax_bar = plt.subplots(figsize=(7, 2))
        fig_bar.patch.set_facecolor("#2b2b2b")
        ax_bar.set_facecolor('#2b2b2b')
        if top_3:
            labels = [item[0] for item in top_3][::-1]
            # Apply renaming function to each label
            labels = [self.proper_name(label) for label in labels]
            values = [item[1] for item in top_3][::-1]
            ax_bar.barh(labels, values, color=["#d46954", "#d4c754", "#8ed454"], height=0.3)
            ax_bar.set_xlim(0, top_3[0][1] * 1.2)
            ax_bar.set_xlabel("Percentage", color="white")
            ax_bar.set_title("Next Top 3 Activities", fontsize=14, color="white", fontweight='bold')
            ax_bar.tick_params(colors="white")
            ax_bar.xaxis.set_major_formatter(PercentFormatter(xmax=100))  # shows % symbol

        for spine in ax_bar.spines.values():
            spine.set_visible(False)

        bar_canvas = FigureCanvasTkAgg(fig_bar, master=self.main_content_frame)
        bar_canvas.draw()
        self.bar_widget = bar_canvas.get_tk_widget()
        self.bar_widget.grid(row=4, column=0, columnspan=2, pady=(0, 40), sticky="n")

    def handle_prediction_error(self, tab, error):
        # Clear all widgets first
        for widget in self.main_content_frame.winfo_children():
            widget.destroy()

        # Remove about frame if it is open
        if self.showing_about:
            self.showing_about = False
            self.about_frame.grid_remove()
            self.main_content_frame.grid()

        # Reset UI content to original state
        self.original_content()

        # Update label for error message
        self.label.configure(text="Please insert the compatible CSI data!")

    def proper_name(self, name):
        if name == "clap":
            return "Clap"
        elif name == "jump":
            return "Jump"
        elif name == "rubhand":
             return "Rubhand"
        elif name == "nopeople":
            return "No people"
        elif name == "squatting":
            return "Squatting"
        elif name == "waving":
            return "Waving"
        elif name == "pushpull":
            return "Push & Pull"
        elif name == "standing":
            return "Standing"
        elif name =="punching":
            return "Punching"
        elif name == "twist":
            return "Twist"

