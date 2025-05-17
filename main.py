from tkinter import *
import customtkinter as ctk

from model_frame import ModelFrame
from results_frame import ResultsFrame

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