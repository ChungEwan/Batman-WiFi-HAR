import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def display_progress_chart(data, root):
    """
    Display a progress chart with a circular plot for the highest value
    and a horizontal bar graph for the next top 3 values in a Tkinter window.

    :param data: Dictionary with activity names as keys and values as scores.
    :param root: Tkinter root or frame to display the chart.
    """
    # Sort data in descending order
    sorted_data = sorted(data.items(), key=lambda item: item[1], reverse=True)

    # Extract the highest value and the next top 3 values
    highest = sorted_data[0]
    top_3 = sorted_data[1:4]

    # Create a matplotlib figure
    fig, axes = plt.subplots(2, 1, figsize=(5, 7), gridspec_kw={'height_ratios': [2, 1]})
    fig.subplots_adjust(hspace=0.5)

    # Circular plot for the highest value
    highest_label, highest_value = highest
    remaining = 100 - highest_value
    axes[0].pie([highest_value, remaining], labels=[f"{highest_label} ({highest_value}%)", ""],
                colors=['#71D191', '#EAEAEA'], startangle=90, counterclock=False, wedgeprops={'width': 0.3})
    axes[0].set_title("Highest Activity", fontsize=14)

    # Horizontal bar graph for the next top 3 values
    if top_3:
        labels = [item[0] for item in top_3][::-1]  # Reverse the order of labels
        values = [item[1] for item in top_3][::-1]  # Reverse the order of values
        axes[1].barh(labels, values, color=['#ADD8E6', '#FFFFE0', '#DDA0DD'])
        axes[1].set_xlim(0, 100)
        axes[1].set_xlabel("Percentage")
        axes[1].set_title("Top 3 Activities", fontsize=14)

    # Embed the matplotlib figure into Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Example usage
if __name__ == "__main__":
    # Example data
    example_data = {"Clap": 60, "Jump": 90, "No People": 45, "Punching": 75, "Rub Hand": 30}

    # Create a Tkinter window
    root = tk.Tk()
    root.title("Progress Chart")
    root.geometry("600x800")

    # Display the progress chart
    display_progress_chart(example_data, root)

    # Run the Tkinter main loop
    root.mainloop()