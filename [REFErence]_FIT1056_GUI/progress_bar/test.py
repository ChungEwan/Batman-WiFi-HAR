import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as py

def visualize_progress_plotly_for_tkinter(data):
    if not data:
        print("No data provided.")
        return

    sorted_data = sorted(data.items(), key=lambda item: item[1], reverse=True)

    # Circular progress for the highest value
    fig = make_subplots(rows=2, cols=1, row_heights=[0.3, 0.7],
                        specs=[[{'type':'pie'}], [{'type':'xy'}]]) # Specify subplot types

    if sorted_data:
        highest_key, highest_value = sorted_data[0]
        total_highest = sum(data.values()) if data else 1
        percentage_highest = (highest_value / total_highest) * 100

        fig.add_trace(go.Pie(labels=['Progress', ''], # Empty label for 'Remaining'
                             values=[percentage_highest, 100 - percentage_highest],
                             hole=0.8,
                             textinfo='percent',
                             hoverinfo='value+percent',
                             marker_colors=['#FF69B4', 'rgba(0,0,0,0)'], # Pink and transparent
                             showlegend=False,
                             name=highest_key), row=1, col=1)

        fig.update_layout(annotations=[dict(text=f"{highest_key}<br>{percentage_highest:.1f}%",
                                            x=0.5, y=0.5, font_size=12, showarrow=False)],
                          title_text="Activity Progress")

    # Horizontal bar charts for the next top 3 values
    if len(sorted_data) > 1:
        top_3 = sorted_data[1:min(4, len(sorted_data))]
        bar_labels = [item[0] for item in top_3][::-1] # Reverse to show highest at the top
        bar_values = [item[1] for item in top_3][::-1]
        total_bar = sum(data.values()) if data else 1
        bar_percentages = [(val / total_bar) * 100 for val in bar_values]
        bar_colors = ['#ADD8E6', '#FFFFE0', '#DDA0DD'] # Light blue, light yellow, light purple

        fig.add_trace(go.Bar(x=bar_percentages,
                             y=bar_labels,
                             orientation='h',
                             text=[f"{p:.1f}%" for p in bar_percentages],
                             textposition='outside',
                             marker_color=bar_colors[:len(bar_labels)],
                             showlegend=False,
                             name='Top Activities'), row=2, col=1)

        fig.update_layout(yaxis=dict(autorange="reversed", title="Activity"), # Reverse y-axis
                          xaxis_title="Percentage")

    fig.update_layout(height=600, showlegend=False) # Remove overall legend

    # Save as PNG with transparent background
    py.offline.plot(fig, filename='activity_progress.png', auto_open=False, image_width=800, image_height=600, config={'displayModeBar': False})
    print("Plot saved to activity_progress.png")

# Example dictionary (replace with your actual data)
my_data = {"Clap": 60, "Jump": 90, "No People": 45, "Punching": 75, "Rub Hand": 30}
visualize_progress_plotly_for_tkinter(my_data)