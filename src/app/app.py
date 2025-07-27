import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import requests
import time
import pandas as pd
import torch
import joblib
import sys
import os
import warnings
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from sort.sort import *
from classify_by_movement import *
from functions_features import *
from preprocessing import *

warnings.filterwarnings("ignore")

fps = 50
    
class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Uploader")
        self.root.geometry("1000x500")
        
        # Frame container for the row
        self.top_controls_frame = tk.Frame(root)
        self.top_controls_frame.pack(pady=10)
        
        # Entry for the name of the test
        self.name_label = tk.Label(self.top_controls_frame, text="Name of test:")
        self.name_label.pack(side=tk.LEFT, padx=5)
        self.name_entry = tk.Entry(self.top_controls_frame, width=30)
        self.name_entry.pack(side=tk.LEFT, padx=5)

        # Entry for indicate the total time to analysis
        self.name_label = tk.Label(self.top_controls_frame, text="Time analyzed:")
        self.name_label.pack(side=tk.LEFT, padx=5)
        self.analysis_time_entry = tk.Entry(self.top_controls_frame, font=("Arial", 10), width=10)
        self.analysis_time_entry.insert(0, "30")
        self.analysis_time_entry.pack(side=tk.LEFT, padx=5)

        # Entry for indicate the max trajectoy lenght to visualize
        self.name_label = tk.Label(self.top_controls_frame, text="Maximum trajectory displayed:")
        self.name_label.pack(side=tk.LEFT, padx=5)
        self.max_trajectory_displayed = tk.Entry(self.top_controls_frame, font=("Arial", 10), width=10)
        self.max_trajectory_displayed.insert(0, "50")
        self.max_trajectory_displayed.pack(side=tk.LEFT, padx=5)
        
        # Select classes
        self.class_label = tk.Label(self.top_controls_frame, text="Classes:")
        self.class_label.pack(side=tk.LEFT, padx=5)

        self.class_options = ttk.Combobox(self.top_controls_frame, values=["2 classes", "3 classes", "4 classes"], state="readonly", width=10)
        self.class_options.current(0)
        self.class_options.pack(side=tk.LEFT, padx=5)

        # Button to select video
        self.select_button = tk.Button(self.top_controls_frame, text="Select video", command=self.select_video)
        self.select_button.pack(side=tk.LEFT, padx=5)
        
        # Buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(pady=10)

        # Button to init the process
        self.start_button = tk.Button(self.button_frame, text="Start", command=self.start_process, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=5)

        # Button to play again
        self.replay_button = tk.Button(self.button_frame, text="Play again", command=self.play_video, state=tk.DISABLED)
        self.replay_button.pack(side=tk.LEFT, padx=5)
        
        # Button to stop
        self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_process, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Show results
        self.view_results_button = tk.Button(root, text="Show results", command=self.view_results, state=tk.DISABLED)
        self.view_results_button.pack(pady=10)
        
        self.running = False
        
        # Status label
        self.status_label = tk.Label(root, text="Select video to analyze", fg="blue")
        self.status_label.pack()

        # Canvas to show the video
        self.canvas = tk.Canvas(root, width=500, height=300, bg="black")
        self.canvas.pack()
        
        # Time transcurred
        self.time_label = tk.Label(root, text="Time: 0.00 s", font=("Arial", 12), fg="black")
        self.time_label.pack(pady=5)

        self.video_path = None
        self.cap = None
        self.running = False
        
    def traking_video(self):
        # Load the YOLOv5 model from the checkpoint
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='../../YOLO_model/best_yolov5s.pt')
        
        cap = cv2.VideoCapture(self.video_path)

        # Initialize the tracking algorithm
        tracker = Sort(max_age=15, min_hits=20, iou_threshold=0.1)

        # Initialize variables
        tracking_data = []
        trajectories = defaultdict(list)
        frame_id = 0

        # Process the video frame by frame
        def process():
            nonlocal frame_id
            while cap.isOpened() and app.running:
                # Get frame
                ret, frame = cap.read()
                
                if not ret or frame_id > float(self.analysis_time_entry.get()) * 50:
                    break

                results = model(frame) # Detections
                bbox_data = results.pandas().xyxy[0]
                detections = bbox_data[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].values
                labels = bbox_data['class'].values

                tracks = tracker.update(detections) # Updated the tracking algorithm
                
                for (xmin, ymin, xmax, ymax, track_id), label in zip(tracks, labels):
                    cx = (xmin + xmax) * 0.5
                    cy = (ymin + ymax) * 0.5
                    tracking_data.append([frame_id, track_id, label, cx, cy, xmin, ymin, xmax, ymax])
                    
                df = pd.DataFrame(tracking_data, columns=['frame_id', 'track_id', 'class', 'cx', 'cy', 'xmin', 'ymin', 'xmax', 'ymax'])
                df.to_csv('./results/video_predicted/tracking/tracking_' + self.get_test_name() + '.csv', index=False)
                
                predictions = []
                
                try:
                    self.calculate_features()
                except Exception as e:
                    print("Not enough data (features)",e)
                    
                try:
                    self.preprocessing_data()
                except Exception as e:
                    print("Not enough data (preprocessing):", e)
                    
                try:
                    self.classify_data()
                except Exception as e:
                    print("Not enough data (classify):", e)
                    
                try:
                    predictions = pd.read_csv('./results/video_predicted/predictions/' + self.get_test_name() + '.csv')
                except Exception as e:
                    print("Not enough data (predictions):", e)
                
                for (xmin, ymin, xmax, ymax, track_id), label in zip(tracks, labels):
                    cx = (xmin + xmax) * 0.5
                    cy = (ymin + ymax) * 0.5
                        
                    try:
                        label =  int(predictions[predictions['sperm_id'] == track_id]['label'])
                        default_color = (128, 128, 128)  # Gris por defecto

                        if self.class_options.get() == '3 classes':
                            label_colors = {
                                0: (0, 255, 0), # Progressive/Progressive/Rapdly progressive
                                1: (255, 0, 0), # Non progressive/Non progressive/Slowly progressive
                                2: (0, 0, 255), # -/-/Inmotile
                                3: (0, 0, 255) # -/Inmotile/Non progressive
                            }
                        else:
                            label_colors = {
                                0: (0, 255, 0), # Progressive/Progressive/Rapdly progressive  
                                1: (255, 0, 0), # Non progressive/Non progressive/Slowly progressive
                                2: (0, 255, 255), # -/-/Inmotile
                                3: (0, 0, 255) # -/Inmotile/Non progressive
                            }
                        color = label_colors.get(label, default_color)
                    except Exception as e:
                        color = (0, 0, 0)
                    
                    # Draw bbox
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 1)
                    cv2.putText(frame, f'ID {int(track_id)}', (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1,  cv2.LINE_AA )
                    
                    # Save point for trajectory
                    trajectories[track_id].append((cx, cy))
                    
                    # Draw path
                    max_trajectory_lenght = int(self.max_trajectory_displayed.get())
                    points = trajectories[track_id][-max_trajectory_lenght:]
                    for p1, p2 in zip(points, points[1:]):
                        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1)
                          
                # Show canvas
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.resize(rgb_frame, (500, 300))
                img = ImageTk.PhotoImage(Image.fromarray(rgb_frame))

                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
                self.canvas.image = img

                frame_id += 1
                elapsed_time = frame_id / fps
                self.time_label.config(text=f"Time: {elapsed_time:.2f} s")
                root.update_idletasks()

            cap.release()
            self.status_label.config(text="Tracking completed", fg="blue")

            # Save CSV
            df = pd.DataFrame(tracking_data, columns=['frame_id', 'track_id', 'class', 'cx', 'cy', 'xmin', 'ymin', 'xmax', 'ymax'])
            df.to_csv('./results/video_predicted/tracking/tracking_' + self.get_test_name() + '.csv', index=False)
            self.view_results_button.config(state=tk.NORMAL)

        threading.Thread(target=process, daemon=True).start()
        
    def calculate_centroid_velocity(self):
        # Load the tracking data from a CSV file
        df = pd.read_csv('./results/video_predicted/tracking/tracking_' + self.get_test_name() + '.csv')

        # Calculate velocity for each track_id
        df['velocity_x'] = 0.0
        df['velocity_y'] = 0.0
        df['speed'] = 0.0

        # Frame rate of the video (frames per second)
        dt = 1 / fps  # Time interval between frames

        # Group by track_id and calculate velocity
        for track_id, group in df.groupby('track_id'):
            # Calculate displacement (delta x and delta y)
            group['delta_x'] = group['cx'].diff()
            group['delta_y'] = group['cy'].diff()

            # Calculate velocity (pixels per second)
            group['velocity_x'] = np.round(group['delta_x'] / dt,2)
            group['velocity_y'] = np.round(group['delta_y'] / dt,2)

            # Calculate speed (magnitude of velocity)
            group['speed'] = np.round((group['velocity_x']**2 + group['velocity_y']**2)**0.5,2)
            
            
            # Calculate mean and maximum velocity
            group["mean_velocity"] = np.round(group['speed'].mean(),2)
            group["max_velocity"] = np.round(group['speed'].max(),2)
            
            df.loc[group.index, ['mean_velocity', 'max_velocity']] = group[['mean_velocity', 'max_velocity']].fillna(0)
            
            # Update the original DataFrame
            df.loc[group.index, ['velocity_x', 'velocity_y', 'speed']] = group[['velocity_x', 'velocity_y', 'speed']].fillna(0)

        # Save the updated DataFrame with velocity data
        df.to_csv('./results/video_predicted/centroid_velocity/centroid_velocity_' + self.get_test_name() + '.csv', index=False)
    
    
    
    def calculate_features(self):
        # Load the tracking data from a CSV file
        df = pd.read_csv('./results/video_predicted/tracking/tracking_' + self.get_test_name() + '.csv')
        
        columns = ['sperm_id','total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','lin','wob','str','bcf']
        data = pd.DataFrame(columns=columns)

        # Group by track_id and calculate velocity
        for track_id, group in df.groupby('track_id'):
            if len(group) >= 50:
                # Convert the columns to a list of tuples
                trajectory_path = list(zip(group['cx'], group['cy']))
                
                # Basic measures
                time_elapsed = calculate_time_elapsed(trajectory_path,fps)
                displacement = calculate_displacement(trajectory_path)
                total_distance = calculate_total_distance(trajectory_path)

                # Standard measures
                vcl = calculate_VCL(trajectory_path,fps)
                vsl = calculate_VSL(trajectory_path,fps)
                vap = calculate_VAP(trajectory_path,fps)
                alh = calculate_ALH(trajectory_path)
                mad = calculate_MAD(trajectory_path)
                
                # Commonly measures
                linearity = calculate_linearity(trajectory_path,fps)
                wob = calculate_WOB(trajectory_path,fps)
                straightness = calculate_STR(trajectory_path,fps)
                bcf = calculate_BCF(trajectory_path,fps)
                #curvature = calculate_curvature(trajectory_path)

                new_row = pd.DataFrame([[track_id,total_distance,displacement,time_elapsed,vcl,vsl,vap,alh,mad,linearity,wob,straightness,bcf]], columns=data.columns)
                data = pd.concat([data,new_row], ignore_index=True)
                
        # Save the DataFrame
        data.to_csv('./results/video_predicted/features/features_' + self.get_test_name() + '.csv', index=False)
    
    
    
    def preprocessing_data(self):
        # Load the tracking data from a CSV file
        df = pd.read_csv('./results/video_predicted/features/features_' + self.get_test_name() + '.csv')
        
        df = df.drop('sperm_id', axis=1)
        df_cleaned = deleted_null_values(df)
        df_scaler = scaler(df_cleaned)
        #df_cleaned_outliers = iqr_median_impute(df_scaler, exclude_cols=['label'])
        
        df = pd.DataFrame(df_scaler, columns=['total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','lin','wob','str','bcf'])
        
        # Save the updated DataFrame with velocity data
        df.to_csv('./results/video_predicted/preprocessing/' + self.get_test_name() + '_preprocessing.csv', index=False)



    def classify_data(self):
        
        # Load data
        df = pd.read_csv('./results/video_predicted/preprocessing/' + self.get_test_name() + '_preprocessing.csv')
        df2 = pd.read_csv('./results/video_predicted/features/features_' + self.get_test_name() + '.csv')
        
        # Delete unused column
        X = df[['vcl', 'vsl', 'vap', 'lin', 'str']]
        
        # Load model
        selected_option = self.class_options.get()
        if selected_option == '2 classes':
            loaded_model = joblib.load('../../models/random_forest_2c.joblib')
        elif  selected_option == '3 classes':
            loaded_model = joblib.load('../../models/random_forest_3c.joblib')
        elif  selected_option == '4 classes':
            loaded_model = joblib.load('../../models/random_forest_4c.joblib')
        
        # Predict
        #y_pred = np.argmax(loaded_model.predict(X), axis=1)
        y_pred=loaded_model.predict(X)
        
        df2['label'] = y_pred
        df2.to_csv('./results/video_predicted/predictions/' + self.get_test_name() + '.csv', index=False)



    def select_video(self):
        """ Allows you to select a video """
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
        
        if self.video_path:
            name_test = self.name_entry.get().strip()
            
            if not name_test:
                self.status_label.config(text="Enter a name for the test", fg="red")
                return
            
            self.status_label.config(text="Video selected: OK", fg="green")
            self.start_button.config(state=tk.NORMAL)
            self.replay_button.config(state=tk.DISABLED)



    def get_test_name(self):
        """ Gets the name entered in the text box """
        return self.name_entry.get().strip()


            
    def start_process(self):
        """ Starts video playback """
        name_test = self.name_entry.get().strip()

        if not self.video_path or not name_test:
            self.status_label.config(text="Missing data. Enter a name and select a video.", fg="red")
            return

        self.status_label.config(text=f"Loading video '{name_test}'...", fg="orange")
        self.root.update_idletasks()
            
        self.running = True
        self.stop_button.config(state=tk.NORMAL)
        self.start_button.config(state=tk.DISABLED)
        self.replay_button.config(state=tk.DISABLED)
        
        self.traking_video()
        
        self.status_label.config(text=f"Replaying: {name_test}", fg="green")
        self.replay_button.config(state=tk.NORMAL)
        
        
        
    def stop_process(self):
        self.running = False
        self.status_label.config(text="Manually stopped process", fg="red")
        self.stop_button.config(state=tk.DISABLED)



    def play_video(self):
        if not self.video_path:
            return
        
        # Load the tracking data with velocity
        df_predictions = pd.read_csv('./results/video_predicted/predictions/' + self.get_test_name() + '.csv')
        df_tracks = pd.read_csv('./results/video_predicted/tracking/tracking_' + self.get_test_name() + '.csv')
        
        trajectories = defaultdict(list)
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.replay_button.config(state=tk.DISABLED)
        
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        max_frames = fps * float(self.analysis_time_entry.get())  # 30 seconds

        def update():
            # Process the video frame by frame
            frame_id = 0
            while self.running and self.cap.isOpened():
                # Get frame
                ret, frame = self.cap.read()
                
                if not ret or frame_id >= max_frames:
                    break
                
                # Get the data for the current frame
                frame_data = df_tracks[df_tracks['frame_id'] == frame_id]

                # Draw velocity vectors on the frame
                for _, row in frame_data.iterrows():
                    cx, cy = int(row['cx']), int(row['cy'])
                    xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                        
                    try:
                        label = int(df_predictions[df_predictions['sperm_id'] == track_id]['label'])
                        default_color = (128, 128, 128)  # Gris por defecto

                        if self.class_options.get() == '3 classes':
                            label_colors = {
                                0: (0, 255, 0), # Progressive/Progressive/Rapdly progressive
                                1: (255, 0, 0), # Non progressive/Non progressive/Slowly progressive
                                2: (0, 0, 255), # -/-/Inmotile
                                3: (0, 0, 255) # -/Inmotile/Non progressive
                            }
                        else:
                            label_colors = {
                                0: (0, 255, 0), # Progressive/Progressive/Rapdly progressive  
                                1: (255, 0, 0), # Non progressive/Non progressive/Slowly progressive
                                2: (0, 255, 255), # -/-/Inmotile
                                3: (0, 0, 255) # -/Inmotile/Non progressive
                            }
                        color = label_colors.get(label, default_color)
                    except Exception as e:
                        color = (0, 0, 0)
                    
                    track_id = row['track_id']
                    
                    # Draw bbox
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 1)
                    cv2.putText(frame, f'ID {int(track_id)}', (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1,  cv2.LINE_AA )
                    
                    # Save point for trajectory
                    trajectories[track_id].append((cx, cy))
                
                    # Draw path
                    points = trajectories[track_id]
                    for p1, p2 in zip(points, points[1:]):
                        cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 1)

                # Show canvas
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.resize(rgb_frame, (500, 300))
                img = ImageTk.PhotoImage(Image.fromarray(rgb_frame))

                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
                self.canvas.image = img

                frame_id += 1
                elapsed_time = frame_id / fps
                self.time_label.config(text=f"Time: {elapsed_time:.2f} s")
                root.update_idletasks()

            self.cap.release()
            self.replay_button.config(state=tk.NORMAL)
            self.status_label.config(text="Reproduction completed", fg="blue")

        threading.Thread(target=update, daemon=True).start()
        
    def view_results(self):
        
        df = pd.read_csv('./results/video_predicted/predictions/' + self.get_test_name() + '.csv')
        
        selected_option = self.class_options.get()
        if selected_option == '2 classes':
            class_names = ["Progressive", "Non-progressive"]
            color = ['green', 'red']
        elif  selected_option == '3 classes':
            class_names = ["Progressive", "Non-progressive", "Inmotile"]
            color = ['green', 'blue', 'red']
        elif  selected_option == '4 classes':
            class_names = ["Rapidly progressive", "Slowdly progressive", "Non-progressive", "Inmotile"]
            color = ['green', 'blue', 'yellow', 'red']
        
        # Replace numeric values with class names
        y_pred = df['label']
        y_pred_mapped = [class_names[label] for label in y_pred]
        counts = pd.Series(y_pred_mapped).value_counts().sort_index().tolist()
        
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Resultados - Gráficos")
        
        # Create a count plot with different colors per class
        fig, ax = plt.subplots(figsize=(5, 4))

        # Add the count labels on top of each bar
        ax.bar(class_names, counts, color=color)

        ax.set_title("Distribution of Sperm Motility Categories")
        ax.set_xlabel("Categories")
        ax.set_ylabel("Count")
        
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack()

        '''results_file = "output/results.csv"
        if os.path.exists(results_file):
            try:
                # Para Windows
                os.startfile(results_file)
            except AttributeError:
                # Para Linux/Mac
                subprocess.call(["xdg-open", results_file])
        else:
            messagebox.showwarning("Archivo no encontrado", "No se encontró el archivo de resultados.")'''

    def upload_video(self):
        if not self.video_path:
            self.status_label.config(text="No video selected!", fg="red")
            return

        self.status_label.config(text="Uploading...", fg="orange")

        def upload():
            url = "https://DeepSpermMotility.com/upload"
            files = {'file': open(self.video_path, 'rb')}
            response = requests.post(url, files=files)

            if response.status_code == 200:
                self.status_label.config(text="Successful upload!", fg="green")
            else:
                self.status_label.config(text="Error in upload", fg="red")

        threading.Thread(target=upload, daemon=True).start()



# Running the application
root = tk.Tk()
app = VideoApp(root)
root.mainloop()