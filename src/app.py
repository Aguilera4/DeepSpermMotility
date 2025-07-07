import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import requests
import time
import pandas as pd
import torch
from sort.sort import *
from classify_by_movement import *
from functions_features import *
from preprocessing import *
import joblib


fps = 50
    
class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Uploader")
        self.root.geometry("600x500")
        
        # Entrada para el nombre de la prueba
        self.name_label = tk.Label(root, text="Nombre de la Prueba:")
        self.name_label.pack()
        self.name_entry = tk.Entry(root, width=40)
        self.name_entry.pack(pady=5)

        # Botón para seleccionar video
        self.select_button = tk.Button(root, text="Seleccionar Video", command=self.select_video)
        self.select_button.pack(pady=10)
        
        # Botón para iniciar el proceso
        self.start_button = tk.Button(root, text="Iniciar", command=self.start_process, state=tk.DISABLED)
        self.start_button.pack(pady=10)


        # Botón para reproducir de nuevo
        self.replay_button = tk.Button(root, text="Reproducir de nuevo", command=self.play_video, state=tk.DISABLED)
        self.replay_button.pack(pady=10)
        
        self.stop_button = tk.Button(root, text="⏹️ Detener", command=self.stop_process, state=tk.DISABLED)
        self.stop_button.pack(pady=10)
        self.running = False
        
        # Etiqueta de estado
        self.status_label = tk.Label(root, text="Selecciona video a analizar", fg="blue")
        self.status_label.pack()

        # Lienzo para mostrar el video
        self.canvas = tk.Canvas(root, width=500, height=300, bg="black")
        self.canvas.pack()

        self.video_path = None
        self.cap = None
        self.running = False
        
    def traking_video(self):# Video capture
        # Load the YOLOv5 model from the checkpoint
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='../YOLO_model/best_yolov5s.pt')
        
        cap = cv2.VideoCapture(self.video_path)

        # Initialize the tracking algorithm
        tracker = Sort(max_age=15, min_hits=20, iou_threshold=0.1)

        # Initialize variables
        tracking_data = []
        trajectories = {}
        frame_id = 0

        # Process the video frame by frame
        def process():
            nonlocal frame_id
            while cap.isOpened() and app.running:
                ret, frame = cap.read()
                
                if not ret or frame_id > 1500:
                    break

                results = model(frame)
                bbox_data = results.pandas().xyxy[0]
                detections = bbox_data[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].values
                labels = bbox_data['class'].values

                tracks = tracker.update(detections)

                for idx, track in enumerate(tracks):
                    xmin, ymin, xmax, ymax, track_id = track
                    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
                    tracking_data.append([frame_id, track_id, labels[idx], cx, cy, xmin, ymin, xmax, ymax])

                    # Dibujar bbox
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 1)
                    cv2.putText(frame, f'ID {int(track_id)}', (int(xmin), int(ymin)-10), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1,  cv2.LINE_AA )
                    
                    if track_id not in trajectories:
                        trajectories[track_id] = []
                    trajectories[track_id].append((cx, cy))
                    
                    # Draw path
                    for i in range(1, len(trajectories[track_id])):
                        cv2.line(frame, (int(trajectories[track_id][i - 1][0]),int(trajectories[track_id][i - 1][1])), (int(trajectories[track_id][i][0]),int(trajectories[track_id][i][1])), (0, 255, 0), 1)
                                    
                # Mostrar en canvas
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.resize(rgb_frame, (500, 300))
                img = ImageTk.PhotoImage(Image.fromarray(rgb_frame))

                self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
                self.canvas.image = img  # mantener referencia

                frame_id += 1
                root.update_idletasks()
                time.sleep(1/fps)

            cap.release()
            self.status_label.config(text="Tracking finalizado", fg="blue")

            # Guardar CSV
            df = pd.DataFrame(tracking_data, columns=['frame_id', 'track_id', 'class', 'cx', 'cy', 'xmin', 'ymin', 'xmax', 'ymax'])
            df.to_csv('../results/video_predicted/tracking/tracking_' + self.get_test_name() + '.csv', index=False)

        threading.Thread(target=process, daemon=True).start()
        
    def calculate_centroid_velocity(self):
        # Load the tracking data from a CSV file
        df = pd.read_csv('../results/video_predicted/tracking/tracking_' + self.get_test_name() + '.csv')

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
        df.to_csv('../results/video_predicted/centroid_velocity/centroid_velocity_' + self.get_test_name() + '.csv', index=False)

        print("Velocity data saved to sperm_tracking_with_velocity")
    
    def calculate_features(self):
        # Load the tracking data from a CSV file
        df = pd.read_csv('../results/video_predicted/tracking/tracking_' + self.get_test_name() + '.csv')

        columns = ['sperm_id','total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','lin','wob','str','bcf']
        data = pd.DataFrame(columns=columns)

        # Group by track_id and calculate velocity
        for track_id, group in df.groupby('track_id'):
            if len(group) >= 100:
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
        data.to_csv('../results/video_predicted/features/features_' + self.get_test_name() + '.csv', index=False)
    
    def preprocessing_data(self):  
        # Load the tracking data from a CSV file
        df = pd.read_csv('../results/video_predicted/features/features_' + self.get_test_name() + '.csv')
        
        df = df.drop('sperm_id', axis=1)
        df_cleaned = deleted_null_values(df)
        df_scaler = scaler(df_cleaned)
        #df_cleaned_outliers = iqr_median_impute(df_scaler, exclude_cols=['label'])
        
        df = pd.DataFrame(df_scaler, columns=['total_distance','displacement','time_elapsed','vcl','vsl','vap','alh','mad','lin','wob','str','bcf'])
        
        # Save the updated DataFrame with velocity data
        df.to_csv('../results/video_predicted/preprocessing/' + self.get_test_name() + '_preprocessing.csv', index=False)
        
    def classify_data(self):
        # Load model
        loaded_model = joblib.load('../models/simple_NN_3c.joblib')
        
        # Load data
        df = pd.read_csv('../results/video_predicted/preprocessing/' + self.get_test_name() + '_preprocessing.csv')
        df2 = pd.read_csv('../results/video_predicted/features/features_' + self.get_test_name() + '.csv')
        
        # Delete unused column
        X = df[['vcl', 'vsl', 'vap', 'alh', 'str']]
        
        print(X)
        # Mapping of numeric values to class names
        class_names = {
            0: "Progressive",
            1: "Non-progressive",
            2: "Inmotile"
        }
        
        # Predict
        y_pred = np.argmax(loaded_model.predict(X), axis=1)
        #y_pred=loaded_model.predict(X)
        
        # Replace numeric values with class names
        print(y_pred)
        y_pred_mapped = [class_names[label] for label in y_pred]
        
        # Create a count plot with different colors per class
        ax = sns.countplot(x=y_pred_mapped, palette="Set2")

        # Add the count labels on top of each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        fontsize=12, color='black', 
                        xytext=(0, 5), textcoords='offset points')

        plt.title("Distribution of Sperm Motility Categories")
        plt.xlabel("Categories")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()
        
        
        df2['label'] = y_pred
        
        df2.to_csv("aa.csv")

    def select_video(self):
        """ Permite seleccionar un video """
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
        
        if self.video_path:
            prueba_nombre = self.name_entry.get().strip()
            
            if not prueba_nombre:
                self.status_label.config(text="⚠️ Escribe un nombre para la prueba", fg="red")
                return
            
            self.status_label.config(text="Video seleccionado: OK", fg="green")
            self.start_button.config(state=tk.NORMAL)  # Habilitar botón de iniciar
            self.replay_button.config(state=tk.DISABLED)  # Desactivar botón de reproducir de nuevo

    def get_test_name(self):
        """ Obtiene el nombre ingresado en la caja de texto """
        return self.name_entry.get().strip()
            
    def start_process(self):
        """ Inicia la reproducción del video """
        prueba_nombre = self.name_entry.get().strip()

        if not self.video_path or not prueba_nombre:
            self.status_label.config(text="⚠️ Faltan datos. Ingresa un nombre y selecciona un video.", fg="red")
            return

        self.status_label.config(text=f"Cargando video '{prueba_nombre}'...", fg="orange")
        self.root.update_idletasks()
            
        #sperm_video_classify.classify_video(self.video_path,self.get_test_name())
            
        self.running = True
        self.stop_button.config(state=tk.NORMAL)
        self.start_button.config(state=tk.DISABLED)
        self.replay_button.config(state=tk.DISABLED)
        
        self.traking_video()
        self.calculate_centroid_velocity()
        self.calculate_features()
        self.preprocessing_data()
        self.classify_data()
        
        self.status_label.config(text=f"Reproduciendo: {prueba_nombre}", fg="green")
        self.play_video()
        
    def stop_process(self):
        self.running = False
        self.status_label.config(text="Proceso detenido manualmente ❌", fg="red")
        self.stop_button.config(state=tk.DISABLED)

    def play_video(self):
        if not self.video_path:
            return
        
        # Load the tracking data with velocity
        df = pd.read_csv('../results/video_predicted/centroid_velocity/centroid_velocity_' + self.get_test_name() + '.csv')
        trajectories = {}
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.replay_button.config(state=tk.DISABLED)  # Desactivar botón mientras se reproduce
        
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # Obtener FPS del video
        max_frames = fps * 30  # Máximo de frames a reproducir (5 segundos)

        def update():
            # Process the video frame by frame
            frame_id = 0
            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret or frame_id >= max_frames:
                    break
                
                # Get the data for the current frame
                frame_data = df[df['frame_id'] == frame_id]
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (500, 300))  # Ajustar tamaño

                # Draw velocity vectors on the frame
                for _, row in frame_data.iterrows():
                    cx, cy = int(row['cx']), int(row['cy'])
                    vx, vy = row['velocity_x'], row['velocity_y']

                    # Scale the velocity vector for visualization
                    scale = 0.3  # Adjust this to make the vectors visible
                    end_point = int(cx + vx * scale), int(cy + vy * scale)
                    track_id = row['track_id']
                    if track_id not in trajectories:
                        trajectories[track_id] = []
                    trajectories[track_id].append((cx, cy))
                
                    # Draw path
                    for i in range(1, len(trajectories[track_id])):
                        cv2.line(frame, (int(trajectories[track_id][i - 1][0]),int(trajectories[track_id][i - 1][1])), (int(trajectories[track_id][i][0]),int(trajectories[track_id][i][1])), (0, 255, 0), 2)
                        
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)
                
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.canvas.image = img_tk  # Mantener referencia

                self.root.update_idletasks()
                self.root.after(30)  # Ajuste para mantener fluidez

            self.cap.release()
            self.replay_button.config(state=tk.NORMAL)  # Habilitar botón de reproducir de nuevo
            self.status_label.config(text="Reproducción finalizada", fg="blue")

        threading.Thread(target=update, daemon=True).start()

    def upload_video(self):
        if not self.video_path:
            self.status_label.config(text="¡No hay video seleccionado!", fg="red")
            return

        self.status_label.config(text="Subiendo...", fg="orange")

        def upload():
            url = "https://your-server.com/upload"  # Cambiar por endpoint real
            files = {'file': open(self.video_path, 'rb')}
            response = requests.post(url, files=files)

            if response.status_code == 200:
                self.status_label.config(text="¡Subida exitosa!", fg="green")
            else:
                self.status_label.config(text="Error en la subida", fg="red")

        threading.Thread(target=upload, daemon=True).start()

# Ejecutar la aplicación
root = tk.Tk()
app = VideoApp(root)
root.mainloop()
