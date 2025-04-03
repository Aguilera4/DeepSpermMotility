import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import requests
import time
import sperm_video_classify
import pandas as pd

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
        
        
        # Etiqueta de estado
        self.status_label = tk.Label(root, text="Selecciona video a analizar", fg="blue")
        self.status_label.pack()

        # Lienzo para mostrar el video
        self.canvas = tk.Canvas(root, width=500, height=300, bg="black")
        self.canvas.pack()

        self.video_path = None
        self.cap = None
        self.running = False

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
        
            
        sperm_video_classify.classify_video(self.video_path,self.get_test_name())

        self.status_label.config(text=f"Reproduciendo: {prueba_nombre}", fg="green")
        self.running = True
        self.play_video()

    def play_video(self):
        if not self.video_path:
            return
        
        # Load the tracking data with velocity
        df = pd.read_csv('../results/video_predicted/centroid_velocity/centroid_velocity_' + self.get_test_name() + '.csv')
        trajectories = {}
        
        self.cap = cv2.VideoCapture(self.video_path)
        self.replay_button.config(state=tk.DISABLED)  # Desactivar botón mientras se reproduce
        
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))  # Obtener FPS del video
        max_frames = fps * 5  # Máximo de frames a reproducir (5 segundos)

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
