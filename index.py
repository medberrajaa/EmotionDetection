
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
from moviepy.editor import VideoFileClip
import threading
import pygame
import textwrap
import tkinter.font as tkFont
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import shutil
from keras.models import Sequential, load_model
import time
import traitement_audio as aud
import keras
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import get_class as get_class


# Créer un modèle séquentiel et Chargement du modèle


def load_keras_model(model_path):
    try:
        # Créer un modèle séquentiel
        model = Sequential()

        # Ajouter les couches une par une en utilisant les configurations du modèle sauvegardé
        model.add(InputLayer(batch_input_shape=(None, 128, 128, 1)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(6, activation='softmax'))

        # Charger les poids du modèle sauvegardé
        model.load_weights(model_path)

        print("Le modèle a été importé avec succès.")
        return model
    except Exception as e:
        print(f"Erreur lors de l'importation du modèle : {e}")
        return None
# Chemin vers le modèle
model_path = 'model/audiooo.keras'
# Chargement du modèle
model_audio = load_keras_model(model_path)






# root

root = tk.Tk()
root.title("Analyseur de sensation")
root.grid_columnconfigure(0, weight=1)
root.geometry("+60+10")
root.configure(bg="red")
root.iconphoto(True, tk.PhotoImage(file='img/logo.png'))

background_image = Image.open("img/backround2.jpg")
background_image_resized = background_image.resize((1465, 660), Image.LANCZOS) #1160 650
background_photo = ImageTk.PhotoImage(background_image_resized)

background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)



canvas_width, canvas_height = 600, 400
video = None
audio = None
clip=None
cnt=0
is_playing = False
current_frame = None
playing = False
paused = False
total_frames = None
btn_play = None
btn_pause = None
scale_position = None
label_position = None
fps = None
label_time = None
label_path=None
zone_text_video=None
zone_text_audio=None
zone_text_camera=None
chk_state_video = tk.BooleanVar()
chk_state_audio = tk.BooleanVar()
font_title = tkFont.Font(family="Arial", size=12, weight="bold", slant="roman")
font_button = tkFont.Font(family="Arial", size=8, weight="bold", slant="roman")
font_text = tkFont.Font(family="Roboto", size=9, weight="bold", slant="roman")
bg1="#F1FDF3"
bg2="#D1E9D2"
bg3="#99CDA9"
root.configure(bg=bg1)
cap = None
cnt2=0
canvas2=None
button_confirme=None
# Initialisation du classificateur Haar pour la détection de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)







# center la fenetre
def center_window(root, width, height):
    # Obtenez la taille de l'écran
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Calculez la position de la fenêtre pour la centrer
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    
    # Définissez les dimensions et la position de la fenêtre
    root.geometry(f"{width}x{height}+{x}+{y-36}")






# afficher home 
def show_home():
    global cnt
    global cap
    
    # Fermer la capture de la caméra
    if cap is not None:
        cap.release()
        cap = None
    if cnt > 0:
        reset()
        vider_dossier()

    center_window(root, 350, 450)
    root.resizable(False,False)
    clear_widgets()

    # Label text
    text_label = tk.Label(root, text="analyse des sentiments", font=("Arial", 16,"bold"),background=bg1,borderwidth=3, relief="solid", highlightbackground=bg3, highlightthickness=2)
    text_label.pack(pady=20)
    
    # Image
    img = Image.open("img/logo2.png")
    # img = img.resize((100, 100), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    
    img_label = tk.Label(root, image=img,background=bg2,borderwidth=2, relief="solid", highlightbackground=bg3, highlightthickness=2)
    img_label.image = img  # Keep a reference to avoid garbage collection
    img_label.pack(pady=20)
    import_vid_button.pack(pady=20)
    camera_button.pack(pady=20)





# afficher page import video
def show_video_import():
    center_window(root, 1160, 640)
    root.resizable(False,False)
    clear_widgets()
    create_widgets()
    create_widgets_result()
    create_widgets_valider()




# afficher page camera
def show_camera():
    global start_time,duration
    # Démarrage du compteur de temps
    start_time = time.time()
    duration = 150  # Durée de la détection en secondes
    center_window(root, 1160, 640)
    root.resizable(False,False)
    clear_widgets()
    create_widgets_camera()

import_vid_button = tk.Button(root, text="Importer un video", font=font_button,padx=15,pady=10,bg=bg2,command=show_video_import)
camera_button = tk.Button(root, text="Ouvrir la caméra", font=font_button,padx=15,pady=10,bg=bg2,command=show_camera)

def clear_widgets():
    for widget in root.winfo_children():
        widget.pack_forget()
        widget.grid_forget()


# afficher live cam avec prediction
def show_camera_feed():
    global canvas2, cap, face_cascade, start_time, duration,zone_text_camera
  
    if cap != None:
        ret, frame = cap.read()
        if ret:
            # Conversion de l'image en niveaux de gris pour la détection de visages
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Détection des visages dans l'image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Dessiner des rectangles autour des visages détectés
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                prediction, proba = get_class.get_class(Image.fromarray(frame[y:y+h, x:x+w]))
                (text_w, text_h), baseline = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.putText(frame, prediction, (x, y-baseline-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image)
            imgtk = ImageTk.PhotoImage(image=pil_image)
            # Convertir l'image OpenCV en image PIL

            # Afficher l'image dans le canvas
            canvas2.create_image(0, 0, anchor=tk.NW, image=imgtk)
            canvas2.imgtk = imgtk  # Garder une référence de l'image pour éviter la collecte des ordures

        # Calculer le temps restant
        elapsed_time = time.time() - start_time
        remaining_time = max(0, duration - elapsed_time)
        # Vérifier la durée spécifiée
        if time.time() - start_time > duration:   
            show_home()
        # Mettre à jour le texte avec le temps restant
        zone_text_camera.configure(text=f"Temps restant: {int(remaining_time)} secondes")
        # Appeler cette fonction à nouveau après 10 ms
        root.after(10, show_camera_feed)









# creation des widgets_camera
def create_widgets_camera():
    global canvas2, cap,zone_text_camera
    
    camera_frame = tk.LabelFrame(root, text="Caméra", font=font_title, bg=bg2, borderwidth=2, relief="solid", highlightbackground=bg3, highlightthickness=2)
    camera_frame.pack(pady=20, padx=20)

    canvas2 = tk.Canvas(camera_frame, width=canvas_width, height=canvas_height, bg=bg2, borderwidth=2, relief="solid", highlightbackground=bg3, highlightthickness=2)
    canvas2.pack()

    Retour_button2 = tk.Button(camera_frame, width=30, text="Retour", font=font_button, bg=bg3, command=show_home)
    Retour_button2.pack(pady=15, padx=20)
    
    #res camera
    zone_text_camera = tk.Label(root,width=50, font=font_text, borderwidth=2, relief="solid", bg=bg2, highlightbackground="#cccccc", highlightthickness=2)
    zone_text_camera.pack(padx=20, pady=35,expand=True)
    texte_camera = "Aucun résultat actuellement"
    zone_text_camera.configure(text=texte_camera)

    # Initialiser la capture de la caméra
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra.")
        zone_text_camera.configure(text="Erreur: Impossible d'ouvrir la caméra.")
        return
    
    # Démarrer l'affichage de la caméra
    show_camera_feed()









# creation des widgets_import_video_1
def create_widgets():
    global total_frames, label_time, label_path

    video_frame = tk.LabelFrame(root, text="Lecteur vidéo",font=font_title,bg=bg2, borderwidth=2, relief="solid", highlightbackground=bg3, highlightthickness=2)
    video_frame.grid(row=0,column=0,pady=20,padx=20, sticky="ew")

    global canvas, btn_pause, btn_play
    canvas = tk.Canvas(video_frame, width=canvas_width, height=canvas_height,bg=bg2, borderwidth=2, relief="solid", highlightbackground=bg3, highlightthickness=2)
    canvas.pack()

    btn_frame = tk.LabelFrame(root,text="Controleur video",font=font_title,bg=bg2, borderwidth=2, relief="solid", highlightbackground=bg3, highlightthickness=2)
    btn_frame.grid(row=1,column=0,pady=(5,10),padx=20)
    
    message_font = tkFont.Font(family="Helvetica", size=10, slant="italic")
    label_path = tk.Label(btn_frame, text="aucun vidéo n'est sélectionné", font=message_font,padx=10,bg=bg2)
    label_path.grid(row=2, column=0, columnspan=5, padx=10,pady=(10,0),sticky="s")

    label_time = tk.Label(btn_frame, text="00:00:00/00:00:00",font=font_button,bg=bg2)
    label_time.grid(row=0, column=0, columnspan=5, padx=10)

    btn_open = tk.Button(btn_frame, text="Ouvrir", command=open_file,font=font_button,bg=bg1)
    btn_open.grid(row=1, column=0, padx=10)

    btn_play = tk.Button(btn_frame, text="Reprendre", state="disabled", command=play_media,font=font_button,bg=bg1)
    btn_play.grid(row=1, column=1, padx=10)

    btn_pause = tk.Button(btn_frame, text="Pause", state="disabled", command=pause_media,font=font_button,bg=bg1)
    btn_pause.grid(row=1, column=2, padx=10)

    label_volume=tk.Label(btn_frame,text="Volume",font=font_button,bg=bg2)
    label_volume.grid(row=1, column=3, padx=(10,0))

    scale_volume = tk.Scale(btn_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=set_volume,bg=bg2)
    scale_volume.set(50)
    scale_volume.grid(row=1, column=4, padx=3,pady=(0,12))







# creation des widgets_import_video_2
def create_widgets_result():
    global zone_text_video,zone_text_audio,button_confirme

    video_audio_res=tk.LabelFrame(root, text="Résultat",font=font_title,bg=bg2, borderwidth=2, relief="solid", highlightbackground=bg3, highlightthickness=2)
    video_audio_res.grid(row=0,column=1,pady=20,padx=20, sticky="ew")

    video_frame_res = tk.LabelFrame(video_audio_res, text="Résultat vidéo (Un seul visage)",font=font_title,bg=bg2, borderwidth=1, relief="solid", highlightbackground=bg3, highlightthickness=2)
    video_frame_res.grid(row=0,column=0,pady=35,padx=25)

    audio_frame_res = tk.LabelFrame(video_audio_res, text="Résultat audio (La totalité d'audio)",font=font_title,bg=bg2, borderwidth=1, relief="solid", highlightbackground=bg3, highlightthickness=2)
    audio_frame_res.grid(row=1,column=0,pady=35,padx=25)

    #res video
    zone_text_video = tk.Label(video_frame_res,width=50, font=font_text, borderwidth=2, relief="solid", bg=bg1, highlightbackground="#cccccc", highlightthickness=2)
    zone_text_video.pack(padx=20, pady=20,expand=True)
    texte_vide1 = "Aucun résultat actuellement "
    zone_text_video.configure(text=texte_vide1)

    #res audio
    zone_text_audio = tk.Label(audio_frame_res,width=50, font=font_text, borderwidth=2, relief="solid", bg=bg1, highlightbackground="#cccccc", highlightthickness=2)
    zone_text_audio.pack(padx=20, pady=20,expand=True)
    texte_vide2 = "Aucun résultat actuellement"
    zone_text_audio.configure(text=texte_vide2)







# creation des widgets_import_video_3
def create_widgets_valider():
    global chk_state_video,chk_state_audio,button_confirme

    video_audio_inclue=tk.LabelFrame(root, text="",font=font_title,bg=bg2, borderwidth=2, relief="solid", highlightbackground=bg3, highlightthickness=2)
    video_audio_inclue.grid(row=1,column=1)

    video_audio_inclue2=tk.LabelFrame(video_audio_inclue, text="Conclusion des sensations",font=font_title,bg=bg2, borderwidth=2, relief="solid", highlightbackground=bg3, highlightthickness=2)
    video_audio_inclue2.pack(pady=1,padx=10)
    

    chk2_audio = tk.Checkbutton(video_audio_inclue2, text="Audio", variable=chk_state_audio,font=font_button,bg=bg2)
    chk2_audio.grid(row=0,column=0,columnspan=2, padx=5, pady=5)
    
    btn_conf_font = tkFont.Font(family="Arial", size=12, weight="bold", slant="roman")
    button_confirme=tk.Button(video_audio_inclue2,width=30,text="Confirmer",font=btn_conf_font,bg=bg3,command=confirme)
    button_confirme.config(state="disabled")
    button_confirme.grid(row=1,column=0,columnspan=2,pady=5,padx=5)

    Retour_button = tk.Button(video_audio_inclue,width=30, text="Retour",font=btn_conf_font,bg=bg3,command=show_home)
    Retour_button.pack(pady=5,padx=5)






# saut de ligne dans la label 
def update_label_with_wrapped_text(label, text, width):
    wrapped_text = textwrap.fill(text, width=width)
    label.config(text=wrapped_text)







# ouvrir un video 
def open_file():
    global video, audio, btn_play, btn_pause, total_frames, fps, scale_position, label_path,clip,cnt,button_confirme
    
    if cnt != 0:
        reset()
        vider_dossier()

    file_path = filedialog.askopenfilename(title="Ouvrir une vidéo", filetypes=[("Fichiers vidéo", "*.mp4;*.avi")])
    if file_path:
        # label_path.config(text=f"Vidéo sélectionnée: {file_path}")  
        update_label_with_wrapped_text(label_path, f"Vidéo sélectionnée: {file_path}", width=95)

        btn_pause.configure(state="active")
        button_confirme.config(state="active")

        video = cv2.VideoCapture(file_path)
        clip = VideoFileClip(file_path)

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))

        if os.path.exists("temp/audio_temporaire.mp3"):
            os.remove("temp/audio_temporaire.mp3")

        audio = clip.audio
        audio.write_audiofile("temp/audio_temporaire.mp3")

        btn_play.config(state="disabled")
        btn_pause.config(state="active")

        threading.Thread(target=play_pause).start()
        threading.Thread(target=play_video).start()

        cnt +=1

            
    



# Réinitialiser
def reset():
    global video, audio, clip, is_playing, playing,scale_position, paused, current_frame, total_frames, fps, btn_play, btn_pause,label_time,label_path,zone_text_video,zone_text_audio

    # Arrêter la lecture de la vidéo et de l'audio
    pause_video()
    pygame.mixer.quit()
    # Réinitialiser les variables et états
    video=None
    audio=None
    clip=None
    is_playing = False
    playing = False
    paused = False
    scale_position=None
    current_frame = None
    total_frames = None
    fps = None
    label_time.config(text="00:00:00/00:00:00")
    label_path.config(text="aucun vidéo n'est sélectionné")
    zone_text_audio.config(text="Aucun résultat actuellement")
    zone_text_video.config(text="Aucun résultat actuellement")
    pygame.mixer.quit()  


    



# sync video & audio *******************************************************

def play_video():
    global is_playing
    if video is not None and not is_playing:
        is_playing = True
        update_video(canvas)

        

def boucle():
     global video
     video.set(cv2.CAP_PROP_POS_FRAMES, 0)
     if is_playing:
            update_video(canvas)


def update_video(canvas):
    global is_playing, current_frame, video,playing,cnt2,zone_text_video
    
    if video:
        ret, frame = video.read()
        if ret:
            if(int(cnt2)%50 == 0):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    prediction, proba = get_class.get_class(Image.fromarray(frame[y:y+h, x:x+w]))
                    zone_text_video.config(text=f"{prediction} : {proba:.3f}")

            cnt2 += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, _ = frame.shape
            scale_factor = min(canvas_width / width, canvas_height / height)
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            
            current_frame = ImageTk.PhotoImage(image=Image.fromarray(frame))
            canvas.create_image((canvas_width - frame.shape[1]) // 2, (canvas_height - frame.shape[0]) // 2, anchor=tk.NW, image=current_frame)
            
            if is_playing:
                root.after(10, update_video, canvas)
            
            update_time_label() 
        else:
            playing=False
            threading.Thread(target=play_pause).start()
            threading.Thread(target=boucle).start()
        
            


def update_time_label():
    global label_time, fps, total_frames, video
    current_frame_pos = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    current_seconds = current_frame_pos // fps
    current_time = "{:02d}:{:02d}:{:02d}".format(current_seconds // 3600, (current_seconds % 3600 // 60), current_seconds % 60)

    total_seconds = total_frames // fps
    total_time = "{:02d}:{:02d}:{:02d}".format(total_seconds // 3600, (total_seconds % 3600 // 60), total_seconds % 60)

    time_label_text = f"{current_time}/{total_time}"
    label_time.config(text=time_label_text)



def pause_video():
    global is_playing
    is_playing = False



def play_pause():
    global  playing, paused

    pygame.mixer.init()
    if not playing:
        pygame.mixer.music.load("temp/audio_temporaire.mp3")
        pygame.mixer.music.play()
        playing = True
    else:
        if not paused:
            pygame.mixer.music.pause()
            paused = True
        else:
            pygame.mixer.music.unpause()
            paused = False




def pause_media():
    global btn_play, btn_pause
    pause_video()
    play_pause()
    btn_play.configure(state="active")
    btn_pause.configure(state="disabled")



def play_media():
    global btn_play, btn_pause
    play_video()
    play_pause()
    btn_play.configure(state="disabled")
    btn_pause.configure(state="active")

# fin   sync video & audio *******************************************************







# volume
def set_volume(volume):
    volume_level = float(volume) / 100
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        volume.SetMasterVolume(volume_level, None)




# vider le dossier temp
def vider_dossier():
    dossier='./temp'
    # Vérifier si le dossier existe
    if os.path.exists(dossier):
        # Parcourir tous les fichiers et sous-dossiers dans le dossier
        for fichier in os.listdir(dossier):
            chemin_fichier = os.path.join(dossier, fichier)
            try:
                # Supprimer les fichiers
                if os.path.isfile(chemin_fichier) or os.path.islink(chemin_fichier):
                    os.unlink(chemin_fichier)
                # Supprimer les dossiers et leur contenu
                elif os.path.isdir(chemin_fichier):
                    shutil.rmtree(chemin_fichier)
            except Exception as e:
                print(f"Erreur en supprimant {chemin_fichier}: {e}")
    else:
        print(f"Le dossier {dossier} n'existe pas")




# action on_closing root
def on_closing():
    global cnt
    global cap

    if cap is not None:
        cap.release()

    if cnt>0:
        reset()
        vider_dossier()
    root.destroy()






#traitement audio 
def audio_pred():
    global model_audio,zone_text_audio

    result_string_audio, proba=aud.prediction_audio("temp/audio_temporaire.mp3",model_audio)
    zone_text_audio.configure(text=f"{result_string_audio} : {proba:.3f}")
    
def confirme():
    global chk_state_audio

    if chk_state_audio.get():
       audio_pred()
    else:
       return
#fin traitement









if __name__ == "__main__":
    # create_widgets()
    # create_widgets_result()
    # create_widgets_valider()
    show_home()
    root.protocol("WM_DELETE_WINDOW", on_closing)  
    root.mainloop()





