import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Liste des classes
classes =  ["sad" ,"surprise" ,"angry" ,"fear" ,"happy","neutral"]
# ['happy', 'sad', 'angry', 'fear']
# Initialisation du label encoder
le = LabelEncoder()
le.fit(classes)

# Fonction pour générer un spectrogramme de taille fixe
def generate_spectrogram(file_path, max_pad_len=128):
    audio, sample_rate = librosa.load(file_path, sr=None)
    # Appliquer la fonction envelope pour filtrer l'audio
    mask = envelope(audio, sample_rate, threshold=0.0005)
    audio = audio[mask]
    
    # Générer le spectrogramme
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Adapter le spectrogramme à la taille fixe
    if log_spectrogram.shape[1] > max_pad_len:
        log_spectrogram = log_spectrogram[:, :max_pad_len]
    else:
        pad_width = max_pad_len - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
    
    return log_spectrogram

# Fonction pour appliquer un filtre sur l'audio
def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

# Fonction de prédiction audio
def prediction_audio(path, model):
    if model is None:
        raise ValueError("Le modèle est None. Assurez-vous que le modèle est chargé correctement.")
    
    # Gérer les étiquettes inconnues
    unknown_label = "unknown"
    
    T1 = generate_spectrogram(path)
    spectrogram = T1.reshape(1, 128, 128, 1)
    
    # Faire la prédiction
    prediction = model.predict(spectrogram)
    
    # Obtenir l'indice de la classe prédite avec la probabilité maximale
    predicted_index = np.argmax(prediction)

    # Vérifier si l'indice est valide
    if predicted_index < len(le.classes_):
        # Inverser la transformation et obtenir l'étiquette prédite
        predicted_label = le.inverse_transform([predicted_index])[0]
    else:
        # Si l'indice est invalide, attribuer une étiquette par défaut
        predicted_label = unknown_label
    
    return predicted_label, prediction[0][predicted_index]

