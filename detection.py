import cv2
import streamlit as st

# Charger le classificateur Haar pour la détection des visages
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(scale_factor, min_neighbors, rectangle_color, save_images):
    """Fonction principale pour détecter les visages."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Erreur : Impossible d'accéder à la webcam.")
        return

    stframe = st.empty()  # Placeholder Streamlit pour afficher les images

    while not st.session_state.stop_detection:
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur : Impossible de capturer une image depuis la webcam.")
            break

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détecter les visages
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        # Dessiner des rectangles autour des visages détectés
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)

        # Convertir l'image BGR au format RGB pour Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", caption="Face Detection")

        # Enregistrer l'image si demandé
        if save_images and len(faces) > 0:
            cv2.imwrite("detected_faces.jpg", frame)
            st.success("Image enregistrée avec succès !")

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()

def app():
    st.title("Détection de visages avec OpenCV")
    st.write("Bienvenue dans l'application de détection de visages ! Suivez les étapes ci-dessous :")
    st.markdown("""
        1. Appuyez sur **Start Detection** pour lancer la détection de visages.
        2. Utilisez les curseurs pour ajuster les paramètres de détection : 
           - **scaleFactor** : Facteur d'échelle pour redimensionner l'image. 
           - **minNeighbors** : Nombre minimum de voisins pour valider un visage détecté.
        3. Choisissez la couleur des rectangles via le sélecteur de couleur.
        4. Si activé, les images avec visages détectés seront sauvegardées.
        5. Appuyez sur **Stop Detection** pour arrêter.
    """)

    # Contrôle des états
    if "stop_detection" not in st.session_state:
        st.session_state.stop_detection = False

    # Paramètres interactifs
    scale_factor = st.slider("Facteur d'échelle (scaleFactor)", 1.1, 2.0, 1.3, 0.1)
    min_neighbors = st.slider("Nombre minimum de voisins (minNeighbors)", 1, 10, 5, 1)
    rectangle_color = st.color_picker("Choisissez la couleur des rectangles", "#00FF00")
    save_images = st.checkbox("Enregistrer les images avec visages détectés", value=False)

    # Boutons pour contrôler la détection
    if st.button("Start Detection"):
        st.session_state.stop_detection = False
        # Convertir la couleur hexadécimale en BGR pour OpenCV
        color_bgr = tuple(int(rectangle_color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
        detect_faces(scale_factor, min_neighbors, color_bgr, save_images)

    if st.button("Stop Detection"):
        st.session_state.stop_detection = True

if __name__ == "__main__":
    app()

