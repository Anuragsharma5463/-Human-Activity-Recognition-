# -Human-Activity-Recognition-

# Human Activity Recognition Streamlit App

This project is a Streamlit web app for human activity recognition. It can detect activities from a captured camera image or an uploaded image and show safety/posture alerts for selected activities.

## Features

- Browser camera input using Streamlit
- Image upload support
- TensorFlow/Keras `.h5` model loading
- Optional MediaPipe pose landmarks
- Activity confidence score
- Warning messages for risky activities such as fighting, texting, sleeping, sitting, and using laptop

## Project Structure

```text
human_activity/
+-- app.py
+-- requirements.txt
+-- README.md
+-- action_model.h5
+-- model/
    +-- action_model.h5
```

The app first looks for the model at `model/action_model.h5`. If it is not found, it tries `action_model.h5` in the project root.

## Run Locally

1. Install Python 3.10 or 3.11.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the Streamlit app:

```bash
streamlit run app.py
```

4. Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click **New app**.
4. Select your GitHub repository.
5. Set the main file path as:

```text
app.py
```

6. Click **Deploy**.

## Important Notes

- Keep `requirements.txt` in the root folder.
- Keep the trained model file in `model/action_model.h5`.
- Streamlit Cloud supports browser camera capture through `st.camera_input`, not the desktop OpenCV webcam window.
- If MediaPipe fails on the deployment server, the app will still run without pose landmarks.

## Activity Classes

The model supports these classes:

```text
calling, clapping, cycling, dancing, drinking, eating, fighting, hugging,
laughing, listening_to_music, running, sitting, sleeping, texting, using_laptop
```
