# Here's how the application will work:

# The user selects a video file to upload.
# The application loads the video, extracts keypoints, and runs the TCN model for prediction.
# It displays the predicted class as the output.

# Run the application using in the terminal:
# python upload-predict.py

# 1. Saving the Model (Required);
# After training your model, use this code to save the model’s state dictionary, which contains all the parameters.
# # Save the trained model
# torch.save(model.state_dict(), "saved_model.pth")

# Import necessary libraries
import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
import numpy as np
import torch
import mediapipe as mp
import torch.nn as nn

class_labels = [
    "can_you_help_me", "doesnt_matter", "good_bey", "i_have_to_go",
    "i_think_you_are_wrong", "sorry_cant_stay", "sorry_for_being_late",
    "speak_slowly", "thanks_for_your_concern", "wish_you_luck_in_work",
    "wish_you_good_journey", "wish_you_good_vacation", "please_quickly",
    "explain_again", "repeat_again", "free_or_busy", "happy_to_know_you",
    "i_disagree", "i_agree", "i_would_like_to_meet_you", "any_service",
    "come_quickly", "happy_new_year", "how_can_i_call_you", "wait_please",
    "lets_go_swim", "whats_your_name", "what_about_going_for_a_walk",
    "unbelievable", "can_i_take_from_your_time"
]

# Dictionary to map English labels to Arabic meanings
class_labels_ar = {
    "can_you_help_me": "هل يمكنك مساعدتي",
    "doesnt_matter": "لا يهم",
    "good_bey": "مع السلامة",
    "i_have_to_go": "يجب أن أذهب",
    "i_think_you_are_wrong": "أعتقد أنك مخطئ",
    "sorry_cant_stay": "عذرًا، لا أستطيع البقاء",
    "sorry_for_being_late": "آسف على التأخير",
    "speak_slowly": "تحدث ببطء",
    "thanks_for_your_concern": "شكرًا على اهتمامك",
    "wish_you_luck_in_work": "أتمنى لك التوفيق في العمل",
    "wish_you_good_journey": "أتمنى لك رحلة سعيدة",
    "wish_you_good_vacation": "أتمنى لك إجازة سعيدة",
    "please_quickly": "من فضلك بسرعة",
    "explain_again": "اشرح مرة أخرى",
    "repeat_again": "أعد مرة أخرى",
    "free_or_busy": "هل أنت متفرغ أم مشغول",
    "happy_to_know_you": "سعيد بمعرفتك",
    "i_disagree": "لا أوافق",
    "i_agree": "أوافق",
    "i_would_like_to_meet_you": "أود مقابلتك",
    "any_service": "أي خدمة",
    "come_quickly": "تعال بسرعة",
    "happy_new_year": "سنة جديدة سعيدة",
    "how_can_i_call_you": "كيف يمكنني الاتصال بك",
    "wait_please": "انتظر من فضلك",
    "lets_go_swim": "لنذهب للسباحة",
    "whats_your_name": "ما اسمك",
    "what_about_going_for_a_walk": "ماذا عن الذهاب للمشي",
    "unbelievable": "غير معقول",
    "can_i_take_from_your_time": "هل يمكنني أن آخذ من وقتك"
}

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])


# Define the Temporal Convolutional Network (TCN) model
class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(TemporalConvNet, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.AdaptiveAvgPool1d(1)  # Global Average Pooling to reduce sequence dimension
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # TCN expects input of shape [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # Permute to match TCN input requirements
        x = self.tcn(x)
        x = x.squeeze(-1)  # Remove the last dimension after global pooling
        output = self.fc(x)
        return output


# Load the trained model
output_size = 30  # Number of classes (adjust to your project)
input_size = 258  # Number of features per frame
hidden_size = 512  # Hidden layer size
model = TemporalConvNet(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("./tcn_model.pth"))  # Load your trained model
model.eval()

import torch.nn.functional as F  # Import for softmax

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []

    # Use MediaPipe for keypoint extraction
    with mp.solutions.holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            _, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

    cap.release()
    sequence = torch.tensor([sequence], dtype=torch.float32)

    # Perform prediction
    with torch.no_grad():
        output = model(sequence)  # Raw model output (logits)
        probabilities = torch.softmax(output, dim=1)  # Convert logits to probabilities
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        probability = probabilities[0, predicted_class_idx].item()

    # Map the predicted class index to the corresponding class name in Arabic
    predicted_class_name = class_labels[predicted_class_idx]
    predicted_class_arabic = class_labels_ar.get(predicted_class_name, "Unknown")  # Default to "Unknown" if not found

    print(f"Predicted class: {predicted_class_arabic} with probability: {probability:.2f}")
    return predicted_class_arabic, probability


# Define the GUI
def open_file():
    file_path = filedialog.askopenfilename(title="Select a Video File", filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
    if file_path:
        predicted_class, probability = predict_video(file_path)
        result_label.config(text=f"Predicted Class: {predicted_class} \n with Probability: {probability:.2f}")

# Set up the Tkinter window
root = tk.Tk()
root.title("Sign Language Prediction")
root.geometry("400x200")

# Add widgets
upload_button = Button(root, text="Upload Video", command=open_file)
upload_button.pack(pady=20)

result_label = Label(root, text="Predicted Class: ")
result_label.pack(pady=10)

# Run the application
root.mainloop()