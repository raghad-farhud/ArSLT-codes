#!/usr/bin/env python
# coding: utf-8

# 1. Extract Keypoints from Videos: Iterate through your video dataset, extract keypoints using MediaPipe, and save them to a folder.
# 2. Train the Model: Use the extracted keypoints to train your deep learning model.
# 
# ## Step 1: Extract Keypoints from Videos

# In[1]:


# pip install tensorflow==2.16.1 opencv-python mediapipe scikit-learn matplotlib


# In[2]:


import cv2
import numpy as np
import os
import mediapipe as mp


# In[3]:


# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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


def draw_landmarks(image, results):
    # Draw pose, left hand, and right hand landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


# In[7]:


# Path to video dataset
VIDEO_PATH = '../video-dataset'
DATA_PATH = 'Data_MP'
# Actions to detect
actions = np.array([
 "can_you_help_me",
 "doesnt_matter",
 "good_bey",
 "i_have_to_go",
 "i_think_you_are_wrong",
 "sorry_cant_stay",
 "sorry_for_being_late",
 "speak_slowly",
 "thanks_for_your_concern",
    "wish_you_luck_in_work"
    "wish_you_good_journey",
    "wish_you_good_vacation",
    "please_quickly",
    "explain_again",
    "repeat_again",
    "free_or_busy",
    "happy_to_know_you",
    "i_disagree",
    "i_agree",
    "i_would_like_to_meet_you",
    "any_service",
    "come_quickly",
    "happy_new_year",
    "how_can_i_call_you",
    "wait_please",
    "lets_go_swim",
    "whats_your_name",
    "what_about_going_for_a_walk",
    "unbelievable",
    "can_i_take_from_your_time"
])


# In[8]:


# Create folders for each action
for action in actions:
    action_dir = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_dir):
        os.makedirs(action_dir)


# In[9]:


# Process each video
for action in actions:
    video_files = os.listdir(os.path.join(VIDEO_PATH, action))
    print(f"Processing action: {action}, Number of videos: {len(video_files)}")  # Debug print
    for idx, video_file in enumerate(video_files):
        cap = cv2.VideoCapture(os.path.join(VIDEO_PATH, action, video_file))
        frame_num = 0
        video_folder = os.path.join(DATA_PATH, action, str(idx))
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        print(f"Processing video: {video_file}, Saving to folder: {video_folder}")  # Debug print
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            video_keypoints = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process the frame
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                
                # Only save non-zero keypoints (frames where detection happened)
                if np.count_nonzero(keypoints) != 0:
                    npy_path = os.path.join(video_folder, f"{frame_num}.npy")
                    # print(f"Saving keypoints to: {npy_path}")  # Debug print
                    np.save(npy_path, keypoints)
                    video_keypoints.append(keypoints)

                # Real-time keypoint visualization
                # draw_landmarks(image, results)
                # cv2.imshow('Keypoint Extraction', image)
                # if cv2.waitKey(10) & 0xFF == ord('q'):
                #     break

                frame_num += 1

        # Save all keypoints for the video as a sequence
        sequence_npy_path = os.path.join(DATA_PATH, action, f"video_{idx}_sequence.npy")
        np.save(sequence_npy_path, video_keypoints)
        cap.release()
        print(f"Finished processing video: {video_file}, Total frames: {frame_num}")  # Debug print


# In[19]:


cv2.destroyAllWindows()

