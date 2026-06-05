import cv2
import mediapipe as mp
import numpy as np
import csv
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

# Mediapipe Hands و Pose
hand_detector = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose_detector = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def angle_between(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

data_list = []

while True:
    ret, image = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    pose_results = pose_detector.process(image_rgb)
    hand_results = hand_detector.process(image_rgb)

    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    posture_correct = 0
    elbow_angle = 0
    wrist_x = wrist_y = elbow_x = elbow_y = shoulder_x = shoulder_y = finger_tip_y = 0
    shoulder_conf = elbow_conf = wrist_conf = 0

    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        shoulder_x, shoulder_y, shoulder_conf = shoulder.x, shoulder.y, shoulder.visibility
        elbow_x, elbow_y, elbow_conf = elbow.x, elbow.y, elbow.visibility
        wrist_x, wrist_y, wrist_conf = wrist.x, wrist.y, wrist.visibility

        # زاوية الكوع
        elbow_angle = angle_between(
            (shoulder_x, shoulder_y),
            (elbow_x, elbow_y),
            (wrist_x, wrist_y)
        )

        if elbow_angle > 160 and wrist_y < shoulder_y:
            posture_correct = 1

        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # إذا تم الكشف عن اليد
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_tip_y = min([hand_landmarks.landmark[i].y for i in [8, 12, 16, 20]])

    data_list.append([
        time.strftime('%Y-%m-%d %H:%M:%S'),
        elbow_angle, wrist_x, wrist_y, wrist_conf,
        elbow_x, elbow_y, elbow_conf, shoulder_x, shoulder_y, shoulder_conf,
        finger_tip_y, posture_correct
    ])

    status_text = "Raise your Hands" if posture_correct else "Adjust your elbow and your palm up"
    cv2.putText(image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if posture_correct else (0, 0, 255), 2)

    conf_text = f"Shoulder:{shoulder_conf:.2f} Elbow:{elbow_conf:.2f} Wrist:{wrist_conf:.2f}"
    cv2.putText(image, conf_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Hand Pose Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

with open('hand_posture.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'timestamp', 'elbow_angle', 'wrist_x', 'wrist_y', 'wrist_conf',
        'elbow_x', 'elbow_y', 'elbow_conf', 'shoulder_x', 'shoulder_y', 'shoulder_conf',
        'finger_tip_y', 'posture_correct'
    ])
    writer.writerows(data_list)

print("Data saved to hand_posture.csv ✅")