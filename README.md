# Hand And Body Pose Detection – Real-time Posture Correction

This system uses MediaPipe and OpenCV to detect your hand and body pose in time.

It checks the angle of your elbow. Makes sure you are standing or sitting correctly.

All the information it collects is saved to a file called hand_posture.csv.

## What This System Does

- It looks at your body pose, including your shoulders, elbows, wrists and hand landmarks.

- It calculates your elbow angle as you move your hand.

- It checks if your posture is good or not.

Your elbow angle should be than 160 degrees.

Your wrist should be above your shoulder.

- It saves all the data from each frame like the time, angles and how sure it is, to the hand_posture.csv file.

- You get to see how you are doing on the screen with messages and scores.

## What We Used To Build This

- We used Python 3.x to make this system work.

- OpenCV helps us take videos and show them on the screen.

- MediaPipe is good, at finding the pose of your body and hands.

- NumPy is a library that helps us calculate the angles.

- We use CSV files to save all the data we collect.


pip install -r requirements.txt

```
