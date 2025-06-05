import cv2
cap = cv2.VideoCapture("input_video.mp4")
ret, frame = cap.read()
print("Video loaded successfully:", ret)
cap.release()