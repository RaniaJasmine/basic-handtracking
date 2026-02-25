import cv2 #handles camera and image display
import mediapipe as mp #google's ml library for hand trackin

print("OpenCV version:", cv2.__version__)
print("MediaPipe version:", mp.__version__)

# Initialize MediaPipe
mp_hands = mp.solutions.hands #Gets the hand detection model from MediaPipe This is the actual AI model that detects hands
mp_drawing = mp.solutions.drawing_utils#Gets the drawing utilities,Contains functions to draw landmarks on the image
mp_drawing_styles = mp.solutions.drawing_styles# Gets predefined drawing styles. Provides nice colors and styles for the landmarks

print("\nStarting hand tracking... Press 'q' to quit")

# Open webcam
cap = cv2.VideoCapture(0) #Cap stands for video capture.0 is ur default webcam.this line connects code to hardware

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()#it stops,if cam opening was unsuccessful
#if camera opens.
print("Camera opened successfully!")

# Initialize hand detection
with mp_hands.Hands(
    static_image_mode=False,#static image =true means picture,else video
    max_num_hands=2,#only 2 hands max
    min_detection_confidence=0.5,#min confidence can be from 0-1
    min_tracking_confidence=0.5) as hands:#Minimum confidence to keep tracking a hand
    
    while cap.isOpened():#so cam doesnt close
        success, frame = cap.read()#to read frame.success is boolean (True if frame read OK), frame contains the image
        if not success:
            print("Failed to grab frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)#mirror effect
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame,THIS IS THE CORE! Feeds the image to MediaPipe's AI
        results = hands.process(frame_rgb)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:#checks if hands are detected,so it draws on said detected hands
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        # Show frame
        cv2.imshow('Hand Tracking', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Camera released. Goodbye!")





##THIS IS FOR COUNTING FINGERS
# Add this inside the if results.multi_hand_landmarks: block
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(...)  # your existing code
        
        # Count fingers (simple version)
        fingers = []
        # Thumb
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers
        for tip in [8, 12, 16, 20]:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        total_fingers = sum(fingers)
        cv2.putText(frame, f'Fingers: {total_fingers}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)