import cv2

frontal_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
profile_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_profileface.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'  
)

for cascade, name in [(frontal_face_cascade, 'Frontal Face'),
                      (profile_face_cascade, 'Profile Face'),
                      (eye_cascade, 'Eye')]:
    if cascade.empty():
        print(f"Erro ao carregar {name}!")
        exit()

cap = cv2.VideoCapture("People Walking Free Stock Footage, Royalty-Free No Copyright Content.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frontal_faces = frontal_face_cascade.detectMultiScale(gray, 1.1, 5)
    profile_faces = profile_face_cascade.detectMultiScale(gray, 1.1, 5)
    all_faces = list(frontal_faces) + list(profile_faces)

    for (x, y, w, h) in all_faces:
        face_roi = frame[y:y+h, x:x+w]
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(face_roi, (99,99), 30)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=3,  
            minSize=(20, 20)  
        )
        
        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
            roi_color[ey:ey+eh, ex:ex+ew] = cv2.GaussianBlur(eye_roi, (49,49), 15)

    cv2.imshow('Face & Eye Blur', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()