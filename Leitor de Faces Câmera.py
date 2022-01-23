import cv2
#biblioteca de classificação da opencv, pego no github do projeto
xml_haar = 'haarcascade_frontalface_alt2.xml'
#carregando a base de dados do classificador
faceClassification = cv2.CascadeClassifier(xml_haar)

capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

while not cv2.waitKey(20) & 0xFF == ord("q"):

    ret, frame_color = capture.read()
    gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)

    faces = faceClassification.detectMultiScale(gray)

    for x, y, w, h in faces:
        cv2.rectangle(frame_color, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('color', frame_color)
    cv2.imshow('gray', gray)
