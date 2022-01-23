import cv2

# biblioteca de classificação da opencv, pego no github do projeto
xml_haar = 'haarcascade_frontalface_alt2.xml'
# carregando a base de dados do classificador
faceclass = cv2.CascadeClassifier(xml_haar)
# captura do vídeo
capture = cv2.VideoCapture(1)
# definições de resolução de vídeo
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

# “loop” de operação: o programa é encerrado com a tecla "q" pressionada
while not cv2.waitKey(20) & 0xFF == ord("q"):

    # a captura de video e direcionada para a variável frame_color, a deteção das faces é feita na variável gray
    # o algoritmo responde melhor na leitura de faces numa imagem em tons de cinza
    ret, frame_color = capture.read()
    gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
    faces = faceclass.detectMultiScale(gray)

    # O ‘loop’ for cria o retângulo em torno do rosto da pessoa em cima da variável frame_color.
    # Isto, a partir dos dados obtidos na variável gray sobre o rosto

    for x, y, w, h in faces:
        cv2.rectangle(frame_color, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('color', frame_color)
