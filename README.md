# Reconhecimento facial com banco de imagens
Código para identificar pessoas através de um banco de imagens prévio.

Foi utilizado o Anaconda3 integrado com Pycharm no desenvolvimento.

| bibliotecas utilizadas: | Função                 |
|-------------------------|------------------------|
| numpy                   | Manipulação de vetores |
| Pillow                  | Manipulação de imagens |
| Face_Recognition        | Reconhecimento Facial  |
| opencv                  | Recursos de vídeo      |

# Leitor de biometria facial por câmera

Este código utiliza somente a bilioteca opencv.

Fluxograma:

![Esquemático do código](fluxogramas/fluxograma%20leitor%20por%20vídeo.jpeg)

Algumas notas:
1. A biblioteca opencv precisa do arquivo xml disponível no github dos desenvolvedores, sendo que a haarcascade_frontalface_alt2 não é a única.
2. O comando `capture.set(cv2.CAP_PROP_FRAME_(WIDTH e HEIGHT)` dependem de renderização de vídeo, então, evitar colocar resoluções altas.
3. O comando ``faceclass.detect(MultiScale)()`` funciona melhor em imagens não coloridas.


