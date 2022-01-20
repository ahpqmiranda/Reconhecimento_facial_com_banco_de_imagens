import cv2
from PIL import Image
import face_recognition as fr
import dlib

print(dlib.cuda.get_num_devices())

def main(void):
    #aquisição de imagem com a biblioteca face_recognition
    #carrega a imagem na variável foto
    foto = fr.load_image_file('pessoas-conhecidas/alan3.png')

    #encontra faces usando o modelo HOG
    face_locations = fr.face_locations(foto)
    face_render = fr.face_encodings(foto)

    #número de rostos encontrados
    print('Existem {} pessoas nesta imagem'.format(len(face_locations)))

    present_original_image = Image.fromarray(foto)
    present_original_image.show(void, 'pessoa encontrada')


    while not cv2.waitKey(50) & 0xFF == ord('q'):
        #manipulação de imagem com a biblioteca pillow
