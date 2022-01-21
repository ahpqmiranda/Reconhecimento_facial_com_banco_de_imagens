from PIL import Image
import face_recognition as fr


imagem = fr.load_image_file("pessoas-conhecidas/alan2wanessa2.jpg")
faces = fr.face_locations(imagem)
print('Foram encontradas {} pessoas'.format(len(faces)))
for coordenadas in faces:
    x, y, w, z = coordenadas
    rostoEncontrado = imagem[x:w, y:z]
    recorte_da_foto_original = Image.fromarray(rostoEncontrado)
    recorte_da_foto_original.show()
