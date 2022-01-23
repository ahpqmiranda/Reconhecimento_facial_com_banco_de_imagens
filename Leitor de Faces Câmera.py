# PIL, biblioteca de manipulação de imagens
from PIL import Image
# face-recogniti biblioteca p/ reconhecimento facial (IA)
import face_recognition as fr  # abreviei o nome da biblioteca para fr

# carregando um arquivo de imagem
imagem = fr.load_image_file("pessoas-conhecidas/alan3wanessa3.jpg")
# converte a imagem em vetor (R4)
faces = fr.face_locations(imagem)
# cada vetor é uma pessoa encontrada, uso a função len para contar quantos vetores tem
print('Foram encontradas {} pessoas'.format(len(faces)))

# esse loop vai realizar o processo de 0 a n vetores encontrados em "faces"
for coordenadas in faces:
    top, right, bottom, left = coordenadas  # separo as coordenadas do vetor em variáveis (as variáveis precisam ser chamadas de top, right, bottom, left ou não funciona, ele não chama a biblioteca api)
    rosto = imagem[top:bottom, left:right]  # monto uma box com dimensões equivalentes aos rostos encontrados
    img_final = Image.fromarray(rosto)  # recorta a imagem original com a box na região do rosto da pessoa
    img_final.show()  # mostra a imagem recortada do rosto
