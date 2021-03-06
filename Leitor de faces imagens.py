import face_recognition as fr
from PIL import Image, ImageDraw
import numpy as np


#banco de imagens
imagem001 = fr.load_image_file("pessoas-conhecidas/alan1.jpg")
imagem001_encoding = fr.face_encodings(imagem001)

imagem002 = fr.load_image_file('pessoas-conhecidas/Ronaldo que não é o Cristiano.jpeg')
imagem002_encoding = fr.face_encodings(imagem002)

imagem003 = fr.load_image_file('pessoas-conhecidas/Ruan.jpeg')
imagem003_encoding = fr.face_encodings(imagem003)

imagem004 = fr.load_image_file('pessoas-conhecidas/Michel.jpeg')
imagem004_encoding = fr.face_encodings(imagem004)

imagem005 = fr.load_image_file('pessoas-conhecidas/victória.jpeg')
imagem005_encoding = fr.face_encodings(imagem005)

imagem006 = fr.load_image_file('pessoas-conhecidas/Giordano.jpeg')
imagem006_encoding = fr.face_encodings(imagem006)

# bloco de imagens previamente armazenadas
# cada imagem adicionada precisa ser vinculada a versão encoding e separado por virgula
database_rostos = [
    imagem001_encoding,
    imagem002_encoding,
    imagem003_encoding,
    imagem004_encoding,
    imagem005_encoding,
    imagem006_encoding
    ]

# bloco de nomes previamente armazenados
# cada nome adicionado precisa ser como string e separado por virgula
database_nomes = [
    "Alan",
    "Ronaldo",
    "Ruan",
    "Michel",
    "Victória",
    "Giordano"
    ]

# carregamento de uma imagem para identificar os rostos
loaded_imagem = fr.load_image_file("pessoas-conhecidas/Ronaldo que não é o Cristiano.jpeg")

# análise da imagem para identificação dos rostos presentes
faces = fr.face_locations(loaded_imagem)
faces_encodings = fr.face_encodings(loaded_imagem, faces)

#usa a biblioteca pillow para gerar um recorte do rosto e testar com alguma do banco de dados
imagem_final = Image.fromarray(loaded_imagem)
render = ImageDraw.Draw(imagem_final)

# Loop para encontrar rostos desconhecidos
# (face_encoding e top, right, bottom, left estão definidas no API, logo, devem ter obrigatoriamente estes nomes para serem utilizadas)
# o loop a cima vai procurar um nome associado a database_rostos, se não encontrar, vai usar o nome 'indigente'
# sim, o 'else' das funções for.. else pode ser omitido neste tipo de estrutura, já subentende

for (top, right, bottom, left), face_encoding in zip(faces, faces_encodings):

    teste_positivo = fr.compare_faces(database_rostos, face_encoding)
    nome = "Desconhecido"


    # esse aqui é o bloco do algoritmo que encaixa os nomes
    # face distances verifica na verdade a verossimilhança entre as imagens carregada no face_encoding e no banco de imagens
    # usa a numpy.argmin para verificar a distância entre os pontos de rosto, quanto mais próximo do encontrado no registro, melhor
    # quando o teste confere positivo, é um match e ele puxa do banco de nomes, o respectivo nome
    face_distances = fr.face_distance(database_rostos, face_encoding)
    matched = np.argmin(face_distances)
    if teste_positivo[matched]:
        nome = database_nomes[matched]

    # aqui é criado um retangulo da imagem (render), segue as proporções do vetor gerado pelo face_encoding (R4)
    render.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
    # cria uma legenda na imagem
    text_width, text_height = render.textsize(nome)
    render.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    render.text((left + 6, bottom - text_height - 5), nome, fill=(255, 255, 255, 255))

# uma vez gerado o quadro de legenda, as informações são apagadas
del render
# Mostra na tela a imagem final
imagem_final.show()

