import face_recognition as fr
from PIL import Image, ImageDraw, ImageFont
import numpy as np


#banco de imagens
imagem001 = fr.load_image_file("pessoas-conhecidas/alan3.png")
imagem001_encoding = fr.face_encodings(imagem001)[0]

# Load a second sample picture and learn how to recognize it.
imagem002 = fr.load_image_file('pessoas-conhecidas/wanessa.jpg')
imagem002_encoding = fr.face_encodings(imagem002)[0]

imagem003 = fr.load_image_file('pessoas-conhecidas/Ruan.jpeg')
imagem003_encoding = fr.face_encodings(imagem003)

imagem004 = fr.load_image_file('pessoas-conhecidas/WhatsApp Image 2022-01-22 at 16.18.37.jpeg')
imagem004_encoding = fr.face_encodings(imagem004)

imagem005 = fr.load_image_file('pessoas-conhecidas/WhatsApp Image 2022-01-22 at 15.46.32.jpeg')
imagem005_encoding = fr.face_encodings(imagem005)

# bloco de imagens previamente armazenadas
# cada imagem adicionada precisa ser vinculada a versão encoding e separado por virgula
database_rostos = [
    imagem001_encoding,
    imagem002_encoding,
    imagem003_encoding,
    imagem004_encoding,
    imagem005_encoding,

# bloco de nomes previamente armazenados
# cada nome adicionado precisa ser como string e separado por virgula
database_nomes = [
    "Alan",
    "Wanessa",
    "Ruan",
    "Michel",
    "a garota do grupo"
]

# carregamento de uma imagem para identificar os rostos
loaded_imagem = fr.load_image_file("pessoas-conhecidas/alan1wanessa1.jpg")

# análise da imagem para identificação dos rostos presentes
faces = fr.face_locations(loaded_imagem)
faces_encodings = fr.face_encodings(loaded_imagem, faces)

#usa a biblioteca pillow para gerar um recorte do rosto e testar com alguma do banco de dados
imagem_final = Image.fromarray(loaded_imagem)
render = ImageDraw.Draw(imagem_final)

# Loop para encontrar rostos desconhecidos
# (face_encoding e top, right, bottom, left estão definidas no API, logo, devem ter obrigatoriamente estes nomes para serem utilizadas)
for (top, right, bottom, left), face_encoding in zip(faces, faces_encodings):
    # fr.compare_faces vai... comparar os rostos
    teste_positivo = fr.compare_faces(database_rostos, face_encoding)
    #não identificou alguém? indigente
    #tem um else aqui
    nome = "Indigente"
    # o loop a cima vai procurar um nome associado a database_rostos, se não encontrar, vai usar o nome 'indigente'
    # sim, o 'else' das funções for.. else pode ser omitido neste tipo de estrutura, já subentende

    # esse aqui é o bloco do algoritmo que encaixa os nomes
    # face distances verifica na verdade a verossimilhança entre as imagens carregada no face_encoding e no banco de imagens
    # usa a numpy.argmin para verificar a distância entre os pontos de rosto, quanto mais próximo do encontrado no registro, melhor
    # quando o teste confere positivo, é um match e ele puxa do banco de nomes, o respectivo nome
    face_distances = fr.face_distance(database_rostos, face_encoding)
    matched = np.argmin(face_distances)
    if teste_positivo[matched]:
        nome = database_nomes[matched]

    # aqui é criado um retangulo da imagem (render), segue as proporções do vetor gerado pelo face_encoding (R4)
    render.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0)) #Cor verde
    # cria uma legenda na imagem
    text_width, text_height = render.textsize(nome)
    render.rectangle(((left, bottom - text_height), (right, bottom)), fill=(0, 255, 0), outline=(0, 255, 0))
    render.text((left + 20, bottom - text_height), nome, fill=(0, 0, 0, 0))

    # DAR UM JEITO DE COLOCAR FORMATAÇÃO DE STRING

# uma vez gerado o quadro de legenda, as informações são apagadas
del render
# Mostra na tela a imagem final
imagem_final.show()
