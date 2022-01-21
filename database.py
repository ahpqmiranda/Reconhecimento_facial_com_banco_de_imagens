import face_recognition
from PIL import Image, ImageDraw
import numpy as np

# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("pessoas-conhecidas/alan3.png")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file('pessoas-conhecidas/alan3wanessa3.jpg')
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "Alan",
    "Wanessa"
]

# Load an image with an unknown face
unknown_image = face_recognition.load_image_file("pessoas-conhecidas/alan1wanessa1.jpg")

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(unknown_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Desconhecido"

    # If a match was found in known_face_encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 20, bottom - text_height), name, fill=(255, 255, 255, 255))


# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
pil_image.show()

# You can also save a copy of the new image to disk if you want by uncommenting this line
# pil_image.save("image_with_boxes.jpg")


'''

import face_recognition as fr
from PIL import Image, ImageDraw
import numpy as np

# armazenando os rostos conhecidos
image001 = fr.load_image_file("pessoas-conhecidas/alan3.png")
image001_encoding = fr.face_encodings(image001)[0]

image002 = fr.load_image_file("pessoas-conhecidas/alan3wanessa3.jpg")
image002_encoding = fr.face_encodings(image002)[0]

# Create arrays of known face encodings and their names
rostos_conhecidos = [
    image001_encoding,
    image002_encoding
]
nome_dos_rostos_conhecidos = [
    "Alan",
    "Wanessa"
]
# ---------------------------------------------------------------------------------------------
# ---------------------------- fim do bloco banco de dados ------------------------------------
# ---------------------------------------------------------------------------------------------

# Load an image with an unknown face
foto_teste = fr.load_image_file("two_people.jpg")

# procura pelas imagens dos rostos das pessoas
faces = fr.face_locations(foto_teste)
faces2 = fr.face_encodings(foto_teste, faces)

# vai converter o vetor de "faces" em imagem já com o rosto recortado (biblioteca pillow)
recorte = Image.fromarray(faces2)
# combina as imagens com o ImageDraw
desenho: ImageDraw = ImageDraw.Draw(recorte)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(faces, faces2):
    # See if the face is a match for the known face(s)
    identificado = fr.compare_faces(nome_dos_rostos_conhecidos, faces2)

    nome = "Não Conhecido"

    # If a match was found in known_face_encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # Or instead, use the known face with the smallest distance to the new face
    distancia_rosto = fr.face_distance(rostos_conhecidos, face_encoding)
    melhor_identificado = np.argmin(distancia_rosto)
    if identificado[melhor_identificado]:
        nome = rostos_conhecidos[melhor_identificado]

    # Draw a box around the face using the Pillow module
    desenho.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    text_width, text_height = desenho.textsize(nome)
    desenho.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    desenho.text((left + 6, bottom - text_height - 5), nome, fill=(255, 255, 255, 255))


# Remove the drawing library from memory as per the Pillow docs
del desenho

# Display the resulting image
recorte.show()

# You can also save a copy of the new image to disk if you want by uncommenting this line
# pil_image.save("image_with_boxes.jpg")

'''