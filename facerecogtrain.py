# Fase 2 - Reconocimiento de caras
# Entrenamiento basado en imagenes capturadas
import cv2
import os
import numpy as np

dataPath = '/Users/patricio/git/computervision/data'
personas = [filename for filename in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, filename))] # captura lista de directorios dentro de al ruta dataPath
print('Rostros: ', personas)
# alternativa para listar directorios
''' for filename in os.listdir(dataPath):
    if os.path.isdir(os.path.join(dataPath,filename)):
        personas.append(filename) 
'''

labels = []
rostros = []
label = 0

for nameDir in personas:
    persona = dataPath + '/'+ nameDir
    # print('Imagenes')

    for fileName in os.listdir(persona):
        print('Rostros: ', nameDir + '/'+ fileName)
        labels.append(label)
        rostros.append(cv2.imread(persona + '/' +fileName,0)) 
        # 0 indica imagen en escala de grises

        image = cv2.imread(persona+'/'+fileName, 0)
        # test lectura de imagenes
        #cv2.imshow('image', image)
        #cv2.waitKey(10)

    label = label +1
#cv2.destroyAllWindows()

# Entrenamiento
print("Entrenamiento...")

# modelo Eigenface
face_recon = cv2.face.EigenFaceRecognizer_create()
face_recon.train(rostros, np.array(labels))
face_recon.write("data/modelEigenFace.xml")

# modelo Fisherface 
#face_recon = cv2.face.FisherFaceRecognizer_create()

print("Modelo creado.")
# testear imagenes
#print('numero de etiquetas: ', np.count_nonzero(np.array(labels)==0))
