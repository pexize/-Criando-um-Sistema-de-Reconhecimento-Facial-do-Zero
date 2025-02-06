import cv2
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

# Lista de embeddings conhecidos e nomes correspondentes
known_embeddings = []
known_names = []

# Função para extrair embeddings faciais usando DeepFace
def extract_face_embedding(image_path):
    try:
        # Extrair embedding usando Facenet
        result = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
        
        # Extrair o embedding do dicionário retornado
        if isinstance(result, list):  # DeepFace pode retornar uma lista de resultados
            embedding = np.array(result[0]['embedding'])
        else:  # Ou um único dicionário
            embedding = np.array(result['embedding'])
        
        return embedding
    except Exception as e:
        print(f"Erro ao extrair embedding: {e}")
        return None

# Adicionar embeddings de pessoas conhecidas
def add_known_person(image_path, name):
    embedding = extract_face_embedding(image_path)
    if embedding is None:
        raise ValueError(f"Não foi possível extrair o embedding da imagem '{image_path}'.")
    
    known_embeddings.append(embedding)
    known_names.append(name)
    print(f"Embedding adicionado para {name}: {embedding[:5]}...")  # Imprime os primeiros valores do embedding

# Adicionar pessoas conhecidas
add_known_person('person1.jpg', 'DiCaprio')
add_known_person('person2.jpg', 'Cameron')
add_known_person('person3.jpg', 'Morgan')
add_known_person('person4.jpg', 'Zooey')


# Capturar vídeo da webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível abrir a webcam.")
else:
    print("Webcam aberta com sucesso!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame da webcam.")
        break
    
    # Salvar o frame temporariamente para processamento pelo DeepFace
    temp_image_path = "temp_frame.jpg"
    cv2.imwrite(temp_image_path, frame)
    
    # Detectar rostos no frame usando OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Recortar o rosto detectado
        face = frame[y:y+h, x:x+w]
        
        # Salvar o rosto temporariamente para processamento pelo DeepFace
        face_temp_path = "temp_face.jpg"
        cv2.imwrite(face_temp_path, face)
        
        # Extrair embedding do rosto
        embedding = extract_face_embedding(face_temp_path)
        
        if embedding is not None:
            # Comparar com todos os embeddings conhecidos
            best_match_index = -1
            best_similarity = -1
            for i, known_embedding in enumerate(known_embeddings):
                similarity = cosine_similarity([embedding], [known_embedding])
                similarity = similarity[0][0]
                print(f"Similaridade com {known_names[i]}: {similarity:.4f}")
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_index = i
            
            # Definir rótulo com base na melhor correspondência
            if best_similarity > 0.5:  # Limiar ajustado para Facenet
                label = known_names[best_match_index]
                color = (0, 255, 0)  # Verde para correspondência
            else:
                label = "Unknown"
                color = (0, 0, 255)  # Vermelho para desconhecido
            
            # Desenhar a caixa delimitadora e o rótulo
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Mostrar o frame
    cv2.imshow('Face Recognition', frame)
    
    # Sair ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar todas as janelas
cap.release()
cv2.destroyAllWindows()