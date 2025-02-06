# DESAFIO DIO - Criando um Sistema de Reconhecimento Facial do Zero #
Este projeto implementa um sistema de reconhecimento facial em tempo real usando Python. Ele captura vídeo da webcam, detecta rostos nos frames e identifica pessoas conhecidas comparando características faciais únicas (embeddings) com imagens de referência previamente cadastradas. O sistema utiliza bibliotecas como OpenCV , DeepFace e MTCNN para detecção facial, extração de embeddings e cálculo de similaridade.

## Principais Funcionalidades ##
Detecção de Rostos : Detecta rostos em tempo real usando algoritmos avançados.
Identificação de Pessoas : Compara embeddings faciais com imagens de referência para identificar pessoas conhecidas.
Visualização : Exibe caixas delimitadoras e rótulos com os nomes das pessoas identificadas ou "Unknown" para rostos desconhecidos.
Fácil Configuração : Basta adicionar imagens de referência e executar o script para iniciar o reconhecimento.
Como Funciona
O sistema extrai embeddings faciais usando o modelo Facenet (ou outros modelos suportados pelo DeepFace).
Calcula a similaridade entre os embeddings dos rostos detectados e os embeddings das imagens de referência.
Define um limiar de similaridade para decidir se um rosto corresponde a uma pessoa conhecida ou é desconhecido.
## Requisitos ##
Bibliotecas: OpenCV, DeepFace, MTCNN, Scikit-Learn.
Imagens de referência: Fotos claras e bem posicionadas de pessoas conhecidas.
Uso
Execute o script Python.
A câmera será ativada, e o sistema começará a detectar e identificar rostos.
Para encerrar, pressione a tecla q.
## Aplicações Práticas ##
Controle de acesso baseado em reconhecimento facial.
Identificação de indivíduos em ambientes controlados.
Protótipos de sistemas de segurança ou automação.
