import cv2
import tensorflow as tf

# Carregar o modelo treinado
model = tf.keras.models.load_model('meu_modelo.h5')

# Inicializar a câmera
cap = cv2.VideoCapture(0)

while True:
    # Capturar um frame
    ret, frame = cap.read()

    # Pré-processar a imagem (segmentação, normalização, etc.)
    # ...

    # Fazer a predição
    previsao = model.predict(frame)

    # Mostrar a imagem e a legenda
    cv2.imshow('Video', frame)
    cv2.putText(frame, 'Gestos: ' + str(previsao), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Sair ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a câmera e fechar as janelas
cap.release()
cv2.destroyAllWindows()

