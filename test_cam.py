import cv2

#url = "rtsp://192.168.1.39:8554/live"
url = "rtsp://admin:admin@192.168.1.39:8554/live"

cap = cv2.VideoCapture(url)

ret, frame = cap.read()
if ret:
    print("✓ Conexion exitosa! Frame recibido.")
    cv2.imwrite("test_frame.jpg", frame)
    print("  Guardado como test_frame.jpg")
else:
    print("✗ Sin frame. No se pudo conectar.")

cap.release()