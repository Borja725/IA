import socket

mi_socket = socket.socket()
mi_socket.bind(('192.168.1.42', 8000))
mi_socket.listen(5)

while True:
    conexion, addr = mi_socket.accept()
    print("Nueva conexión establecida")
    print(addr)
    mensaje = "Hola desde el servidor"
    conexion.send(mensaje.encode('utf-8'))  # Codificar la cadena en bytes antes de enviarla
    conexion.close()