import serial
import time

def face_cordinates(ser):
    ser = serial.Serial('COM4', 9600, timeout=1)
    time.sleep(2)  

    def callback(x, y):
        print(f"Face detected at ({x}, {y})")
        ser.write(str(x).encode(), str(y).encode)
        print(f"Sent to Arduino: {x}")
        time.sleep(3)