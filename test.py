
import cv2
img = cv2.imread('assets/La_12.png')
h, w = img.shape[:2]
print(f'Imagen: {w}x{h}')
cv2.imshow('La_12', img)
cv2.setMouseCallback('La_12', lambda e, x, y, f, p: print(f'x={x}, y={y}') if e==1 else None)
cv2.waitKey(0)