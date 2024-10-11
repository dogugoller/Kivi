import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("kivi.jpg")
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #imageyi hsv yaptım(rengin daha kolay işlenmesi için)

alt = np.array([25,55,0]) # kivi rengi için renk aralağını belirledim
ust = np.array([45,255,255])

mask = cv2.inRange(image_hsv, alt, ust)
source = cv2.bitwise_and(image, image, mask=mask) #maskede belirlediğim renk aralıklarını tuttum
gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) # gri ton

ret,thresh = cv2.threshold(gray, 0,255, cv2.THRESH_BINARY) # thresh sayesinde arkaplan siyah oldu
blur = cv2.medianBlur(thresh, 35) #gürültü azaltmak için

kernel1 = np.ones((5,5), np.uint8)
kernel2 = np.ones((13,13), np.uint8)

erosion = cv2.erode(blur, kernel1, iterations=1)
dilation = cv2.dilate(erosion, kernel2, iterations=1)

plt.figure(figsize=(15,10))


plt.subplot(1,2,1) # satır, sütun, indeks
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Önce")
plt.axis('off')

plt.subplot(1,2,2) # satır, sütun, indeks
plt.imshow(dilation, cmap="gray")
plt.title("Sonra")
plt.axis('off')

plt.savefig("kiviyeni.png", bbox_inches='tight')  # sonucu kaydet
plt.show()