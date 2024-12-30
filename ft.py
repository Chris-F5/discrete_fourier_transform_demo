import numpy as np
import cv2

def compress(factor):
    lenna = cv2.imread('lenna.png')
    grey_lenna = cv2.cvtColor(lenna, cv2.COLOR_BGR2GRAY)

    #image = np.array([[1,2,1],
    #                  [1,3,1],
    #                  [1,2,1]])
    image = np.array(grey_lenna)
    print(image.dtype)
    assert(len(image.shape) == 2)
    assert(image.shape[0] == image.shape[1])
    L = image.shape[0]
    print(L)

    x = np.arange(L)
    xx = x.reshape((L,1)) @ x.reshape((1,L))
    Enkx = np.exp(-2j*(np.pi/L)*xx)
    Epkx = np.exp(+2j*(np.pi/L)*xx)

    transform = Enkx @ image @ Enkx

    # compress
    #factor = 1
    max_freq = int(L * factor)
    mask = (x<=max_freq).reshape((L,1)) @ (x<=max_freq).reshape((1,L))
    transform = transform * mask

    reverse = (1/L**2) * (Epkx @ transform @ Epkx)

    cv2.imwrite(f"out{factor}.png", np.real(reverse).astype(np.uint8))

for factor in [0.02, 0.05, 0.2, 0.8]:
    compress(factor)

