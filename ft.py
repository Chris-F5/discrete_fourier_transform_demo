import numpy as np
import cv2

def ft_transform(image):
    assert(len(image.shape) == 2)
    assert(image.shape[0] == image.shape[1])
    L = image.shape[0]

    x = np.arange(L)
    xx = x.reshape((L,1)) @ x.reshape((1,L))
    Enkx = np.exp(-2j*(np.pi/L)*xx)

    transform = Enkx @ image @ Enkx

    return transform

def ft_inverse(transform):
    assert(len(transform.shape) == 2)
    assert(transform.shape[0] == transform.shape[1])

    x = np.arange(L)
    xx = x.reshape((L,1)) @ x.reshape((1,L))
    Epkx = np.exp(+2j*(np.pi/L)*xx)

    reverse = (1/L**2) * (Epkx @ transform @ Epkx)
    return reverse

if __name__ == '__main__':
    lenna = cv2.imread('lenna.png')
    grey_lenna = cv2.cvtColor(lenna, cv2.COLOR_BGR2GRAY)
    image = np.array(grey_lenna)
    transform = ft_transform(image)
    for factor in [0.02, 0.05, 0.2, 0.8, 1.0]:
        L = transform.shape[0]
        max_freq = int(L * factor)
        x = np.arange(L)
        mask = (x<=max_freq).reshape((L,1)) @ (x<=max_freq).reshape((1,L))
        compressed_transform = transform * mask

        reverse = ft_inverse(compressed_transform)

        cv2.imwrite(f"out{factor}.png", np.real(reverse).astype(np.uint8))
