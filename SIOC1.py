import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
from io import BytesIO

def manual_convolve1d(signal, kernel):
    sig_len = len(signal)
    ker_len = len(kernel)
    res = np.zeros(sig_len)
    pad = ker_len // 2
    padded = np.pad(signal, pad, mode='edge')
    for i in range(sig_len):
        res[i] = np.sum(padded[i:i+ker_len] * kernel[::-1])
    return res

def manual_convolve2d(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    res = np.zeros_like(image)
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode='edge')
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            res[i, j] = np.sum(region * kernel)
    return res

def get_h3_kernel(scale):
    size = 2 * scale + 1
    t = np.linspace(-1, 1, size)
    kernel = 1 - np.abs(t)
    kernel[kernel < 0] = 0
    return kernel / np.sum(kernel)

def get_mean_kernel(factor):
    return np.ones((factor, factor)) / (factor**2)

def interpolate_1d(signal, scale):
    new_len = len(signal) * scale
    upsampled = np.zeros(new_len)
    upsampled[::scale] = signal
    kernel = get_h3_kernel(scale)
    return manual_convolve1d(upsampled, kernel) * scale

def downscale_image(img, factor):
    blurred = manual_convolve2d(img.astype(float), get_mean_kernel(factor))
    return blurred[::factor, ::factor]

def upscale_image(img, scale):
    h, w = img.shape
    temp = np.zeros((h, w * scale))
    for r in range(h):
        temp[r, :] = interpolate_1d(img[r, :], scale)
        
    new_h, new_w = h * scale, w * scale
    final = np.zeros((new_h, new_w))
    for c in range(new_w):
        final[:, c] = interpolate_1d(temp[:, c], scale)
    return final

def calculate_mse(y, y_hat):
    n = len(y.flatten())
    return np.sum((y - y_hat)**2) / n

x = np.linspace(-np.pi, np.pi, 100)
y_orig = np.sin(x)
y_interp = interpolate_1d(y_orig, 10)

url = "https://raw.githubusercontent.com/284095-bot/SIOC1/main/Chess.png"
with urllib.request.urlopen(url) as response:
    img_orig = np.array(Image.open(BytesIO(response.read())).convert('L'))

img_small = downscale_image(img_orig, 2)
img_big = upscale_image(img_small, 2)

print(f"MSE 1D: {calculate_mse(np.sin(np.linspace(-np.pi, np.pi, 1000)), y_interp)}")
print(f"MSE Image: {calculate_mse(img_orig.astype(float), img_big)}")

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(img_orig, cmap='gray')
axs[0].set_title("Oryginal")
axs[1].imshow(img_small, cmap='gray')
axs[1].set_title("Zmniejszony x2")
axs[2].imshow(img_big, cmap='gray')
axs[2].set_title(f"Powiekszony x2\nMSE: {calculate_mse(img_orig.astype(float), img_big):.2f}")
plt.show()
