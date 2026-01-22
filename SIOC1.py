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

def get_kernel(name, scale):
    size = 2 * scale + 1
    t = np.linspace(-2, 2, size)
    if name == 'triangle':
        k = 1 - np.abs(t)
    elif name == 'sinc':
        k = np.sinc(t)
    elif name == 'cubic':
        a = -0.5
        abs_t = np.abs(t)
        k = np.where(abs_t < 1, (a+2)*abs_t**3 - (a+3)*abs_t**2 + 1,
            np.where(abs_t < 2, a*abs_t**3 - 5*a*abs_t**2 + 8*a*abs_t - 4*a, 0))
    else:
        k = np.ones(size)
    k[k < -1] = -1 # Clipping dla stabilności wizualnej
    return k / np.sum(k)

def interpolate_1d(signal, scale, kernel_name='triangle'):
    new_len = len(signal) * scale
    upsampled = np.zeros(new_len)
    upsampled[::scale] = signal
    kernel = get_kernel(kernel_name, scale)
    return manual_convolve1d(upsampled, kernel) * scale

def calculate_mse(y, y_hat):
    return np.mean((y - y_hat)**2)

functions = {
    "sin(x)": lambda x: np.sin(x),
    "sin(1/x)": lambda x: np.sin(1/(x + 1e-6)),
    "sign(sin(8x))": lambda x: np.sign(np.sin(8*x))
}

kernels = ['triangle', 'sinc', 'cubic']
x_nodes = np.linspace(-np.pi, np.pi, 100)
x_fine = np.linspace(-np.pi, np.pi, 1000)

for f_name, f in functions.items():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Funkcja {f_name}")
    y_nodes = f(x_nodes)
    y_true = f(x_fine)
    for i, k_name in enumerate(kernels):
        y_interp = interpolate_1d(y_nodes, 10, k_name)
        mse = calculate_mse(y_true, y_interp)
        axs[i].plot(x_fine, y_true, 'k--', alpha=0.3, label='Oryginał')
        axs[i].plot(x_fine, y_interp, 'r', label='Interp')
        axs[i].scatter(x_nodes, y_nodes, c='k', s=5, label='Węzły')
        axs[i].set_title(f"jadro: {k_name}\nMSE: {mse:.5f}")
    plt.tight_layout()

url = "https://raw.githubusercontent.com/284095-bot/SIOC1/main/Chess.png"
with urllib.request.urlopen(url) as response:
    img_orig = np.array(Image.open(BytesIO(response.read())).convert('L'))

img_small = manual_convolve2d(img_orig.astype(float), np.ones((2,2))/4)[::2, ::2]

h, w = img_small.shape
temp = np.zeros((h, w * 2))
for r in range(h):
    temp[r, :] = interpolate_1d(img_small[r, :], 2, 'triangle')
final = np.zeros((h * 2, w * 2))
for c in range(w * 2):
    final[:, c] = interpolate_1d(temp[:, c], 2, 'triangle')

mse_img = calculate_mse(img_orig.astype(float), final)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(img_orig, cmap='gray'); axs[0].set_title("Oryginal")
axs[1].imshow(img_small, cmap='gray'); axs[1].set_title("Zmniejszony x2")
axs[2].imshow(final, cmap='gray'); axs[2].set_title(f"Powiekszony x2\nMSE: {mse_img:.2f}")
plt.show()
