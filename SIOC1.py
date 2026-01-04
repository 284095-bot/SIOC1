import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from PIL import Image
import urllib.request
from io import BytesIO

# --- KONFIGURACJA ---
# Zmieniono na 15, zeby bylo widac "kanciastosc" interpolacji
N_POINTS = 15

def get_kernel(name, scale):
    num_points = 10 * scale + 1
    t = np.linspace(-num_points//2, num_points//2, num_points)
    
    if name == 'rect': 
        k = np.zeros_like(t)
        k[len(t)//2] = 1.0
    elif name == 'tri': 
        val = np.linspace(-1, 1, num_points)
        k = 1 - np.abs(val)
        k = np.where(k < 0, 0, k)
    elif name == 'sinc': 
        t_sinc = np.linspace(-5, 5, num_points)
        k = np.sinc(t_sinc)
    
    return k / np.sum(k)

def simple_interpolate_1d(signal, scale, kernel_name='tri'):
    upsampled = np.zeros(len(signal) * scale)
    upsampled[::scale] = signal
    kernel = get_kernel(kernel_name, scale)
    result = convolve(upsampled, kernel, mode='same')
    return result * scale

def f1(x): return np.sin(x)
def f2(x): 
    with np.errstate(divide='ignore', invalid='ignore'):
        val = np.sin(1/x)
        val = np.nan_to_num(val)
    return val
def f3(x): return np.sign(np.sin(8*x))

def run_functions():
    x_start = np.linspace(-np.pi, np.pi, N_POINTS)
    
    functions = [
        ("sin(x)", f1, 'sinc'), 
        ("sin(1/x)", f2, 'tri'), 
        ("sgn(sin(8x))", f3, 'rect')
    ]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, func, k_name) in enumerate(functions):
        y_samples = func(x_start)
        
        scale = 10  # Zwiekszamy skale zeby linia byla gladsza miedzy rzadkimi punktami
        y_interp = simple_interpolate_1d(y_samples, scale, k_name)
        x_interp = np.linspace(-np.pi, np.pi, len(y_interp))
        
        y_true = func(x_interp)

        axs[i].plot(x_interp, y_true, color='green', linestyle='--', alpha=0.4, label='Wzorzec')
        axs[i].plot(x_interp, y_interp, color='blue', linewidth=2, label='Interpolacja')
        
        axs[i].set_title(f"{name}\n(Jadro: {k_name})")
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()

def scale_down_avg(image, factor):
    kernel = np.ones((factor, factor)) / (factor**2)
    from scipy.signal import convolve2d
    blurred = convolve2d(image, kernel, mode='same')
    return blurred[::factor, ::factor]

def run_image_scaling(url):
    try:
        with urllib.request.urlopen(url) as response:
            img_data = response.read()
        img = np.array(Image.open(BytesIO(img_data)).convert('RGBA').convert('L'))
    except Exception:
        return

    SCALE = 2
    small = scale_down_avg(img, SCALE)
    h, w = small.shape
    
    temp = np.zeros((h, w * SCALE))
    for r in range(h):
        temp[r, :] = simple_interpolate_1d(small[r, :], SCALE, 'tri')[:w*SCALE]
        
    final = np.zeros((h * SCALE, w * SCALE))
    for c in range(w * SCALE):
        final[:, c] = simple_interpolate_1d(temp[:, c], SCALE, 'tri')[:h*SCALE]
        
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1); plt.imshow(img, cmap='gray'); plt.title("Oryginal"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(small, cmap='gray'); plt.title(f"Zmniejszony x{SCALE}"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(final, cmap='gray'); plt.title(f"Powiekszony x{SCALE}"); plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_functions()
    run_image_scaling("https://raw.githubusercontent.com/284095-bot/SIOC1/main/Chess.png")