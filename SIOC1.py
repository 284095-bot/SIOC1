import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, convolve2d
from PIL import Image
import urllib.request
from io import BytesIO

X_MIN, X_MAX = -np.pi, np.pi
N_SAMPLES = 100

functions = {
    "sin(x)": lambda x: np.sin(x),
    "sin(1/x)": lambda x: np.sin(1 / (x + 1e-6)),
    "sign(sin(8x))": lambda x: np.sign(np.sin(8 * x))
}

kernels_task1 = {
    "triangle": lambda t: np.clip(1 - np.abs(t), 0, None),
    "sinc": np.sinc,
    "cubic": lambda t: np.where(
        np.abs(t) <= 1, 1.5 * np.abs(t)**3 - 2.5 * np.abs(t)**2 + 1,
        np.where((np.abs(t) <= 2), -0.5 * np.abs(t)**3 + 2.5 * np.abs(t)**2 - 4 * np.abs(t) + 2, 0)
    )
}

def interpolate_signal(Y, x_nodes, h, d):
    def interp_func(t_targets):
        diff = (t_targets[None, :] - x_nodes[:, None]) / d
        vals = h(diff).astype(float)
        weights = np.sum(vals, axis=0)
        weights[weights == 0] = 1.0
        return np.sum(Y[:, None] * vals, axis=0) / weights
    return interp_func

def mse_criterion(f_true, f_interp, x_vals):
    return np.mean((f_true(x_vals) - f_interp(x_vals)) ** 2)

def run_functions():
    x = np.linspace(X_MIN, X_MAX, N_SAMPLES)
    d = (X_MAX - X_MIN) / (N_SAMPLES - 1)
    
    for f_name, f in functions.items():
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Funkcja {f_name}", fontsize=16)
        
        Y = f(x)
        col_idx = 0
        for h_name, h in kernels_task1.items():
            scale = 10
            x_gen = np.linspace(X_MIN, X_MAX, N_SAMPLES * scale)
            
            interp = interpolate_signal(Y, x, h, d)
            y_gen = interp(x_gen)
            
            mse = mse_criterion(f, interp, x_gen)
            
            ax = axs[col_idx]
            x_fine = np.linspace(X_MIN, X_MAX, N_SAMPLES * 20)
            ax.plot(x_fine, f(x_fine), 'k--', alpha=0.3, label='Oryginal')
            ax.scatter(x, Y, color='black', s=10, label='Wezly')
            ax.plot(x_gen, y_gen, 'r-', linewidth=1.5, label='Interp')
            
            ax.set_title(f"Jadro: {h_name}\nMSE: {mse:.5f}", fontsize=10)
            if col_idx == 0:
                ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            col_idx += 1
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
    plt.show()

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

def scale_down_avg(image, factor):
    kernel = np.ones((factor, factor)) / (factor**2)
    blurred = convolve2d(image, kernel, mode='same')
    return blurred[::factor, ::factor]

def run_image_scaling(url):
    with urllib.request.urlopen(url) as response:
        img_data = response.read()
    img = np.array(Image.open(BytesIO(img_data)).convert('RGBA').convert('L'))

    SCALE = 2
    small = scale_down_avg(img, SCALE)
    h, w = small.shape
    
    temp = np.zeros((h, w * SCALE))
    for r in range(h):
        temp[r, :] = simple_interpolate_1d(small[r, :], SCALE, 'tri')[:w*SCALE]
        
    final = np.zeros((h * SCALE, w * SCALE))
    for c in range(w * SCALE):
        final[:, c] = simple_interpolate_1d(temp[:, c], SCALE, 'tri')[:h*SCALE]

    min_h = min(img.shape[0], final.shape[0])
    min_w = min(img.shape[1], final.shape[1])
    
    img_crop = img[:min_h, :min_w]
    final_crop = final[:min_h, :min_w]
    
    mse_img = np.mean((img_crop - final_crop) ** 2)
        
    plt.figure(figsize=(12, 6))
    plt.suptitle("Zadanie 2: Skalowanie obrazu", fontsize=16)
    plt.subplot(1, 3, 1); plt.imshow(img, cmap='gray'); plt.title("Oryginal"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(small, cmap='gray'); plt.title(f"Zmniejszony x{SCALE}\n{small.shape}"); plt.axis('off')
    
    plt.subplot(1, 3, 3); plt.imshow(final, cmap='gray'); plt.title(f"Powiekszony x{SCALE}\nMSE: {mse_img:.2f}"); plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_functions()
    run_image_scaling("https://raw.githubusercontent.com/284095-bot/SIOC1/main/Chess.png")
