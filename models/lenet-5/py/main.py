import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image 

HERE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)
    
from pyv.kernels import conv2d, maxpool, relu, bias_add, dense, tensor_add, softmax

class LeNet5:
    def __init__(self, weights_dir, variant="M8"):
        self.weights_dir = weights_dir
        self.variant = variant
        self.params = {}
        self._load_all_params()

    def _load_bin(self, filename, shape):
        path = os.path.join(self.weights_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weight file not found: {path}")
        data = np.fromfile(path, dtype=np.float32)
        return data.reshape(shape).copy()

    def _load_all_params(self):
        print(f"Loading LeNet-5 parameters from {self.weights_dir}...")
        self.params['c1_w'] = self._load_bin("c1.c1.c1.weight.bin", (6, 1, 5, 5))
        self.params['c1_b'] = self._load_bin("c1.c1.c1.bias.bin", (6,))
        self.params['c2_1_w'] = self._load_bin("c2_1.c2.c2.weight.bin", (16, 6, 5, 5))
        self.params['c2_1_b'] = self._load_bin("c2_1.c2.c2.bias.bin", (16,))
        self.params['c2_2_w'] = self._load_bin("c2_2.c2.c2.weight.bin", (16, 6, 5, 5))
        self.params['c2_2_b'] = self._load_bin("c2_2.c2.c2.bias.bin", (16,))
        self.params['c3_w'] = self._load_bin("c3.c3.c3.weight.bin", (120, 16, 5, 5))
        self.params['c3_b'] = self._load_bin("c3.c3.c3.bias.bin", (120,))
        self.params['f4_w'] = self._load_bin("f4.f4.f4.weight.bin", (84, 120))
        self.params['f4_b'] = self._load_bin("f4.f4.f4.bias.bin", (84,))
        self.params['f5_w'] = self._load_bin("f5.f5.f5.weight.bin", (10, 84))
        self.params['f5_b'] = self._load_bin("f5.f5.f5.bias.bin", (10,))

    def predict(self, image_data, visualize=False):
        """
        image_data: np.array of shape (32, 32)
        """
        # Prepare input: Batch=1, Channel=1, H=32, W=32
        x = image_data.reshape(1, 1, 32, 32).astype(np.float32)

        # --- Layer 1: C1 ---
        x = conv2d(x, self.params['c1_w'], stride=(1, 1), pad=(0, 0), variant=self.variant)
        x = bias_add(x, self.params['c1_b'], variant=self.variant)
        x = relu(x, variant=self.variant)
        x = maxpool(x, 2, 2, stride_h=2, stride_w=2, pad_h=0, pad_w=0, variant=self.variant)

        # --- Parallel Block: C2 ---
        x1 = conv2d(x, self.params['c2_1_w'], stride=(1, 1), pad=(0, 0), variant=self.variant)
        x1 = bias_add(x1, self.params['c2_1_b'], variant=self.variant)
        x1 = relu(x1, variant=self.variant)
        x1 = maxpool(x1, 2, 2, stride_h=2, stride_w=2, variant=self.variant)

        x2 = conv2d(x, self.params['c2_2_w'], stride=(1, 1), pad=(0, 0), variant=self.variant)
        x2 = bias_add(x2, self.params['c2_2_b'], variant=self.variant)
        x2 = relu(x2, variant=self.variant)
        x2 = maxpool(x2, 2, 2, stride_h=2, stride_w=2, variant=self.variant)

        x = tensor_add(x1, x2, variant=self.variant)

        # --- Layer 3: C3 ---
        x = conv2d(x, self.params['c3_w'], stride=(1, 1), pad=(0, 0), variant=self.variant)
        x = bias_add(x, self.params['c3_b'], variant=self.variant)
        x = relu(x, variant=self.variant)

		# --- Layer 4: F4 (120 -> 84) ---
        f4_in = x.flatten().astype(np.float32)
        f4_out = dense(f4_in, self.params['f4_w'], self.params['f4_b'], variant=self.variant)
        f4_activated = relu(f4_out, variant=self.variant)  # (84,)

        # --- Layer 5: F5 (84 -> 10) ---
        logits = dense(f4_activated, self.params['f5_w'], self.params['f5_b'], variant=self.variant)

        # --- Final Output ---
        probs = softmax(logits.flatten())

        if probs.shape[0] != 10:
            print(f"DEBUG: Softmax returned shape {probs.shape}, expected (10,)")

        prediction = np.argmax(probs)

        if visualize:
            self._visualize(image_data, probs)

        return prediction
    
    def _visualize(self, img, probs):
        # Flatten to be safe (ensure shape is (10,))
        probs_flat = probs.flatten()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(img, cmap='gray')
        ax1.set_title("Input Image (32x32)")
        
        ax2.bar(range(10), probs_flat, color='firebrick')
        ax2.set_xticks(range(10))
        ax2.set_title(f"Prediction: {np.argmax(probs_flat)}")
        ax2.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig('lenet5_prediction.png')

def load_image_32x32(path: str) -> np.ndarray:
	"""
	Load an image (PNG/JPEG/...) or raw .bin and return a (32, 32) float32 array.
	For images: convert to grayscale, resize to 32x32, normalize to [0,1].
	For .bin: expect 1024 float32 values.
	"""
	ext = os.path.splitext(path)[1].lower()

	if ext == ".bin":
		data = np.fromfile(path, dtype=np.float32)
		if data.size != 32 * 32:
			raise ValueError(f".bin file must contain 1024 floats, got {data.size}")
		return data.reshape(32, 32)

	# Otherwise treat as image
	img = Image.open(path).convert("L")  # grayscale
	img = img.resize((32, 32), Image.BILINEAR)
	img_np = np.asarray(img, dtype=np.float32)

	# Simple normalization to [0, 1]
	img_np /= 255.0

	return img_np  # shape (32, 32), float32
           
# --- Main Entry Point ---
if __name__ == "__main__":
    HERE = os.path.abspath(__file__)
    LENET_DIR = os.path.dirname(HERE)         
    LENET_ROOT = os.path.dirname(LENET_DIR)   

    WEIGHTS_PATH = os.path.join(LENET_ROOT, "model_parameters")
    model = LeNet5(WEIGHTS_PATH, variant="M8")

    if len(sys.argv) == 2:
        digit = sys.argv[1]
        image_path = os.path.join(LENET_ROOT, "images", f"{digit}.png")
    else:
        image_path = os.path.join(LENET_ROOT, "images", "6.png")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input file not found: {image_path}")

    print(f"Using input file: {image_path}")
    img_2d = load_image_32x32(image_path)

    result = model.predict(img_2d, visualize=True)
    print(f"The model predicted: {result}")
