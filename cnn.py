import numpy as np

# Fungsi aktivasi ReLU
def relu(x):
    return np.maximum(0, x)

# Fungsi konvolusi 2D
def convolution2d(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i+kernel_height, j:j+kernel_width] * kernel)

    return output

# Fungsi pooling maksimum
def max_pooling2d(image, pool_size):
    image_height, image_width = image.shape
    pool_height, pool_width = pool_size
    output_height = image_height // pool_height
    output_width = image_width // pool_width
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.max(image[i*pool_height:(i+1)*pool_height, j*pool_width:(j+1)*pool_width])

    return output

# Definisi kernel dan bobot
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])
w_fc = np.random.randn(169, 10)
b_fc = np.random.randn(10)

# Fungsi forward propagation
def forward_propagation(image):
    # Konvolusi lapisan pertama
    conv1 = convolution2d(image, kernel)
    relu1 = relu(conv1)
    # Pooling maksimum
    pool1 = max_pooling2d(relu1, (2, 2))
    # Flatten hasil pooling
    flatten = pool1.flatten()
    # Fully connected layer
    fc = np.dot(flatten, w_fc) + b_fc
    return fc

# Contoh penggunaan
image = np.random.randn(28, 28)  # Gambar ukuran 28x28
output = forward_propagation(image)
print("Output:", output)
