import numpy as np
import os
import gzip
from sklearn.cluster import KMeans

'''
torchvisionのEMNIST Digitsデータセットは破損しているので、こちらのサイトを参照してダウンロードする必要があります
https://www.marvinschmitt.com/blog/emnist-manual-loading/
'''

# EMNIST Digitsデータセットの読み込み
def load_emnist():
    # 指定されたパスからデータをロードする
    emnist_path = './EMNIST'
    
    def load_gz_file(file_path, is_image=True):
        with gzip.open(file_path, 'rb') as f:
            if is_image:
                return np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
            else:
                return np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    x_train = load_gz_file(os.path.join(emnist_path, 'emnist-digits-train-images-idx3-ubyte.gz'))
    y_train = load_gz_file(os.path.join(emnist_path, 'emnist-digits-train-labels-idx1-ubyte.gz'), is_image=False)
    x_test = load_gz_file(os.path.join(emnist_path, 'emnist-digits-test-images-idx3-ubyte.gz'))
    y_test = load_gz_file(os.path.join(emnist_path, 'emnist-digits-test-labels-idx1-ubyte.gz'), is_image=False)
    
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

# グレースケール画像に色を適用する関数
def apply_color(image, color):
    # グレースケール画像を0-1の範囲に正規化
    normalized_image = image
    # カラーチャネルを作成
    colored_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        # 各チャネルに色を適用
        colored_image[:, :, i] = (normalized_image * color[i]).astype(np.uint8)
    return colored_image

# 3種類のラベルを作成
def create_colored_emnist(x, y, colors):
    colored_images = []
    digit_labels = []
    color_labels = []
    combined_labels = []
    num_colors = len(colors)
    
    # 各数字に均等に色を割り当てる
    for digit in range(10):
        digit_indices = np.where(y == digit)[0]
        num_images_per_color = len(digit_indices) // num_colors
        remaining = len(digit_indices) % num_colors
        
        start_idx = 0
        for color_idx, color in enumerate(colors):
            end_idx = start_idx + num_images_per_color
            if color_idx < remaining:
                end_idx += 1
            
            for i in digit_indices[start_idx:end_idx]:
                colored_image = apply_color(x[i], color)
                colored_images.append(colored_image)
                digit_labels.append(y[i])  # 元のデータの数字ラベル
                color_labels.append(color_idx)  # 色ラベル
                combined_labels.append(y[i] * 10 + color_idx)  # 複合ラベル
            start_idx = end_idx

    return (np.array(colored_images), 
            np.array(digit_labels), 
            np.array(color_labels), 
            np.array(combined_labels))

(x_train, y_train), (x_test, y_test) = load_emnist()

# クラスタリングによって定義した色（例として10色）
np.random.seed(42)
seed = 42
colors = np.random.randint(0, 256, size=(10000, 3))
kmeans = KMeans(n_clusters=10)
kmeans.fit(colors)
cluster_centers = kmeans.cluster_centers_.astype(int)

# トレーニングセットのカラーEMNISTデータセットを作成
x_train_colored, y_train_digits, y_train_colors, y_train_combined = create_colored_emnist(x_train, y_train, cluster_centers)

# テストセットのカラーEMNISTデータセットを作成
x_test_colored, y_test_digits, y_test_colors, y_test_combined = create_colored_emnist(x_test, y_test, cluster_centers)

# カラーEMNISTデータセットを保存
os.makedirs('colored_EMNIST', exist_ok=True)
np.save('colored_EMNIST/x_train_colored.npy', x_train_colored) # 画像データ
np.save('colored_EMNIST/y_train_digits.npy', y_train_digits) # 数字ラベル
np.save('colored_EMNIST/y_train_colors.npy', y_train_colors) # 色ラベル
np.save('colored_EMNIST/y_train_combined.npy', y_train_combined) # 複合ラベル
np.save('colored_EMNIST/x_test_colored.npy', x_test_colored)
np.save('colored_EMNIST/y_test_digits.npy', y_test_digits)
np.save('colored_EMNIST/y_test_colors.npy', y_test_colors)
np.save('colored_EMNIST/y_test_combined.npy', y_test_combined)
