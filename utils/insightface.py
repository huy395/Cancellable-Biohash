import numpy as np
import cv2
from insightface.app import FaceAnalysis


def load_model(name="buffalo_sc", det_size=(320, 320), ctx_id=0): # det_size(640,640) với kaggle dataset 256,256 với cfp
    """
    Load InsightFace model.
    name: model name ('buffalo_sc', 'antelopev2', etc.)
    det_size: detection resolution
    ctx_id: GPU ID (-1 nếu dùng CPU)
    """
    app = FaceAnalysis(name=name)
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app


def preprocess_image(img_path, gamma=1.2):
    """
    Đọc và chuẩn hóa ảnh theo InsightFace.
    Bao gồm:
        - Cân bằng sáng (illumination normalization)
        - Gamma correction
        - Giữ nguyên kích thước gốc
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")

    # Chuyển RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(img, (640, 640))

    # --- (1) Cân bằng sáng theo kênh YCrCb ---
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)  # cân bằng histogram kênh sáng
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    img_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)

    # --- (2) Gamma correction ---
    img_float = img_eq.astype(np.float32) / 255.0
    img_gamma = np.power(img_float, 1.0 / gamma)
    img_gamma = np.clip(img_gamma * 255.0, 0, 255).astype(np.uint8)

    # --- (3) Normalize về [0,1], giữ nguyên kích thước ---
    img_out = img_gamma.astype(np.float32) / 255.0

    return img_out

def augment_image(img):
    """
    Tạo danh sách ảnh augment từ 1 ảnh RGB uint8.
    """
    aug_images = []

    h, w = img.shape[:2]

    # 1. Original
    aug_images.append(img)

    # 2. Horizontal flip
    aug_images.append(cv2.flip(img, 1))

    # 3. Slight rotations
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        aug_images.append(rotated)

    # 4. Slight brightness change
    for alpha in [0.9, 1.1]:
        bright = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        aug_images.append(bright)

    # 5. Slight crop (scale 95%)
    scale = 0.95
    new_h, new_w = int(h * scale), int(w * scale)
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2
    cropped = img[start_y:start_y+new_h, start_x:start_x+new_w]
    cropped = cv2.resize(cropped, (w, h))
    aug_images.append(cropped)

    return aug_images



def get_embedding(img_path, model):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")

    faces = model.get(img)
    if len(faces) == 0:
        raise ValueError(f"Không phát hiện khuôn mặt trong ảnh: {img_path}")

    face = max(faces, key=lambda f: f.det_score)
    return face.normed_embedding.astype(np.float64)
    
def get_embeddings_with_aug(img_path, model):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    aug_imgs = augment_image(img)

    embeddings = []

    for aug in aug_imgs:
        aug_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
        faces = model.get(aug_bgr)
        if len(faces) == 0:
            continue

        face = max(faces, key=lambda f: f.det_score)
        emb = face.normed_embedding
        embeddings.append(emb.astype(np.float64))

    return np.array(embeddings)  # shape (N_aug, d)

