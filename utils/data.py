import os
import numpy as np

class FEHashDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        # Load user folders
        self.user_labels = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        if len(self.user_labels) == 0:
            raise RuntimeError("Không tìm thấy user folder nào.")

        print(f"[INFO] Loaded {len(self.user_labels)} users.")

    def get_images_of_user(self, user_label):
        user_dir = os.path.join(self.root_dir, user_label)
        imgs = [
            os.path.join(user_dir, f)
            for f in os.listdir(user_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        imgs.sort()

        if len(imgs) == 0:
            raise RuntimeError(f"User {user_label} không có ảnh.")

        return imgs

    def select_user_and_padding(self, user_label, P=21, seed=None):
        """
        Chọn:
        - 1 ảnh genuine của user
        - (P-1) padding identities
        - mỗi padding identity 1 ảnh
        """

        if P > len(self.user_labels):
            raise ValueError("P lớn hơn số user hiện có.")
            
        

        rng = np.random.default_rng(seed)

        # ===== Genuine =====
        #user_imgs = self.get_images_of_user(user_label)
        #genuine_img = rng.choice(user_imgs)
        
        if hasattr(self, "custom_user_image") and user_label in self.custom_user_image:
            genuine_img = self.custom_user_image[user_label]
        else:
            genuine_img, vrf_imgs = self.split(user_label, seed)        
        
            print(genuine_img in vrf_imgs)

        # ===== Padding identities =====
        padding_candidates = [u for u in self.user_labels if u != user_label]
        padding_identities = rng.choice(padding_candidates, size=P-1, replace=False)

        padding_images = {}

        for pad_id in padding_identities:
            pad_imgs = self.get_images_of_user(pad_id)
            padding_images[pad_id] = rng.choice(pad_imgs)

        print(f"\n[DEBUG] USER: {user_label}")
        print(f"[DEBUG] Genuine image: {genuine_img}")
        print(f"[DEBUG] Padding identities: {len(padding_images)}")

        return genuine_img, padding_images
        
    def get_all_labels(self):
        return self.user_labels
    
    def split(self, user_label, random_seed = 42):
        
        rng = np.random.default_rng(random_seed)
        imgs = list(self.get_images_of_user(user_label))
        rng.shuffle(imgs)
        
        #chọn ảnh đầu sau khi shuffle cho enroll, còn lại để verify
        enr_img = imgs[0]
        vrf_imgs = imgs[1:]
        
        return enr_img, vrf_imgs
    
    def override_user_image(self, user_label, img_path):
        self.custom_user_image = {user_label: img_path}

def build_test_dataset(dataset, n_pos=2, n_neg=2, random_seed = 42):
    rng = np.random.default_rng(random_seed)
    result = {}

    labels = dataset.get_all_labels()

    for label in labels:
        
        #Split lấy verify
        enroll_img, verify_imgs = dataset.split(label, random_seed)

        #if len(user_images) < n_pos:
            #continue

        pos_imgs = rng.choice(verify_imgs, size = n_pos, replace = False)

        other_labels = [l for l in labels if l != label]
        neg_imgs = []

        for _ in range(n_neg):
            neg_user = rng.choice(other_labels)
            neg_image = rng.choice(dataset.get_images_of_user(neg_user))
            neg_imgs.append(neg_image)

        result[label] = {
            "pos": pos_imgs,
            "neg": neg_imgs
        }

    return result