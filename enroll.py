import os
import numpy as np
import argparse
import hashlib
import torch
from AVET import AVET
from FEHash import FEHash

from utils.insightface import load_model, get_embedding, get_embeddings_with_aug
from utils.data import FEHashDataset
from utils.seed_manager import SeedManager
import sys
print("[DEBUG] argv:", sys.argv)

def enroll(
    dataset,
    user_label,
    s,
    K,
    D,
    p,
    save_folder,
    model,
    seed_mgr,
    projection_type="orf",
    use_avet=True,
    svm_c=0.01
):
    """
    Enrollment một user:
    - AVET  : lưu seed thay vì ma trận R, A, B
    - FEHash: lưu seed thay vì Omega, r — chỉ w, b lưu trực tiếp
    """

    user_folder = os.path.join(save_folder, user_label)
    os.makedirs(user_folder, exist_ok=True)

    assert len(s) >= K, "Bitstring shorter than K"

    # ── USER SEED ────────────────────────────────────────────────────────
    user_seed = seed_mgr.get_user_seed(user_label)
    print(f"\n[ENROLL] User={user_label} | user_seed={user_seed}")

    # ── SELECT DATA ──────────────────────────────────────────────────────
    user_img_path, padding_dict = dataset.select_user_and_padding(
        user_label, P=p, seed=user_seed
    )

    # ── USER EMBEDDINGS ──────────────────────────────────────────────────
    user_embs = get_embeddings_with_aug(user_img_path, model).astype(np.float64)
    print(f"[INFO] User embeddings: {user_embs.shape}")

    # ── PADDING EMBEDDINGS ───────────────────────────────────────────────
    padding_embs = {}
    for pad_id, img_path in padding_dict.items():
        padding_embs[pad_id] = get_embedding(img_path, model).astype(np.float64)

    print(f"[DEBUG] user_emb shape : {user_embs[0].shape}")
    print(f"[DEBUG] padding count  : {len(padding_embs)}")

    # ── AVET ─────────────────────────────────────────────────────────────
    if use_avet:
        # Tạo AVET riêng cho từng user — seed quyết định R, A, B
        avet = AVET(
            m=512,
            n=512 // 2,
            user_seed=user_seed,
            save_dir=save_folder,
        )
        
        avet._save_params(user_id=user_label)

        # Biến đổi user embeddings
        new_user_embs = []
        for emb in user_embs:
            y = avet.enroll(emb, user_id=user_label, save=False)
            y = y / np.linalg.norm(y)
            new_user_embs.append(y)
        user_embs = np.array(new_user_embs)         # (N, 256)

        # Biến đổi padding embeddings
        new_padding_embs = {}
        for pid, emb in padding_embs.items():
            y = avet.enroll(emb, user_id=None, save=False)
            y = y / (np.linalg.norm(y) + 1e-9)
            new_padding_embs[pid] = y
        padding_embs = new_padding_embs

        print(f"[INFO] AVET applied — output dim: {user_embs.shape[1]}")

    # ── TRAIN K BITS ─────────────────────────────────────────────────────
    fehash_list = []

    for bit_idx in range(K):

        bit_seed = seed_mgr.get_bit_seed(user_label, bit_idx)
        bit_val  = int(s[bit_idx])

        print(f"\n[BIT {bit_idx}] seed={bit_seed} | target_bit={bit_val}")

        # Build training set
        X_train, Y_train = [], []

        # User samples
        for emb in user_embs:
            X_train.append(emb)
            Y_train.append(1 if bit_val == 1 else -1)

        # Padding samples — cân bằng pos/neg
        rng = np.random.default_rng(bit_seed)
        pad_ids = list(padding_embs.keys())
        rng.shuffle(pad_ids)

        num_pos = sum(1 for y in Y_train if y ==  1)
        num_neg = sum(1 for y in Y_train if y == -1)

        for pid in pad_ids:
            X_train.append(padding_embs[pid])
            if num_pos < num_neg:
                Y_train.append(1);  num_pos += 1
            else:
                Y_train.append(-1); num_neg += 1

        X_train = np.array(X_train, dtype=np.float64)
        Y_train = np.array(Y_train)

        print(f"[DEBUG] label dist: {np.unique(Y_train, return_counts=True)}")

        # FEHash — seed=bit_seed → Omega, r tái tạo được khi load
        fe = FEHash(
            projection_type=projection_type,
            D=D,
            d=user_embs.shape[1],
            seed=bit_seed,          # dùng để tái tạo Omega, r
            random_state=bit_seed,  # seed SVM
        )
        fe.fit(X_train, Y_train)

        print(f"[DEBUG] Norm w: {np.linalg.norm(fe.w):.4f}")

        # Lưu seed + w + b (không lưu Omega, r)
        fe.save(os.path.join(user_folder, f"bit_{bit_idx}.pkl.zst"))
        fehash_list.append(fe)

    # ── BITSTRING + SHA3 ─────────────────────────────────────────────────
    #sample_emb = user_embs[0]

    #bitstring = "".join(str(hf.hash(sample_emb)) for hf in fehash_list)

    # Lưu target bitstring
    with open(os.path.join(user_folder, "bitstring.txt"), "w") as f:
        f.write(s[:K])

    # Lưu hash của predicted bitstring
    sha3_hash = hashlib.sha3_256(s[:K].encode("utf-8")).hexdigest()
    with open(os.path.join(user_folder, "bitstring_sha3_256.txt"), "w") as f:
        f.write(sha3_hash)

    #print(f"Target s    : {s[:32]}")
    #print(f"Hash result : {bitstring[:32]}")
    print(f"[DONE] User={user_label} | bits={len(bitstring)} | sha={sha3_hash[:8]}...")

    return bitstring, fehash_list


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FEHash Enrollment")

    parser.add_argument("--data_root", required=True, help="Dataset root folder")
    parser.add_argument("--bitstring_file", required=True, help="User bitstrings")
    parser.add_argument("--save_folder", default="templates", help="Save templates")
    parser.add_argument("--model_name", default="buffalo_sc",  help="InsightFace model")
    parser.add_argument("--K", type=int, default=128,  help="Number of hash bits")
    parser.add_argument("--D", type=int, default=512,  help="Projection dimension")
    parser.add_argument("--p", type=int, default=21,  help="Padding users")
    parser.add_argument("--projection",  choices=["rff", "orf"], default="orf")
    parser.add_argument("--global_seed",  type=int, default=42)
    parser.add_argument("--use_avet", action="store_true", help="Enable AVET")
    parser.add_argument("--svm_c", type=float, default=0.01, help="SVM regularization C")

    args = parser.parse_args()

    ctx_id = 0 if torch.cuda.is_available() else -1
    model  = load_model(args.model_name, ctx_id=ctx_id)

    dataset = FEHashDataset(args.data_root)
    seed_mgr = SeedManager(args.global_seed)
    os.makedirs(args.save_folder, exist_ok=True)

    # Load bitstrings
    with open(args.bitstring_file, "r") as f:
        bitstrings = [line.strip() for line in f if line.strip()]

    user_labels = dataset.get_all_labels()
    assert len(bitstrings) >= len(user_labels), "Not enough bitstrings for all users"

    # Enroll tất cả users
    for idx, user_label in enumerate(user_labels):

        user_seed = seed_mgr.get_user_seed(user_label)
        rng = np.random.default_rng(user_seed)
        s   = rng.choice(bitstrings)

        print("=" * 60)
        print(f"Enroll user {idx+1}/{len(user_labels)} : {user_label}")

        enroll(
            dataset = dataset,
            user_label = user_label,
            s = s,
            K = args.K,
            D = args.D,
            p = args.p,
            save_folder = args.save_folder,
            model = model,
            seed_mgr  = seed_mgr,
            svm_c  = args.svm_c,
            projection_type = args.projection,
            use_avet = args.use_avet,
        )

    print("\n[ALL USERS ENROLLED SUCCESSFULLY]")
