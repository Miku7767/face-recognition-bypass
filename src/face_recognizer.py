"""
ArcFace 人脸识别模块 —— 基于 InsightFace buffalo_l 预训练模型
支持：人脸库构建、1:N识别、LFW评估
"""
import cv2
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine


class FaceRecognizer:
    def __init__(self, model_name="buffalo_l", det_size=(640, 640), ctx_id=-1):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.gallery_embeddings = []
        self.gallery_labels = []

    # ---------- 人脸库管理 ----------
    def build_gallery(self, gallery_dir: str):
        """遍历 gallery_dir 下每个子文件夹，子文件夹名作为人员ID"""
        gallery_path = Path(gallery_dir)
        self.gallery_embeddings.clear()
        self.gallery_labels.clear()

        for person_dir in sorted(gallery_path.iterdir()):
            if not person_dir.is_dir():
                continue
            person_id = person_dir.name
            for img_path in person_dir.glob("*"):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                emb = self._extract_embedding(str(img_path))
                if emb is not None:
                    self.gallery_embeddings.append(emb)
                    self.gallery_labels.append(person_id)

        print(f"人脸库构建完成：{len(self.gallery_embeddings)} 张人脸，"
              f"{len(set(self.gallery_labels))} 个身份")

    # ---------- 1:N 识别 ----------
    def identify(self, img_path: str, threshold: float = 0.65):
        """返回 (best_person_id, best_similarity) 或 (None, 0)"""
        if not self.gallery_embeddings:
            raise RuntimeError("人脸库为空，请先调用 build_gallery()")

        query_emb = self._extract_embedding(img_path)
        if query_emb is None:
            return None, 0.0

        best_idx, best_sim = -1, -1.0
        for idx, gal_emb in enumerate(self.gallery_embeddings):
            sim = 1.0 - cosine(query_emb, gal_emb)
            if sim > best_sim:
                best_sim, best_idx = sim, idx

        if best_sim >= threshold:
            return self.gallery_labels[best_idx], best_sim
        return None, best_sim

    # ---------- 1:1 验证（供 LFW 评估使用） ----------
    def verify(self, img1_path: str, img2_path: str, threshold: float = 0.65):
        emb1 = self._extract_embedding(img1_path)
        emb2 = self._extract_embedding(img2_path)
        if emb1 is None or emb2 is None:
            return False, 0.0
        sim = 1.0 - cosine(emb1, emb2)
        return sim >= threshold, sim

    # ---------- LFW pairs.txt 评估 ----------
    def evaluate_lfw(self, lfw_dir: str, pairs_file: str):
        """按 LFW View 2 标准协议评估，返回 accuracy。使用缓存加速。"""
        lfw_root = Path(lfw_dir)

        # ---- 第1步：解析所有 pairs，收集需要的图片路径 ----
        tasks = []  # [(img1, img2, is_same), ...]
        with open(pairs_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 3:
                    name, n1, n2 = parts
                    tasks.append((
                        str(lfw_root / name / f"{name}_{int(n1):04d}.jpg"),
                        str(lfw_root / name / f"{name}_{int(n2):04d}.jpg"),
                        True,
                    ))
                elif len(parts) == 4:
                    n1, i1, n2, i2 = parts
                    tasks.append((
                        str(lfw_root / n1 / f"{n1}_{int(i1):04d}.jpg"),
                        str(lfw_root / n2 / f"{n2}_{int(i2):04d}.jpg"),
                        False,
                    ))

        # ---- 第2步：预计算所有唯一图片的 embedding（缓存） ----
        all_paths = set()
        for p1, p2, _ in tasks:
            all_paths.add(p1)
            all_paths.add(p2)

        cache = {}
        print(f"提取 {len(all_paths)} 张图片的特征向量...")
        for i, p in enumerate(sorted(all_paths)):
            cache[p] = self._extract_embedding(p)
            if (i + 1) % 500 == 0 or i + 1 == len(all_paths):
                print(f"\r  特征提取: {i + 1}/{len(all_paths)}", end="", flush=True)
        print()

        # ---- 第3步：逐对比对 ----
        total, correct = 0, 0
        same_dists, diff_dists = [], []
        N = len(tasks)

        for i, (p1, p2, same) in enumerate(tasks):
            emb1, emb2 = cache[p1], cache[p2]
            if emb1 is not None and emb2 is not None:
                sim = 1.0 - cosine(emb1, emb2)
                ok = sim >= 0.65
            else:
                sim, ok = 0.0, False

            total += 1
            if (ok and same) or (not ok and not same):
                correct += 1
            if same:
                same_dists.append(1.0 - sim)
            else:
                diff_dists.append(1.0 - sim)

            if (i + 1) % 300 == 0 or i + 1 == N:
                print(f"\r  比对: {i + 1}/{N}  当前准确率={correct/total:.4f}", end="", flush=True)

        print()
        acc = correct / total if total else 0
        print(f"LFW 评估完成：{correct}/{total} = {acc:.4f}")

        if same_dists:
            print(f"  同人余弦距离  mean={np.mean(same_dists):.4f}  std={np.std(same_dists):.4f}")
        if diff_dists:
            print(f"  不同人余弦距离 mean={np.mean(diff_dists):.4f}  std={np.std(diff_dists):.4f}")

        return acc, same_dists, diff_dists

    # ---------- 内部方法 ----------
    def _extract_embedding(self, img_path: str):
        # 用 np.fromfile + cv2.imdecode 替代 cv2.imread，
        # 解决 Windows 上 cv2.imread 不支持中文路径的问题
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [警告] 无法读取图像: {img_path}")
            return None
        faces = self.app.get(img)
        if len(faces) == 0:
            print(f"  [警告] 未检测到人脸: {img_path}")
            return None
        return faces[0].normed_embedding


# ==================== 命令行入口 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ArcFace 人脸识别")
    sub = parser.add_subparsers(dest="cmd")

    # build-gallery
    p_build = sub.add_parser("build-gallery")
    p_build.add_argument("--gallery", required=True)
    p_build.add_argument("--output", default="outputs/gallery.npz")

    # identify
    p_id = sub.add_parser("identify")
    p_id.add_argument("--gallery", required=True)
    p_id.add_argument("--query", required=True)
    p_id.add_argument("--threshold", type=float, default=0.65)

    # evaluate-lfw
    p_lfw = sub.add_parser("evaluate-lfw")
    p_lfw.add_argument("--lfw-dir", required=True, help="LFW 原始数据集目录")
    p_lfw.add_argument("--pairs", required=True, help="pairs.txt 路径")

    args = parser.parse_args()
    fr = FaceRecognizer()

    if args.cmd == "build-gallery":
        fr.build_gallery(args.gallery)
        np.savez(args.output,
                 embeddings=np.array(fr.gallery_embeddings),
                 labels=np.array(fr.gallery_labels))
        print(f"已保存人脸库至 {args.output}")

    elif args.cmd == "identify":
        fr.build_gallery(args.gallery)
        person, sim = fr.identify(args.query, args.threshold)
        if person:
            print(f"识别成功: {person}  相似度={sim:.4f}")
        else:
            print(f"未识别到库中人  最高相似度={sim:.4f}")

    elif args.cmd == "evaluate-lfw":
        fr.evaluate_lfw(args.lfw_dir, args.pairs)

    else:
        # 默认：交互演示
        print("=" * 55)
        print("ArcFace 人脸识别演示（InsightFace buffalo_l）")
        print("=" * 55)
        demo_gallery = "data/gallery"
        demo_query = "data/query"
        if Path(demo_gallery).exists():
            fr.build_gallery(demo_gallery)
            if fr.gallery_embeddings:
                for q in Path(demo_query).glob("*"):
                    if q.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        person, sim = fr.identify(str(q))
                        label = person or "陌生人"
                        print(f"  {q.name:30s} → {label:15s} (相似度={sim:.4f})")
            else:
                print("人脸库为空，请先在 data/gallery/ 下放置人员子文件夹")
        else:
            print("请先创建 data/gallery/ 和 data/query/ 目录并放入图片")
