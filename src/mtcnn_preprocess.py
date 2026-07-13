"""
MTCNN 人脸检测 + 对齐预处理模块
替代 Dlib 方案，使用 MTCNN 进行更精准的人脸检测和关键点定位
"""
import cv2
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN, extract_face
from PIL import Image


class MTCNNPreprocessor:
    """MTCNN 人脸预处理：检测、对齐、归一化"""

    def __init__(self, target_size=(160, 160), margin=20, device="cpu"):
        self.target_size = target_size
        self.margin = margin
        self.detector = MTCNN(
            image_size=target_size[0],
            margin=margin,
            keep_all=False,
            post_process=True,
            device=device,
        )

    def detect_and_align(self, img_path: str) -> np.ndarray | None:
        """检测人脸并返回对齐裁剪后的图像数组 (H, W, 3) RGB"""
        try:
            img = Image.open(img_path).convert("RGB")
            face = self.detector(img)  # 返回对齐后的人脸 tensor [3, H, W] 或 None
            if face is None:
                return None
            # 转回 numpy array [H, W, 3]
            face_np = face.permute(1, 2, 0).numpy()
            face_np = (face_np * 255).clip(0, 255).astype(np.uint8)
            return face_np
        except Exception as e:
            print(f"  [MTCNN] 处理失败 {img_path}: {e}")
            return None

    def batch_process(self, input_dir: str, output_dir: str):
        """批量预处理整个目录，保持子目录结构"""
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        total, success = 0, 0
        for img_file in in_path.rglob("*"):
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            total += 1
            result = self.detect_and_align(str(img_file))
            if result is not None:
                rel_path = img_file.relative_to(in_path)
                save_to = out_path / rel_path
                save_to.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_to), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                success += 1

        print(f"MTCNN 批量处理完成: {success}/{total} 张人脸检测成功")

    def extract_embedding_ready(self, img_path: str) -> np.ndarray | None:
        """检测对齐后返回归一化到 [-1, 1] 的 tensor 数组 (3, H, W)"""
        try:
            img = Image.open(img_path).convert("RGB")
            face = self.detector(img)
            if face is None:
                return None
            return face.numpy()
        except Exception:
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MTCNN 人脸预处理")
    parser.add_argument("--input", required=True, help="输入目录")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--size", type=int, default=160, help="目标尺寸")
    args = parser.parse_args()

    pp = MTCNNPreprocessor(target_size=(args.size, args.size))
    pp.batch_process(args.input, args.output)
