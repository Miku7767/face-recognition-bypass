"""
商业人脸识别 API 测试 —— 腾讯云 / 阿里云 / 百度云 / 旷视 Face++

使用前需在各平台注册账号并获取 API Key/Secret：
  腾讯云: https://console.cloud.tencent.com/cam/capi
  阿里云: https://ram.console.aliyun.com/manage/ak
  百度云: https://console.bce.baidu.com/iam/#/iam/accesslist
  旷视Face++: https://console.faceplusplus.com.cn

使用方式：
  python commercial_api_test.py --provider tencent --img1 a.jpg --img2 b.jpg
"""
import hashlib
import hmac
import json
import time
import base64
from pathlib import Path

import requests


# ==================== 腾讯云 ====================
class TencentFaceAPI:
    API_URL = "https://iai.tencentcloudapi.com"

    def __init__(self, secret_id: str, secret_key: str, region: str = "ap-guangzhou"):
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.region = region

    def compare_faces(self, img1_base64: str, img2_base64: str, quality: int = 4):
        """1:1 人脸比对"""
        params = {
            "ImageA": img1_base64,
            "ImageB": img2_base64,
            "QualityControl": quality,
        }
        return self._call("CompareFace", params)

    def detect_liveness(self, img_base64: str):
        """静默活体检测"""
        params = {"ImageBase64": img_base64}
        return self._call("DetectLiveFace", params)

    def _call(self, action: str, params: dict):
        payload = json.dumps(params)
        timestamp = int(time.time())
        date = time.strftime("%Y-%m-%d", time.gmtime(timestamp))

        # 签名 v3
        service = "iai"
        host = "iai.tencentcloudapi.com"
        algorithm = "TC3-HMAC-SHA256"
        http_method = "POST"
        canonical_uri = "/"
        canonical_querystring = ""
        ct = "application/json; charset=utf-8"
        canonical_headers = (
            f"content-type:{ct}\nhost:{host}\nx-tc-action:{action.lower()}\n"
        )
        signed_headers = "content-type;host;x-tc-action"
        hashed_payload = hashlib.sha256(payload.encode()).hexdigest()
        canonical_request = (
            f"{http_method}\n{canonical_uri}\n{canonical_querystring}\n"
            f"{canonical_headers}\n{signed_headers}\n{hashed_payload}"
        )

        credential_scope = f"{date}/{service}/tc3_request"
        hashed_request = hashlib.sha256(canonical_request.encode()).hexdigest()
        string_to_sign = f"{algorithm}\n{timestamp}\n{credential_scope}\n{hashed_request}"

        def sign(key, msg):
            return hmac.new(key, msg.encode(), hashlib.sha256).digest()

        secret_date = sign(("TC3" + self.secret_key).encode(), date)
        secret_service = sign(secret_date, service)
        secret_signing = sign(secret_service, "tc3_request")
        signature = hmac.new(
            secret_signing, string_to_sign.encode(), hashlib.sha256
        ).hexdigest()

        authorization = (
            f"{algorithm} Credential={self.secret_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )

        headers = {
            "Authorization": authorization,
            "Content-Type": ct,
            "Host": host,
            "X-TC-Action": action,
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": "2020-03-03",
            "X-TC-Region": self.region,
        }

        resp = requests.post(self.API_URL, headers=headers, data=payload)
        return resp.json()


# ==================== 百度云 ====================
class BaiduFaceAPI:
    TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
    API_URL = "https://aip.baidubce.com/rest/2.0/face/v3"

    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self._token = None

    def _get_token(self) -> str:
        if self._token:
            return self._token
        resp = requests.post(
            self.TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key,
            },
        )
        self._token = resp.json()["access_token"]
        return self._token

    def compare_faces(self, img1_base64: str, img2_base64: str):
        """1:1 人脸比对"""
        return requests.post(
            f"{self.API_URL}/match?access_token={self._get_token()}",
            json=[
                {
                    "image": img1_base64,
                    "image_type": "BASE64",
                    "face_type": "LIVE",
                    "quality_control": "NORMAL",
                },
                {
                    "image": img2_base64,
                    "image_type": "BASE64",
                    "face_type": "LIVE",
                    "quality_control": "NORMAL",
                },
            ],
        ).json()

    def detect_liveness(self, img_base64: str):
        """活体检测"""
        return requests.post(
            f"{self.API_URL}/faceverify?access_token={self._get_token()}",
            json=[{"image": img_base64, "image_type": "BASE64", "face_field": "liveness"}],
        ).json()


# ==================== Face++ ====================
class FacePlusPlusAPI:
    API_URL = "https://api-cn.faceplusplus.com/facepp/v3"

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret

    def compare_faces(self, img1_base64: str, img2_base64: str):
        """1:1 人脸比对"""
        return requests.post(
            f"{self.API_URL}/compare",
            data={
                "api_key": self.api_key,
                "api_secret": self.api_secret,
                "image_base64_1": img1_base64,
                "image_base64_2": img2_base64,
            },
        ).json()

    def detect_faces(self, img_base64: str):
        """人脸检测（含68点关键点+3D头姿）"""
        return requests.post(
            f"{self.API_URL}/detect",
            data={
                "api_key": self.api_key,
                "api_secret": self.api_secret,
                "image_base64": img_base64,
                "return_attributes": "gender,age,smiling,headpose,facequality",
                "return_landmark": 2,
            },
        ).json()


# ==================== 阿里云 ====================
class AlibabaFaceAPI:
    """阿里云视觉智能平台 人脸比对"""

    def __init__(self, access_key_id: str, access_key_secret: str):
        self.ak_id = access_key_id
        self.ak_secret = access_key_secret

    def compare_faces(self, img1_url: str, img2_url: str):
        """阿里云使用 URL 方式"""
        import urllib.parse

        params = {
            "Format": "JSON",
            "Version": "2019-12-30",
            "AccessKeyId": self.ak_id,
            "SignatureMethod": "HMAC-SHA1",
            "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "SignatureVersion": "1.0",
            "SignatureNonce": str(int(time.time() * 1000)),
            "Action": "CompareFace",
            "ImageURLA": img1_url,
            "ImageURLB": img2_url,
        }
        # 简化签名示例（实际使用需完整实现阿里云签名算法）
        return {"status": "N/A", "note": "阿里云需使用SDK或完整签名实现，建议直接用阿里云Python SDK"}


# ==================== 便捷工具 ====================
def image_to_base64(img_path: str) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def benchmark(api, img1: str, img2: str, rounds: int = 10):
    """性能基准测试"""
    b64_1 = image_to_base64(img1)
    b64_2 = image_to_base64(img2)
    latencies = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        api.compare_faces(b64_1, b64_2)
        latencies.append((time.perf_counter() - t0) * 1000)
    print(f"  平均响应: {np.mean(latencies):.1f}ms  "
          f"P50={np.median(latencies):.1f}ms  "
          f"P99={np.percentile(latencies, 99):.1f}ms")


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="商用API人脸识别测试")
    parser.add_argument("--provider", choices=["tencent", "baidu", "facepp", "ali"],
                        required=True, help="API提供商")
    parser.add_argument("--img1", required=True, help="第一张图片路径")
    parser.add_argument("--img2", required=True, help="第二张图片路径")
    parser.add_argument("--key", required=True, help="API Key / SecretId")
    parser.add_argument("--secret", required=True, help="API Secret")
    parser.add_argument("--benchmark", action="store_true", help="性能基准测试")
    args = parser.parse_args()

    providers = {
        "tencent": TencentFaceAPI,
        "baidu": BaiduFaceAPI,
        "facepp": FacePlusPlusAPI,
    }

    api = providers[args.provider](args.key, args.secret)
    b64_1 = image_to_base64(args.img1)
    b64_2 = image_to_base64(args.img2)

    print(f"[{args.provider}] 1:1 人脸比对...")
    result = api.compare_faces(b64_1, b64_2)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.benchmark:
        print(f"[{args.provider}] 性能基准测试 ({10}次)...")
        benchmark(api, args.img1, args.img2)
