#!/usr/bin/env python
"""
系统环境检测脚本 - 门禁系统

供 Unity 项目在启动时或设置页面中检测运行环境是否满足要求。
支持 JSON / text 两种输出格式，方便程序化解析。

用法:
    python check_requirements.py                    # 完整检测 + JSON 输出
    python check_requirements.py --format text     # 人类可读文本输出
    python check_requirements.py --check python    # 仅检测 Python 环境
    python check_requirements.py --check torch      # 仅检测 PyTorch 环境
    python check_requirements.py --check cuda       # 仅检测 CUDA 环境
    python check_requirements.py --check models      # 仅检测模型文件
    python check_requirements.py --check gallery     # 仅检测 Gallery 路径
    python check_requirements.py --check network     # 仅检测后端连通性
    python check_requirements.py --check all         # 等同于无 --check 参数
    python check_requirements.py --silent            # 仅返回 exit code，不输出
    python check_requirements.py --backend-url http://localhost:8000  # 指定后端地址
"""

import os
import sys
import json
import socket
import argparse
import urllib.request
import urllib.error
import importlib
import importlib.util
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GALLERY_FEATURES_DIR = PROJECT_ROOT / "data" / "gallery" / "features"
GALLERY_IMAGES_DIR = PROJECT_ROOT / "data" / "gallery" / "images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MAPPING_FILE = PROJECT_ROOT / "data" / "face_casia_mapping.json"
BACKEND_DEFAULT_URL = "http://localhost:8000/health"


@dataclass
class CheckResult:
    status: str          # "pass" | "warn" | "fail"
    detail: str          # 详细信息
    info: Dict[str, Any] = field(default_factory=dict)  # 附加数据

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RequirementsChecker:
    """门禁系统环境检测器"""

    def __init__(self, backend_url: str = BACKEND_DEFAULT_URL):
        self.backend_url = backend_url
        self.results: Dict[str, CheckResult] = {}
        self._overall: Optional[str] = None

    # ── 基础工具 ────────────────────────────────────────────────────────────

    def _status(self, cond: bool, warn_cond: bool = False) -> str:
        if cond:
            return "pass"
        elif warn_cond:
            return "warn"
        return "fail"

    def _find_file(self, root: Path, pattern: str) -> List[Path]:
        if not root.exists():
            return []
        return list(root.rglob(pattern))

    def _dir_exists_and_writable(self, path: Path) -> tuple[bool, str]:
        if not path.exists():
            return False, f"目录不存在: {path}"
        if not os.access(path, os.W_OK):
            return False, f"目录不可写: {path}"
        return True, f"OK: {path}"

    # ── 检测项 ──────────────────────────────────────────────────────────────

    def check_python_version(self) -> CheckResult:
        """Python 版本检测（3.8 - 3.11）"""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        min_major, min_minor = 3, 8
        max_major, max_minor = 3, 11

        if (version.major, version.minor) < (min_major, min_minor):
            return CheckResult("fail", f"Python {version_str} 不满足最低要求 (>= 3.8)",
                               {"version": version_str, "required": f">= {min_major}.{min_minor}"})
        if (version.major, version.minor) > (max_major, max_minor):
            return CheckResult("warn", f"Python {version_str} 未经测试，推荐 3.8-3.11",
                               {"version": version_str, "recommended": "3.8-3.11"})

        return CheckResult("pass", f"Python {version_str}", {"version": version_str})

    def check_package(self, name: str, min_version: Optional[str] = None,
                      alt_names: Optional[List[str]] = None) -> CheckResult:
        """检测 Python 包是否安装及版本"""
        names = [name] + (alt_names or [])

        installed_name = None
        version_str = None
        installed_version = None

        for pkg_name in names:
            spec = importlib.util.find_spec(pkg_name)
            if spec is not None:
                try:
                    mod = importlib.import_module(pkg_name)
                    ver = getattr(mod, "__version__", None) or "unknown"
                    if ver == "unknown":
                        import pkg_resources
                        ver = pkg_resources.get_distribution(pkg_name).version
                except Exception:
                    ver = "unknown"

                installed_name = pkg_name
                version_str = ver
                try:
                    parts = ver.replace("+", ".").split(".")[:3]
                    installed_version = tuple(int(x) for x in parts if x.isdigit())
                except Exception:
                    installed_version = None
                break

        if installed_name is None:
            detail = f"未安装: {name}（尝试了: {', '.join(names)}）"
            return CheckResult("fail", detail,
                               {"package": name, "alternatives": names, "installed": False})

        info = {"package": installed_name, "version": version_str, "installed": True}

        if min_version and installed_version:
            try:
                min_parts = tuple(int(x) for x in min_version.split(".")[:3] if x.isdigit())
                if installed_version < min_parts:
                    return CheckResult(
                        "fail",
                        f"{installed_name} {version_str} 不满足最低要求 (>={min_version})",
                        info
                    )
            except Exception:
                pass

        return CheckResult("pass", f"{installed_name} {version_str}", info)

    def check_torch(self) -> CheckResult:
        """PyTorch 环境检测"""
        try:
            import torch
            version = torch.__version__
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else None

            info = {"version": version, "cuda_available": cuda_available}
            if cuda_version:
                info["cuda_version"] = cuda_version

            # 检查是否为 GPU 版本
            if "cpu" in version and cuda_available is False:
                # torch 是 CPU 版本，但可能另有 GPU 版本安装
                return CheckResult(
                    "warn",
                    f"PyTorch {version} (CPU only)，如需 GPU 推理请安装 GPU 版本: "
                    "pip install torch --index-url https://download.pytorch.org/whl/cu118",
                    info
                )

            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                info["gpu_count"] = gpu_count
                info["gpu_names"] = gpu_names
                mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info["gpu_memory_gb"] = round(mem, 1)

                if mem < 5.5:
                    return CheckResult(
                        "warn",
                        f"PyTorch {version}, CUDA {cuda_version}, "
                        f"GPU {gpu_names[0]} ({mem:.1f}GB) 显存偏低，推荐 >= 6GB",
                        info
                    )
                return CheckResult(
                    "pass",
                    f"PyTorch {version}, CUDA {cuda_version}, "
                    f"{gpu_count}x {gpu_names[0]} ({mem:.1f}GB)",
                    info
                )
            else:
                return CheckResult(
                    "pass",
                    f"PyTorch {version} (CPU 模式，可正常运行但推理较慢)",
                    info
                )
        except ImportError:
            return CheckResult(
                "fail",
                "PyTorch 未安装。请运行: pip install torch torchvision",
                {"package": "torch", "installed": False}
            )

    def check_cuda(self) -> CheckResult:
        """CUDA 环境检测"""
        try:
            import torch
            if not torch.cuda.is_available():
                return CheckResult(
                    "pass",
                    "CUDA 不可用（CPU 推理模式）",
                    {"cuda_available": False, "mode": "cpu"}
                )

            cuda_ver = torch.version.cuda
            driver_ver = torch.cuda.get_device_capability()

            # 尝试从 nvidia-smi 获取驱动版本
            driver_version = None
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    driver_version = result.stdout.strip().split()[0]
            except Exception:
                pass

            info = {
                "cuda_version": cuda_ver,
                "device_capability": f"{driver_ver[0]}.{driver_ver[1]}",
                "driver_version": driver_version
            }

            # 粗略版本兼容性检查
            if cuda_ver:
                major = int(cuda_ver.split(".")[0])
                if major < 11:
                    return CheckResult(
                        "warn",
                        f"CUDA {cuda_ver} 版本较旧，推荐使用 CUDA 11.x 或 12.x",
                        info
                    )

            return CheckResult(
                "pass",
                f"CUDA {cuda_ver}, Driver {driver_version or 'unknown'}, "
                f"Compute {driver_ver[0]}.{driver_ver[1]}",
                info
            )
        except ImportError:
            return CheckResult(
                "fail",
                "无法检测 CUDA（PyTorch 未安装）",
                {"error": "torch not installed"}
            )

    def check_model_files(self) -> CheckResult:
        """模型文件完整性检测

        搜索 outputs/ 目录下所有 best.pth 融合模型。
        """
        fusion_patterns = ["best.pth"]
        found_models: Dict[str, List[str]] = {}
        missing_models: List[str] = []

        # 在 outputs/ 下搜索所有 best.pth
        found = self._find_file(OUTPUTS_DIR, "best.pth")
        if found:
            # 按实验名称组织
            for p in found:
                rel = p.relative_to(OUTPUTS_DIR)
                key = str(rel.parent.name)  # e.g. "fusion_adaptive_full"
                if key not in found_models:
                    found_models[key] = []
                found_models[key].append(str(p))
        else:
            missing_models = ["best.pth (任意融合实验)"]

        info = {
            "outputs_dir": str(OUTPUTS_DIR),
            "found": found_models,
            "missing": missing_models,
        }

        if not found_models:
            return CheckResult(
                "fail",
                f"未找到任何融合模型 best.pth（搜索目录: {OUTPUTS_DIR}）。"
                "请先执行训练: python scripts/train_fusion.py",
                info
            )
        return CheckResult(
            "pass",
            f"找到 {len(found_models)} 个融合模型: {list(found_models.keys())}",
            info
        )

    def check_gallery_path(self) -> CheckResult:
        """Gallery 路径与文件检测"""
        issues: List[str] = []
        ok_paths: List[str] = []

        # 特征目录
        face_feat = GALLERY_FEATURES_DIR / "face_features.json"
        fp_feat = GALLERY_FEATURES_DIR / "fingerprint_features.json"

        for path in [GALLERY_FEATURES_DIR, GALLERY_IMAGES_DIR]:
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    ok_paths.append(f"[AUTO-CREATED] {path}")
                except Exception as e:
                    issues.append(f"无法创建目录 {path}: {e}")
            elif not os.access(path, os.W_OK):
                issues.append(f"目录不可写: {path}")
            else:
                ok_paths.append(f"OK: {path}")

        # 特征文件
        user_count = 0
        for feat_file in [face_feat, fp_feat]:
            if not feat_file.exists():
                issues.append(f"特征文件不存在: {feat_file}")
            else:
                try:
                    with open(feat_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "users" in data:
                        n = len(data["users"])
                        user_count = max(user_count, n)
                    ok_paths.append(f"OK: {feat_file}")
                except json.JSONDecodeError as e:
                    issues.append(f"特征文件 JSON 格式错误: {feat_file} ({e})")
                except Exception as e:
                    issues.append(f"读取特征文件失败: {feat_file} ({e})")

        info = {
            "gallery_features_dir": str(GALLERY_FEATURES_DIR),
            "gallery_images_dir": str(GALLERY_IMAGES_DIR),
            "face_features_file": str(face_feat),
            "fingerprint_features_file": str(fp_feat),
            "registered_users": user_count,
        }

        if issues and not ok_paths:
            detail = "; ".join(issues)
            return CheckResult("fail", detail, info)

        if issues:
            detail = f"已注册用户: {user_count}; 警告: {'; '.join(issues)}"
            if user_count == 0:
                return CheckResult("warn", f"Gallery 为空（{len(issues)} 个问题）", info)
            return CheckResult("warn", detail, info)

        if user_count == 0:
            return CheckResult("warn", "Gallery 目录正常但尚未注册用户", info)

        return CheckResult("pass", f"Gallery 正常，已注册 {user_count} 个用户", info)

    def check_backend_connectivity(self) -> CheckResult:
        """后端服务连通性检测"""
        # 先检查端口是否可达
        try:
            host = self.backend_url.replace("http://", "").replace("https://", "").split("/")[0]
            if ":" in host:
                host, port_str = host.rsplit(":", 1)
                port = int(port_str)
            else:
                port = 80

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            sock.close()

            if result != 0:
                return CheckResult(
                    "fail",
                    f"无法连接到 {host}:{port}，服务可能未启动",
                    {"host": host, "port": port}
                )
        except Exception as e:
            return CheckResult(
                "warn",
                f"端口检测失败: {e}",
                {"host": host, "port": port if 'port' in locals() else 80}
            )

        # HTTP 健康检查
        try:
            req = urllib.request.Request(self.backend_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                status_code = resp.status
                body = resp.read().decode("utf-8", errors="replace")[:200]
        except urllib.error.URLError as e:
            return CheckResult(
                "fail",
                f"HTTP 请求失败: {e.reason}",
                {"url": self.backend_url}
            )
        except Exception as e:
            return CheckResult(
                "warn",
                f"HTTP 请求异常: {e}",
                {"url": self.backend_url}
            )

        info = {"url": self.backend_url, "http_status": status_code, "response_preview": body}
        if status_code == 200:
            return CheckResult("pass", f"后端服务正常 (HTTP {status_code})", info)
        return CheckResult("warn", f"后端返回异常状态码 HTTP {status_code}", info)

    def check_all(self) -> Dict[str, CheckResult]:
        """执行全部检测"""
        checks = {
            "python_version": self.check_python_version,
            "torch": self.check_torch,
            "cuda": self.check_cuda,
            "model_files": self.check_model_files,
            "gallery": self.check_gallery_path,
            "backend_connectivity": self.check_backend_connectivity,
        }

        # 按顺序检测
        self.results = {}
        for name, fn in checks.items():
            try:
                self.results[name] = fn()
            except Exception as e:
                self.results[name] = CheckResult(
                    "fail", f"检测过程中发生异常: {e}", {"exception": str(e)}
                )

        return self.results

    @property
    def overall(self) -> str:
        """综合判断结果"""
        if self._overall is not None:
            return self._overall
        if not self.results:
            return "unknown"
        statuses = [r.status for r in self.results.values()]
        if any(s == "fail" for s in statuses):
            return "fail"
        if any(s == "warn" for s in statuses):
            return "warn"
        return "pass"

    def to_json(self) -> str:
        """输出 JSON 格式"""
        output = {
            "overall": self.overall,
            "checks": {name: r.to_dict() for name, r in self.results.items()},
            "backend_url": self.backend_url,
        }
        return json.dumps(output, ensure_ascii=False, indent=2)

    def to_text(self) -> str:
        """输出人类可读文本"""
        lines = [
            "=" * 60,
            f"  门禁系统 - 环境检测报告",
            "=" * 60,
            f"综合结果: [{self.overall.upper()}]",
            "-" * 60,
        ]

        for name, result in self.results.items():
            label = {
                "python_version": "Python 环境",
                "torch": "PyTorch 环境",
                "cuda": "CUDA 环境",
                "model_files": "模型文件",
                "gallery": "Gallery 路径",
                "backend_connectivity": "后端连通性",
            }.get(name, name)

            icon = {"pass": "[OK]", "warn": "[WARN]", "fail": "[FAIL]"}[result.status]
            lines.append(f"  {icon} {label}: {result.detail}")

        lines.append("-" * 60)
        lines.append(f"后端地址: {self.backend_url}")
        lines.append("=" * 60)
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="门禁系统环境检测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "text"],
        default="json",
        help="输出格式（默认: json）"
    )
    parser.add_argument(
        "--check", "-c",
        choices=["python", "torch", "cuda", "models", "gallery", "network", "all"],
        default="all",
        help="指定检测项（默认: all）"
    )
    parser.add_argument(
        "--backend-url", "-u",
        default=BACKEND_DEFAULT_URL,
        help=f"后端服务地址（默认: {BACKEND_DEFAULT_URL}）"
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="静默模式，仅返回 exit code（0=pass/warn, 1=fail）"
    )

    args = parser.parse_args()

    # 映射简写到内部名称
    check_map = {
        "python": "python_version",
        "torch": "torch",
        "cuda": "cuda",
        "models": "model_files",
        "gallery": "gallery",
        "network": "backend_connectivity",
        "all": None,  # 全部检测
    }
    check_target = check_map[args.check]

    checker = RequirementsChecker(backend_url=args.backend_url)

    # 执行检测
    if check_target:
        single_check = {
            "python_version": checker.check_python_version,
            "torch": checker.check_torch,
            "cuda": checker.check_cuda,
            "model_files": checker.check_model_files,
            "gallery": checker.check_gallery_path,
            "backend_connectivity": checker.check_backend_connectivity,
        }[check_target]
        checker.results = {check_target: single_check()}
    else:
        checker.check_all()

    # 确定退出码：fail -> 1, pass/warn -> 0
    overall = checker.overall
    exit_code = 0 if overall in ("pass", "warn") else 1

    if not args.silent:
        if args.format == "json":
            print(checker.to_json())
        else:
            print(checker.to_text())

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
