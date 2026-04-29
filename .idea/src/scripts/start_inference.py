#!/usr/bin/env python
"""
门禁系统推理服务启动脚本

用法:
    python start_inference.py                    # 默认启动
    python start_inference.py --reload          # 开发模式（自动重载）
    python start_inference.py --port 9000      # 指定端口
    python start_inference.py --device cpu      # 强制 CPU 推理
"""

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import uvicorn

from inference.config import SERVER_HOST, SERVER_PORT, LOG_LEVEL

DEFAULT_PORT = int(os.getenv("SERVER_PORT", "8000"))
DEFAULT_HOST = os.getenv("SERVER_HOST", "0.0.0.0")


def main():
    parser = argparse.ArgumentParser(description="门禁系统推理服务启动脚本")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST,
                       help=f"监听地址（默认: {DEFAULT_HOST}）")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                       help=f"监听端口（默认: {DEFAULT_PORT}）")
    parser.add_argument("--reload", action="store_true",
                       help="启用热重载（开发模式）")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"],
                       default="auto", help="计算设备")
    parser.add_argument("--log-level", type=str,
                       choices=["debug", "info", "warning", "error"],
                       default="info", help="日志级别")
    parser.add_argument("--workers", type=int, default=1,
                       help="工作进程数（生产环境可增大）")
    args = parser.parse_args()

    if args.device != "auto":
        os.environ["DEVICE"] = args.device

    print("=" * 60)
    print("  门禁系统推理服务")
    print("=" * 60)
    print(f"  监听地址: {args.host}:{args.port}")
    print(f"  计算设备: {os.environ.get('DEVICE', 'auto')}")
    print(f"  日志级别: {args.log_level}")
    print(f"  热重载:   {'开启' if args.reload else '关闭'}")
    print("=" * 60)

    uvicorn.run(
        "inference.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        workers=1 if args.reload else args.workers,
    )


if __name__ == "__main__":
    main()
