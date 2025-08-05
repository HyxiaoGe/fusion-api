#!/usr/bin/env python3
"""
Railway启动脚本
解决环境变量PORT的问题
"""
import os
import subprocess
import sys

def main():
    # 获取PORT环境变量，默认为8000
    port = os.environ.get("PORT", "8000")
    
    # 构建uvicorn命令
    cmd = [
        "uvicorn",
        "main:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--workers", "2"
    ]
    
    print(f"启动命令: {' '.join(cmd)}")
    print(f"使用端口: {port}")
    
    # 执行命令
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()