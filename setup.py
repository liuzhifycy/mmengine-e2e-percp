"""
MMEngine-Lite: 轻量级 MMEngine 训练框架模板
============================================

一个独立、可迁移的深度学习训练框架，基于 MMEngine 构建，
支持目标检测模型的训练、推理、评估和导出。

主要特性:
- 支持单GPU/多GPU分布式训练
- 完整的训练/测试/可视化/导出流程
- 基于 mmdet 的 RetinaNet + ResNet 模型
- COCO 数据集支持
"""

from setuptools import find_packages, setup


def get_version():
    return "0.1.0"


def get_requirements(filename="requirements.txt"):
    with open(filename, "r") as f:
        lines = f.readlines()
    requirements = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements


setup(
    name="mmlite",
    version=get_version(),
    description="A lightweight MMEngine-based training framework",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-repo/mmengine-lite",
    packages=find_packages(exclude=["configs", "tools", "scripts", "tests"]),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
