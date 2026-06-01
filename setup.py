from setuptools import find_packages, setup

setup(
    name="sinn-sdam",
    version="2.0.0",
    description="S-DAM: Spelke-Seeded Dense Associative Memory",
    packages=find_packages(exclude=("tests", "experiments", "notebooks")),
    python_requires=">=3.9",
    install_requires=[
        "torch==2.0.1",
        "torchvision==0.15.2",
        "timm==0.9.7",
        "numpy==1.24.4",
        "pyyaml==6.0.1",
        "matplotlib==3.7.2",
        "scikit-learn==1.3.0",
        "scipy==1.11.2",
        "tqdm==4.65.0",
        "einops==0.6.1",
        "Pillow==9.5.0",
    ],
    extras_require={"dev": ["pytest==7.4.0"]},
)
