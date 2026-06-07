"""S-DAM feasibility probe (go/no-go gate before the full pipeline).

Pure-math layers (variance, decodability) are torch-free and unit-tested with
known-answer synthetic data. Encoder + orchestration require torch/GPU and run
on Colab.
"""
