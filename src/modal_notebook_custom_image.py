import modal

image = (
    modal.Image.debian_slim()
    .uv_pip_install(
        "cuml-cu13==26.8.0",
        "cupy-cuda13x[ctk]==14.2.0",
        "datasets==5.0.1",
        "fsspec==2026.6.0",
        "kernels==0.16.1",
        "nbformat==5.10.4",
        "nnsight==0.7.0",
        "numpy==2.4.6",
        "packaging==26.3",
        "pandas==3.0.3",
        "plotly==7.0.0",
        "scikit-learn==1.9.0",
        "torch==2.13.0",
        "tqdm==4.70.0",
        "transformers==5.15.1",
        extra_index_url="https://pypi.nvidia.com/simple",
    )
    .env(
        {
            "HF_TOKEN": "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",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)


app = modal.App("notebook-image")


@app.function(image=image)
def notebook_image():
    pass
