# What publish_product.py needs on the GPU machine. The script itself is copied there for each cycle, so code changes don't need a rebuild.
FROM python:3.12-slim
RUN pip install -q --no-cache-dir "numpy==2.*" "eccodes==2.*" eccodeslib
