FROM python:3.13-bookworm

COPY dist/rl_pipeline-0.1.0-py3-none-any.whl /tmp/rl_pipeline-0.1.0-py3-none-any.whl

RUN pip install --no-cache-dir /tmp/rl_pipeline-0.1.0-py3-none-any.whl
RUN cd src/rl_pipeline

CMD ["python", "main.py"]
