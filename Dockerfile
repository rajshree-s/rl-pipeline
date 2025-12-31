FROM python:3.13-slim

COPY ./dist/*.whl ./rl_pipeline-0.1.0-py3-none-any.whl
ENV hf_token=$hf_token

RUN pip install ./rl_pipeline-0.1.0-py3-none-any.whl

CMD ["python", "-m","rl_pipeline.main"]
