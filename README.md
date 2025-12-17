# Workflow-CI (MLflow Project)

- Proyek MLflow ada di folder `MLProject` (spec `MLProject`, environment `conda.yaml`, script `modelling.py`, dan dataset `namadataset_preprocessing/`).
- Jalankan lokal: `cd MLProject && conda env create -f conda.yaml && conda run -n mlflow_project_env python modelling.py`.
- Workflow GitHub Actions berada di `.github/workflows/ci-mlflow.yml`; trigger push/PR/main dan `workflow_dispatch`, menjalankan training lalu mengunggah artefak MLflow.
- Docker Hub: lengkapi `dockerhub_link.txt` jika image sudah dipublikasikan (opsional untuk level advanced).
