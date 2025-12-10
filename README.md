# Workflow-CI (MLflow Project)

- Jalankan ulang training otomatis dengan `mlflow run Workflow-CI -P test_size=0.2`.
- Workflow GitHub Actions berada di `.workflow/ci-mlflow.yml` dan memanggil perintah di atas pada setiap push/dispatch.
- Data preprocessing disalin dari eksperimen ke `winequality_preprocessing`. Update folder ini bila ada preprocessing baru.
- Docker Hub: lengkapi `dockerhub_link.txt` jika image sudah dipublikasikan (opsional untuk level advanced).
