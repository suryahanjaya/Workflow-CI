# Workflow-CI
## Dicoding SMSML — Kriteria 3: CI Retraining + Docker

Repository ini berisi workflow CI untuk retraining model dan build Docker image secara otomatis.

---

## 📁 Struktur

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── retraining.yml    ← CI workflow (trigger push + manual)
└── MLProject/
    ├── MLproject             ← MLflow Project definition
    ├── conda.yaml            ← Conda environment (Python 3.12.7)
    ├── modelling.py          ← Training script (manual MLflow logging)
    ├── adult_income_preprocessed.csv
    └── Tautan_DockerHub.txt  ← Link Docker Hub image
```

---

## ⚙️ GitHub Secrets yang Diperlukan

| Secret | Value |
|--------|-------|
| `DAGSHUB_TOKEN` | DagsHub Personal Access Token |
| `DOCKERHUB_USERNAME` | `suryahanjaya` |
| `DOCKERHUB_TOKEN` | Docker Hub Access Token |

---

## 🔗 Links

- **MLflow (DagsHub)**: https://dagshub.com/suryahanjaya/adult-income-classifier.mlflow
- **Docker Hub**: https://hub.docker.com/r/suryahanjaya/adult-income-classifier
