from pathlib import Path
import json
import os

root = Path(os.environ.get("ML_PROJECT_ROOT", ".")).resolve()
run = root / "outputs/course_study/cf_ml_course_20261112_122114"
receipt = json.loads((run / "submission_receipt.json").read_text(encoding="utf-8"))
record = {
    "status": "awaiting_user_upload",
    "selected_algorithm": receipt["selected_algorithm"],
    "submission_csv": receipt["path"],
    "submission_sha256": receipt["sha256"],
}
(run / "kaggle_submission.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print("Submission metadata initialized from the generated CSV receipt.")
