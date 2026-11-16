"""Reuse the locked public acquisition entry point with separate paths."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import fetch_task_context_confirmation as acquisition

acquisition.TARGET = ROOT/"data/second_residual_test_v1"
acquisition.LOCK = ROOT/"results/task_evidence/residual_confirmation_v1/method_lock.json"
acquisition.RESERVATION = ROOT/"data/second_residual_reservation_v1.json"

if __name__ == "__main__":
    acquisition.main()
