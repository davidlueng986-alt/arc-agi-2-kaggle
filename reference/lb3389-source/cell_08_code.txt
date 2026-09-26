import os
import json
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

if rerun_mode:
    data = ArcDataset.from_file("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")
else:
    data = ArcDataset.from_file("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
    data = data.load_replies("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")

decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)

decoder.load_decoded_results("/kaggle/inference_outputs")

submission = data.get_submission(decoder.run_selection_algo())

with open("submission.json", "w") as f:
    json.dump(submission, f)

if not rerun_mode:
    decoder.benchmark_selection_algos()
    with open("submission.json", "r") as f:
        reload_submission = json.load(f)
    print("*** Reload score:", data.validate_submission(reload_submission))