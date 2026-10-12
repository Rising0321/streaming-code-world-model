"""Dataset QC from saved outputs (no re-rendering).

    python -m rtcwm.data.qc clipping CLIP_DIR ...   interpenetration (also run by the renderer)
    python -m rtcwm.data.qc audit    CLIP_DIR ...   camera, motion, exposure, coarse-vs-fine audit
    python -m rtcwm.data.qc coarse   CLIP_DIR ...   coarse proxy swallowing the frame / losing the protagonist
    python -m rtcwm.data.qc lists                   data_lists/bad_windows.json from the reports

Reports go to paths.QC/<check>/<clip>.json; every report has per-window verdicts (window w = frames
90w .. 90w+123) so training can drop single windows.
"""
