"""Render one CARLA clip in the CWM format.

    python -m rtcwm.data.render --port 2000 --map Town03_Opt --seconds 38.9 --seed 20002 --out OUT

A pedestrian protagonist walks out along the paved middle of a sidewalk, turns round, walks back past
its start and out again (always walking forwards). open-code-world-model's FollowCamera follows it;
RGB, depth and instance-segmentation cameras share one pose. See rtcwm.data.clip for the outputs.
Exit code 3 = rendered but rejected by QC (run_clip.sh then retries with a new seed).
"""
