Per-camera lane directions for the wrong-lane check.

    python tools/calibrate_lanes.py <a few minutes of normal traffic from the camera>.mp4

writes `<video name>.json` + `<video name>_preview.jpg` here (check the arrows once).
The wrong-lane module uses, in this order: the file in WRONG_LANE_ZONES, `lane_zones/<video name>.json`,
`lane_zones/default.json`; without any of them it learns the directions from the traffic in each video.
Re-run the calibration if the camera is moved or re-aimed.
