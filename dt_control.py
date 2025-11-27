# dt_control.py
# ------------------------------------------------------------
# Digital Twin control logic for TurtleBot3.
#
# Provide functions that:
#   - Take one row from the dataset (pandas Series)
#   - Take a predicted ML state (0 / 1 / 2)
#   - Print detailed control decisions
#   - Return (new_linear_velocity, new_angular_velocity)
#
# States:
#   0 = normal (no extra control)
#   1 = stuck / high friction (actual v << commanded v)
#   2 = sliding downhill (actual v >> commanded v)
# ------------------------------------------------------------

import math


# ======== STATE 0: NORMAL ========
def handle_state_0_normal(row):
    lv = float(row["linear_velocity"])
    ang = float(row["angular_velocity"])

    print("\n[STATE 0 - NORMAL OPERATION]")
    print(f"  linear v  = {lv:.3f} m/s")
    print(f"  angular v = {ang:.3f} rad/s")
    print("  -> No corrective action required.")

    return lv, ang


# ======== STATE 1: STUCK / HIGH FRICTION ========
def handle_state_1_stuck(row):
    lv = float(row["linear_velocity"])
    av = float(row["actual_velocity"])
    ang = float(row["angular_velocity"])

    eps = 1e-3
    speed_ratio = av / (abs(lv) + eps)

    # Apply strong deceleration (25%), but keep a minimum speed of 0.03 m/s for “crawling”
    base = max(abs(lv) * 0.25, 0.03)
    new_lv = math.copysign(base, lv if lv != 0 else 1)

    # Reduce angular velocity by 50% to minimize wheel-speed difference and avoid spinning in place
    new_ang = ang * 0.5

    print("\n[STATE 1 - STUCK / HIGH FRICTION DETECTED]")
    print(f"  commanded linear v : {lv:.3f} m/s")
    print(f"  actual linear v    : {av:.3f} m/s")
    print(f"  speed ratio        : {speed_ratio:.2f} (actual / commanded)")
    print("  -> Robot seems stuck. Applying crawl mode and stabilizing heading.")
    print(f"  new linear v       : {new_lv:.3f} m/s  (reduced to 25%, min 0.03)")
    print(f"  new angular v      : {new_ang:.3f} rad/s  (reduced by 50%)")

    return new_lv, new_ang


# ======== STATE 2: SLIDING DOWNHILL ========
def handle_state_2_sliding(row):
    lv = float(row["linear_velocity"])
    av = float(row["actual_velocity"])
    ang = float(row["angular_velocity"])

    eps = 1e-3
    speed_ratio = av / (abs(lv) + eps)

    # Sliding downhill: immediately set linear velocity to 0 (brake) and angular velocity to 0 (reduce wheel mismatch)
    new_lv = 0.0
    new_ang = 0.0

    print("\n[STATE 2 - SLIDING ON STEEP DESCENT]")
    print(f"  commanded linear v : {lv:.3f} m/s")
    print(f"  actual linear v    : {av:.3f} m/s")
    print(f"  speed ratio        : {speed_ratio:.2f}")
    print("  -> Sliding detected. Applying emergency braking and straightening wheels.")
    print(f"  new linear v       : {new_lv:.3f} m/s  (STOP)")
    print(f"  new angular v      : {new_ang:.3f} rad/s  (straight wheels)")

    return new_lv, new_ang


# ======== UNIFIED ENTRYPOINT ========
def apply_control_for_state(row, pred_state: int):
    """
    Unified Digital Twin control entrypoint.
    Input:
        row        = pandas Series (one row from CSV)
        pred_state = predicted ML label (0 / 1 / 2)
    Output:
        (new_lv, new_ang)
    """

    if pred_state == 0:
        return handle_state_0_normal(row)
    elif pred_state == 1:
        return handle_state_1_stuck(row)
    elif pred_state == 2:
        return handle_state_2_sliding(row)
    else:
        print(f"\n[UNKNOWN STATE {pred_state}] -> Keeping original commands.")
        lv = float(row["linear_velocity"])
        ang = float(row["angular_velocity"])
        return lv, ang


# ======== OPTIONAL: FUNCTION FOR DIRECT CSV ACCESS ========
def control_from_csv(csv_path: str, row_index: int):
    """
    Directly read one row from CSV and apply corresponding control behavior.
    Useful for quick testing.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"Row index {row_index} out of range (0 ~ {len(df)-1})")

    row = df.iloc[row_index]
    state = int(row["status"])

    print("\n-----------------------------")
    print(f"Row #{row_index}   predicted_state = {state}")
    print("-----------------------------")

    return apply_control_for_state(row, state)
