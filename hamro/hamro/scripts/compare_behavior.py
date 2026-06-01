def compare_behavior(stored_profile, current_input, tolerance=0.4):
    """
    Compare stored user profile with current input.
    Returns True if behavior is within tolerance, False otherwise.
    """
    score = 0
    checks = 0

    try:
        stored_speed = float(stored_profile['avg_typing_speed'])
        current_speed = float(current_input['typing_speed'])

        if stored_speed > 0:
            diff = abs(current_speed - stored_speed) / stored_speed
            if diff <= tolerance:
                score += 1
        checks += 1
    except Exception as e:
        print(f"Typing speed error: {e}")
        checks += 1  # Count it even if it fails to compare

    try:
        if str(current_input['location']) == str(stored_profile['location']):
            score += 1
        checks += 1
    except Exception as e:
        print(f"Location compare error: {e}")
        checks += 1

    try:
        stored_hour = int(stored_profile['login_hour'])
        current_hour = int(current_input['login_hour'])

        if abs(current_hour - stored_hour) <= 2:
            score += 1
        checks += 1
    except Exception as e:
        print(f"Hour compare error: {e}")
        checks += 1

    if checks == 0:
        return False
    return (score / checks) >= 0.66
