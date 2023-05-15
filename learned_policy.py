def policy_ldips(state):
    """Ground truth policy."""
    x_diff = state.get("x_diff")
    v_diff = state.get("v_diff")
    v_self = state.get("v_self")
    v_front = state.get("v_front")

    slow_to_fast = (((x_diff) ** 2 - (v_diff) ** 2) > 899.0263671875)
    fast_to_fast = ((v_front - (v_front) ** 2) > 930.348876953125)
    slow_to_slow = ((v_diff - (x_diff) ** 2) > -899.3515625)
    fast_to_slow = ((x_diff - (x_diff) ** 2) > -870.8091430664062)


    pre = state.get("start")
    if pre == "SLOWER":
        if slow_to_fast:
            post = "FASTER"
        elif slow_to_slow:
            post = "SLOWER"
        else:
            post = "SLOWER"
    elif pre == "FASTER":
        if fast_to_slow:
            post = "SLOWER"
        elif fast_to_fast:
            post = "FASTER"
        else:
            post = "FASTER"
    return post