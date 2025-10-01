import numpy as np


def make_tick_labels(ticks):
    tick_labels = []
    for tick in ticks:
        tick_labels.append(make_tick_label(tick))
    return tick_labels


def make_tick_label(tick):
    res = make_tick_label_dbg(tick)[0]
    res = res.replace(".0$", "$")
    return res 


def make_tick_label_dbg(tick):
    res = make_tick_label_dbg2(tick)
    r0, r1 = res
    r0 = r0.replace(".0$", "$")
    r0 = r0.replace(".0 ", " ")
    return r0, r1


def make_tick_label_dbg2(tick):
    pfx = ""
    if tick == 0:
        return "$0$", "a"
    if tick < 0:
        pfx = "-"
        tick = -tick

    log = np.log10(tick)
    if log == 0:
        return f"${pfx}1$", "b"

    if tick == 10:
        return f"${pfx}10$", "c"

    exponent = int(np.floor(np.log10(tick)))

    base = tick / 10**exponent
    base = np.round(base, 2)
    characteristic = int(base)
    mantissa = base - characteristic
    # print(log, base, characteristic, mantissa, exponent)

    if characteristic == 1 and characteristic == base:
        return f"${pfx}10^{{{exponent}}}$", "d"

    if exponent >= 3 or exponent <= -3:
        if mantissa == 0:
            return f"${pfx}{characteristic} \\times 10^{{{exponent}}}$", "e"
        return f"${pfx}{np.round(base, 1)} \\times 10^{{{exponent}}}$", "f"

    if exponent == 0:
        return f"${pfx}{base}$", "g"

    if exponent == 1:
        return f"${pfx}{base*10**exponent:.1f}$", "h"

    if exponent == 2:
        return f"${pfx}{base*10**exponent:.0f}$", "i"

    res = f"{np.round(tick, 2)}"
    res = res.split("00")[0]
    res = res[:5]
    return f"${pfx}{res}$", "j"


def generate_symlog_ticks(min_val, max_val, linthresh, linscale=1.0):
    """
    Generate reasonable symlog ticks:
    - Always include min_val, max_val, ±linthresh, 0 (if in range)
    - Include log ticks outside linthresh
    - Add clean, round-number ticks in the linear regime (if needed)

    Returns:
        List[float]: Sorted unique tick values
    """
    if min_val == max_val:
        return [min_val]

    ticks = set()

    def log_ticks(lo, hi):
        """Return log-decade ticks between lo and hi (inclusive)."""
        if lo <= 0 or hi <= 0:
            return []
        decades = np.arange(np.ceil(np.log10(lo)), np.floor(np.log10(hi)) + 1)
        return list(10.0 ** decades)

    def round_linear_ticks(lo, hi, step):
        """Return round-number linear ticks between lo and hi at given step."""
        start = np.ceil(lo / step) * step
        end = np.floor(hi / step) * step
        return list(np.arange(start, end + step/2, step))

    # Track whether each region is present
    lin_lo = -linthresh * linscale
    lin_hi = linthresh * linscale
    has_linear = (min_val < linthresh * linscale) and (max_val > -linthresh * linscale)
    has_neg_log = min_val < -linthresh
    has_pos_log = max_val > linthresh
    has_neg_lin = (min_val < 0) and (max_val > lin_lo)
    has_pos_lin = (max_val > 0) and (min_val < lin_hi)

    # --- Negative log ticks ---
    if has_neg_log:
        ticks.update(-np.array(log_ticks(linthresh, abs(min_val))))

    # --- Positive log ticks ---
    if has_pos_log:
        ticks.update(log_ticks(linthresh, max_val))

    # --- Always include critical markers ---
    ticks.update([min_val, max_val])
    if -linthresh >= min_val:
        ticks.add(-linthresh)
    if linthresh <= max_val:
        ticks.add(linthresh)
    if min_val < 0 < max_val:
        ticks.add(0.0)

    n_ticks = len(ticks)

    # --- Linear region ticks ---
    lin_ticks = tick_plan(linthresh, min_val, max_val, n_ticks, 10)
    # if has_linear:
    #     linear_lo = max(min_val, -linthresh * linscale)
    #     linear_hi = min(max_val, linthresh * linscale)

    #     # Decide step size
    #     if n_ticks < 6:  # Min, max, 0, +linthres, -linthresh only
    #         if linthresh % 3 == 0:
    #             div = 3
    #         else:
    #             div = 2
    #         if has_neg_lin:
    #             ticks.add(linear_lo // div)
    #             ticks.add(-2 * linear_lo // div)
    #         if has_pos_lin:
    #             ticks.add(linear_hi // div)
    #             ticks.add(2 * linear_hi // div)
    ticks.update(lin_ticks)

    return sorted(ticks)


def tick_plan(linthresh, vmin, vmax, curr_n_ticks, max_n_ticks=8):
    linthresh = min(linthresh, max(abs(vmin), abs(vmax)))
    exp = int(np.floor(np.log10(linthresh)))
    base = 10 ** exp
    if linthresh == base:
        exp -= 1
        base = 10 ** exp
    big_val = int(linthresh / base)

    def candidate_ticks(inc):
        # Negative ticks
        prop_ticks = []
        if vmin < 0:
            for i in range(1, 10):
                maybe_tick = -inc * i
                if maybe_tick > vmin and maybe_tick > -linthresh:
                    prop_ticks.append(maybe_tick)
                else:
                    break
        # Positive ticks
        if vmax > 0:
            for i in range(1, 10):
                maybe_tick = inc * i
                if maybe_tick < vmax and maybe_tick < linthresh:
                    prop_ticks.append(maybe_tick)
                else:
                    break
        return prop_ticks

    for i in range(1, 5):
        if big_val % i == 0:
            inc = i * base
            ticks = candidate_ticks(inc)
            if len(ticks) + curr_n_ticks <= max_n_ticks:
                return ticks
    return []

if __name__ == "__main__":
    # Testing the function
    check_vals = [
        0, 
        *[10**i for i in range(-3,4)],
        *[2*10**i for i in range(-3,4)],
        *[1.2*10**i for i in range(-3,4)],
        *[1.64642*10**i for i in range(-3,4)],
        *[1.9*10**i for i in range(-3,4)],
        *[2.1*10**i for i in range(-3,4)],
        *[1.99*10**i for i in range(-3,4)],
        *[2.01*10**i for i in range(-3,4)],
    ]
    for val in check_vals:
        asdf = make_tick_label_dbg(val)
        print(f"{val:<23}: {asdf[1]}, {asdf[0]}")
    for val in check_vals:
        asdf = make_tick_label_dbg(-val)
        print(f"{-val:<23}: {asdf[1]}, {asdf[0]}")
