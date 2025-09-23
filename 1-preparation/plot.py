import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any


def load_cleaned(clean_dir: str):
    windows_path = os.path.join(clean_dir, 'windows.npy')
    info_path = os.path.join(clean_dir, 'windows_info.pickle')
    events_path = os.path.join(clean_dir, 'event_windows.json')
    config_path = os.path.join(clean_dir, 'config.json')

    if not os.path.exists(windows_path):
        raise FileNotFoundError(f"Missing {windows_path}")

    windows = np.load(windows_path)
    with open(info_path, 'rb') as f:
        windows_info: List[Dict[str, Any]] = pickle.load(f)
    with open(config_path, 'r') as f:
        config = json.load(f)
    event_ids: List[int] = []
    if os.path.exists(events_path):
        with open(events_path, 'r') as f:
            event_ids = json.load(f).get('event_window_ids', [])

    return windows, windows_info, event_ids, config


def load_testing(test_dir: str):
    windows_path = os.path.join(test_dir, 'windows.npy')
    info_path = os.path.join(test_dir, 'windows_info.pickle')
    events_path = os.path.join(test_dir, 'event_windows.json')
    config_path = os.path.join(test_dir, 'config.json')

    if not os.path.exists(windows_path):
        raise FileNotFoundError(f"Missing {windows_path}")

    windows = np.load(windows_path)
    with open(info_path, 'rb') as f:
        windows_info: List[Dict[str, Any]] = pickle.load(f)
    with open(config_path, 'r') as f:
        config = json.load(f)
    event_ids: List[int] = []
    if os.path.exists(events_path):
        with open(events_path, 'r') as f:
            event_ids = json.load(f).get('event_window_ids', [])

    return windows, windows_info, event_ids, config


def plot_window_pair_raw_vs_clean(raw_window: np.ndarray,
                                  cleaned_window: np.ndarray,
                                  fs: float,
                                  window_id: int,
                                  is_event_raw: bool,
                                  title_prefix: str = "") -> None:
    t_raw = np.arange(len(raw_window)) / fs
    t_clean = np.arange(len(cleaned_window)) / fs

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5), sharex=False)
    fig.suptitle(f"{title_prefix}Window {window_id} | RAW {'EVENT' if is_event_raw else 'NORMAL'}", fontsize=12)

    # RAW + STA/LTA indicator (blue=normal, red=event)
    axes[0].plot(t_raw, raw_window, lw=0.7, color='crimson' if is_event_raw else 'tab:blue')
    axes[0].set_title('RAW (no-filter) + STA/LTA')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # CLEANED (preprocessed windows)
    axes[1].plot(t_clean, cleaned_window, lw=0.8, color='tab:blue')
    axes[1].set_title('CLEANED (detrend+bandpass) window')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_combined_windows(raw_windows: np.ndarray,
                          cleaned_windows: np.ndarray,
                          ids: List[int],
                          fs: float,
                          raw_event_ids: List[int]) -> None:
    if not ids:
        print("No window IDs provided to combine.")
        return
    ids = sorted(ids)
    raw_event_set = set(raw_event_ids)

    cleaned_concat = []
    raw_concat = []
    boundaries = [0]
    event_mask_concat = []

    for wid in ids:
        rw = raw_windows[wid]
        cw = cleaned_windows[wid]
        length = min(len(rw), len(cw))
        rw = rw[:length]
        cw = cw[:length]
        raw_concat.append(rw)
        cleaned_concat.append(cw)
        boundaries.append(boundaries[-1] + length)
        event_mask_concat.append(np.full(length, wid in raw_event_set, dtype=bool))

    cleaned_concat = np.concatenate(cleaned_concat) if cleaned_concat else np.array([])
    raw_concat = np.concatenate(raw_concat) if raw_concat else np.array([])
    event_mask_concat = np.concatenate(event_mask_concat) if event_mask_concat else np.array([])

    if len(cleaned_concat) == 0 or len(raw_concat) == 0:
        print("Nothing to plot after concatenation.")
        return

    events_in_selection = sum(1 for wid in ids if wid in raw_event_set)
    total_events_overall = len(raw_event_set)
    print(f"Concatenated {len(ids)} windows | RAW events in selection: {events_in_selection} | Total RAW events overall: {total_events_overall}")

    t_raw = np.arange(len(raw_concat)) / fs
    t_clean = np.arange(len(cleaned_concat)) / fs

    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=False)
    fig.suptitle(
        f"Combined {len(ids)} windows (IDs: {', '.join(map(str, ids))})\n"
        f"RAW events in selection: {events_in_selection} | Total RAW events overall: {total_events_overall}",
        fontsize=12
    )

    # RAW concatenated with color by event mask (blue=normal, red=event)
    if event_mask_concat.any():
        axes[0].plot(t_raw[~event_mask_concat], raw_concat[~event_mask_concat], lw=0.6, color='tab:blue')
        axes[0].plot(t_raw[event_mask_concat], raw_concat[event_mask_concat], lw=0.6, color='crimson')
    else:
        axes[0].plot(t_raw, raw_concat, lw=0.6, color='tab:blue')
    axes[0].set_title('RAW (no-filter) concatenated (red=event)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    for b in boundaries:
        axes[0].axvline(x=b / fs, color='k', ls='--', lw=0.6, alpha=0.4)

    # CLEANED concatenated; use RAW event mask to highlight corresponding segments
    if event_mask_concat.any():
        axes[1].plot(t_clean[~event_mask_concat], cleaned_concat[~event_mask_concat], lw=0.6, color='tab:blue')
        axes[1].plot(t_clean[event_mask_concat], cleaned_concat[event_mask_concat], lw=0.6, color='crimson')
    else:
        axes[1].plot(t_clean, cleaned_concat, lw=0.6, color='tab:blue')
    axes[1].set_title('CLEANED (detrend+bandpass) concatenated (red=RAW event)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    for b in boundaries:
        axes[1].axvline(x=b / fs, color='k', ls='--', lw=0.6, alpha=0.4)

    plt.tight_layout()
    plt.show()


def parse_ids_input(s: str, max_id: int) -> List[int]:
    s = s.strip()
    if not s:
        return []
    lower = s.lower()
    if lower in ("all", "*"):
        return list(range(max_id + 1))
    ids: List[int] = []
    parts = s.split(',')
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if '-' in p:
            # handle open-ended ranges: "a-" or "-b" or "a-b"
            if p == '-':
                continue
            if p.startswith('-') and p[1:].isdigit():
                b = int(p[1:])
                b = min(max_id, b)
                ids.extend(range(0, b + 1))
                continue
            if p.endswith('-') and p[:-1].isdigit():
                a = int(p[:-1])
                a = max(0, a)
                ids.extend(range(a, max_id + 1))
                continue
            a_str, b_str = p.split('-', 1)
            a, b = int(a_str), int(b_str)
            a, b = max(0, a), min(max_id, b)
            if a <= b:
                ids.extend(list(range(a, b + 1)))
        else:
            ids.append(int(p))
    ids = sorted(set([i for i in ids if 0 <= i <= max_id]))
    return ids


def main():
    print(" RAW vs CLEANED Window Plotter (with RAW STA/LTA)")
    print("=" * 40)

    dataset_dir = os.path.join('E:\\Skripsi\\DEC', 'dataset')  # sesuaikan bila perlu
    cleaned_dir = os.path.join(dataset_dir, 'cleaned')
    testing_dir = os.path.join(dataset_dir, 'testing')

    # Load both sets
    cleaned_win, cleaned_info, cleaned_event_ids, cleaned_cfg = load_cleaned(cleaned_dir)
    raw_win, raw_info, raw_event_ids, raw_cfg = load_testing(testing_dir)

    fs = float(cleaned_cfg.get('sampling_rate_hz', raw_cfg.get('sampling_rate_hz', 100.0)))

    # Sanity: align sizes
    total = min(len(cleaned_win), len(raw_win))
    if len(cleaned_win) != len(raw_win):
        print(f"Warning: cleaned windows ({len(cleaned_win)}) != raw windows ({len(raw_win)}). Using first {total}.")
        cleaned_win = cleaned_win[:total]
        raw_win = raw_win[:total]
        cleaned_info = cleaned_info[:total]
        raw_info = raw_info[:total]

    print(f"Loaded windows: {total}")
    print(f"Sampling rate: {fs} Hz")
    print(f"RAW event windows: {len(raw_event_ids)}")

    print("\nInput window IDs to display (examples):")
    print("  - Single: 42")
    print("  - Multiple: 1,5,10")
    print("  - Range: 100-110")
    print("  - Open range: -100 (start..100), 500- (500..end)")
    print("  - All: all or *")
    print("Press Enter to show 3 random windows.")
    user_in = input("Window IDs: ").strip()

    if user_in:
        ids = parse_ids_input(user_in, max_id=total - 1)
        if not ids:
            print("No valid IDs parsed. Showing 3 random windows.")
            ids = list(np.random.choice(total, size=min(3, total), replace=False))
    else:
        ids = list(np.random.choice(total, size=min(3, total), replace=False))

    combine_choice = input("Combine selected windows into a single plot? [y/N]: ").strip().lower()

    if combine_choice == 'y':
        plot_combined_windows(raw_win, cleaned_win, ids, fs, raw_event_ids)
    else:
        max_show = 20
        if len(ids) > max_show:
            print(f"Selected {len(ids)} windows. Showing first {max_show} to avoid overcrowding.")
            ids = ids[:max_show]
        raw_event_set = set(raw_event_ids)
        for wid in ids:
            is_event_raw = wid in raw_event_set
            plot_window_pair_raw_vs_clean(raw_win[wid], cleaned_win[wid], fs, wid, is_event_raw, title_prefix="RAW+STA/LTA vs CLEANED | ")

    print("\nDone.")


if __name__ == '__main__':
    main()
