"""
voxam — Voice Biomarker Extraction for Clinical Research
====================================================================

A double-clickable application for extracting clinically-labelled
acoustic voice features from audio recordings.

HOW TO USE:
-----------
  1. Install Python from python.org (one time only, see install guide)
  2. Install the required libraries (one time only):
     Open a command prompt and run:
     pip install praat-parselmouth pandas numpy

  3. Double-click this file (voxam.pyw) to launch the app.

WHAT IT DOES:
-------------
Extracts acoustic features from voice recordings — pitch, voice
quality (jitter/shimmer/CPP), clarity (HNR), loudness, timing — and
labels each one with its relevant symptom domain. Users choose which
features to display and save via checkboxes.

Everything runs on your computer. Audio files never leave your machine.

SUPPORTED FORMATS:
------------------
.wav (best), .aiff, .flac
For .mp3 or .m4a, please convert to .wav first using a free online
converter like cloudconvert.com
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import math
import threading
from datetime import datetime

# Check that required libraries are installed
try:
    import parselmouth
    from parselmouth.praat import call
    import pandas as pd
except ImportError as e:
    # Show a friendly error if libraries aren't installed
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Missing libraries",
        f"voxam needs a one-time library installation.\n\n"
        f"Please open a command prompt and run:\n\n"
        f"pip install praat-parselmouth pandas numpy\n\n"
        f"Then double-click voxam again.\n\n"
        f"(Technical detail: {e})"
    )
    raise SystemExit


# ============================================================
# FEATURE DEFINITIONS
# ============================================================
# Each feature belongs to a domain group. The UI builds checkboxes
# from these groups automatically.

FEATURE_GROUPS = [
    ("Laryngeal Control", [
        ("F0 Mean (average pitch)",
         "Lower → fatigue, depression, sedation. Higher → anxiety, pain."),
        ("F0 SD (pitch variability)",
         "Reduced (monotone) → depression, fatigue, sedation."),
        ("F0 Range (pitch range)",
         "Compressed → flat affect. Wider → expressive speech."),
        ("Intensity SD (loudness variability)",
         "Reduced → flat delivery, fatigue, depression."),
    ]),
    ("Voice Quality", [
        ("HNR (voice clarity)",
         "Higher = clearer. Lower → breathy/hoarse. Normal > 20 dB."),
        ("Cepstral Peak Prominence (voice quality)",
         "Overall voice quality. Higher = clearer, more periodic voice. Robust in noisy recordings."),
        ("Shimmer (volume steadiness)",
         "Higher → unsteady volume. Normal < 3.81%. Elevated → breathlessness, fatigue."),
        ("Voice Breaks (voicing interruptions)",
         "More breaks → more interruptions in voicing. Breathlessness or vocal effort."),
    ]),
    ("Neuromuscular & Respiratory", [
        ("Jitter (pitch steadiness)",
         "Higher → unsteady voice. Normal < 1.04%. Elevated → fatigue, weakness."),
        ("Intensity Mean (average loudness)",
         "Reduced → fatigue, weakness, reduced respiratory capacity."),
    ]),
    ("Respiratory & Cognitive Timing", [
        ("Phonation Ratio (speaking-to-silence)",
         "Lower → more silence → breathlessness or cognitive slowing."),
        ("Speech Rate (voiced segments/sec)",
         "Lower → depression, fatigue, cognitive slowing. Dissociates from phonation ratio."),
        ("Mean Pause Duration",
         "Longer pauses → depression, cognitive impairment, respiratory effort."),
    ]),
    ("Vocal Tract & Articulation", [
        ("F1 (first formant)",
         "Changes may reflect oedema, dry mouth, or altered oral cavity."),
        ("F2 (second formant)",
         "Changes over time may reflect fluid shifts or airway changes."),
    ]),
    ("Metadata", [
        ("Recording Duration",
         "Total length of the recording."),
    ]),
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def safe(val):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def extract_voice_features(audio_path):
    """
    Extract all voice features from an audio file.
    Returns a dict keyed by feature name → (value, unit).
    """
    sound = parselmouth.Sound(audio_path)
    duration = sound.get_total_duration()

    # --- Pitch ---
    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    f0_mean = safe(call(pitch, "Get mean", 0, 0, "Hertz"))
    f0_sd = safe(call(pitch, "Get standard deviation", 0, 0, "Hertz"))
    f0_min = safe(call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic"))
    f0_max = safe(call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic"))
    f0_range = (f0_max - f0_min) if (f0_max is not None and f0_min is not None) else None

    # --- Jitter & shimmer ---
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
    jitter = safe(call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
    shimmer = safe(call([sound, point_process], "Get shimmer (local)",
                        0, 0, 0.0001, 0.02, 1.3, 1.6))

    # --- HNR ---
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = safe(call(harmonicity, "Get mean", 0, 0))

    # --- CPP (Cepstral Peak Prominence) ---
    # Computed via PowerCepstrogram, which is Praat's standard CPP pipeline.
    # Robust to connected speech and noisier recordings, unlike jitter/shimmer.
    try:
        cepstrogram = call(sound, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        cpp = safe(call(cepstrogram, "Get CPPS",
                        True, 0.01, 0.001,
                        60, 330, 0.05,
                        "Parabolic", 0.001, 0.0,
                        "Exponential decay", "Robust"))
    except Exception:
        cpp = None

    # --- Intensity ---
    intensity = call(sound, "To Intensity", 75, 0.0)
    int_mean = safe(call(intensity, "Get mean", 0, 0, "dB"))
    int_sd = safe(call(intensity, "Get standard deviation", 0, 0))

    # --- Voiced / unvoiced analysis: phonation ratio, breaks, speech rate, pause duration ---
    n_frames = call(pitch, "Get number of frames")
    time_step = safe(call(pitch, "Get time step"))
    if time_step is None or time_step <= 0:
        time_step = 0.01  # Praat default

    voiced_frames = 0
    breaks = 0
    voiced_segments = 0       # count of contiguous voiced runs (each = one "voiced segment")
    unvoiced_runs = []         # list of unvoiced run lengths in frames
    prev_voiced = False
    current_unvoiced_run = 0

    for i in range(1, n_frames + 1):
        v = call(pitch, "Get value in frame", i, "Hertz")
        is_voiced = (v is not None and
                     not (isinstance(v, float) and math.isnan(v)) and
                     v > 0)
        if is_voiced:
            voiced_frames += 1
            if not prev_voiced:
                # Starting a new voiced segment
                voiced_segments += 1
                if current_unvoiced_run > 0:
                    unvoiced_runs.append(current_unvoiced_run)
                    current_unvoiced_run = 0
            prev_voiced = True
        else:
            if prev_voiced:
                breaks += 1
            current_unvoiced_run += 1
            prev_voiced = False

    phon_ratio = voiced_frames / n_frames if n_frames > 0 else 0

    # Speech rate: voiced segments per second of total recording
    # This is a transcription-free proxy for syllable rate. It correlates
    # reasonably with syllable rate for connected speech but is faster to
    # compute and works on any audio.
    speech_rate = voiced_segments / duration if duration > 0 else None

    # Mean pause duration: mean length of unvoiced runs that are "real" pauses
    # (i.e. longer than ~150 ms — shorter gaps are inter-segment transitions,
    # not pauses). Converts frame counts to seconds.
    min_pause_frames = max(1, int(0.15 / time_step))
    real_pauses = [r * time_step for r in unvoiced_runs if r >= min_pause_frames]
    mean_pause_duration = (sum(real_pauses) / len(real_pauses)) if real_pauses else None

    # --- Formants ---
    formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    f1 = safe(call(formant, "Get mean", 1, 0, 0, "Hertz"))
    f2 = safe(call(formant, "Get mean", 2, 0, 0, "Hertz"))

    def fmt(v, d=2):
        return round(v, d) if v is not None else None

    # Map each feature name → (value, unit)
    # Names here must match FEATURE_GROUPS exactly.
    values = {
        'F0 Mean (average pitch)': (fmt(f0_mean), 'Hz'),
        'F0 SD (pitch variability)': (fmt(f0_sd), 'Hz'),
        'F0 Range (pitch range)': (fmt(f0_range), 'Hz'),
        'Intensity SD (loudness variability)': (fmt(int_sd), 'dB'),

        'HNR (voice clarity)': (fmt(hnr), 'dB'),
        'Cepstral Peak Prominence (voice quality)': (fmt(cpp), 'dB'),
        'Shimmer (volume steadiness)':
            (fmt(shimmer * 100 if shimmer else None, 3), '%'),
        'Voice Breaks (voicing interruptions)': (breaks, 'count'),

        'Jitter (pitch steadiness)':
            (fmt(jitter * 100 if jitter else None, 3), '%'),
        'Intensity Mean (average loudness)': (fmt(int_mean), 'dB'),

        'Phonation Ratio (speaking-to-silence)': (fmt(phon_ratio, 3), 'ratio'),
        'Speech Rate (voiced segments/sec)': (fmt(speech_rate, 2), 'seg/s'),
        'Mean Pause Duration': (fmt(mean_pause_duration, 2), 'sec'),

        'F1 (first formant)': (fmt(f1), 'Hz'),
        'F2 (second formant)': (fmt(f2), 'Hz'),

        'Recording Duration': (fmt(duration), 'sec'),
    }

    return values


# ============================================================
# GUI APPLICATION
# ============================================================

class VoxamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("voxam — Voice Feature Extraction")
        self.root.geometry("1060x760")
        self.root.configure(bg="#f5f5f3")

        self.current_values = None      # dict from extract_voice_features
        self.current_audio_path = None
        self.feature_vars = {}          # feature_name → BooleanVar
        self.feature_notes = {}         # feature_name → clinical note string

        # Flatten groups into a lookup for notes
        for _group_name, features in FEATURE_GROUPS:
            for name, note in features:
                self.feature_notes[name] = note

        self._build_ui()

    def _build_ui(self):
        # Main container
        main = tk.Frame(self.root, bg="#f5f5f3")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        title = tk.Label(
            main, text="voxam",
            font=("Helvetica", 22, "bold"),
            bg="#f5f5f3", fg="#2c2c2a"
        )
        title.pack(anchor="w")

        subtitle = tk.Label(
            main,
            text="Voice feature extraction for clinical research.\n"
                 "All processing happens on your computer — no data leaves your machine.",
            font=("Helvetica", 10),
            bg="#f5f5f3", fg="#5f5e5a",
            justify=tk.LEFT
        )
        subtitle.pack(anchor="w", pady=(2, 16))

        # Controls row
        controls = tk.Frame(main, bg="#f5f5f3")
        controls.pack(fill=tk.X, pady=(0, 12))

        self.select_btn = tk.Button(
            controls, text="Select audio file...",
            command=self.select_file,
            font=("Helvetica", 11),
            padx=16, pady=8,
            bg="#534AB7", fg="white",
            activebackground="#3C3489", activeforeground="white",
            relief=tk.FLAT, cursor="hand2"
        )
        self.select_btn.pack(side=tk.LEFT)

        self.save_btn = tk.Button(
            controls, text="Save results as CSV",
            command=self.save_csv,
            font=("Helvetica", 11),
            padx=16, pady=8,
            bg="#e8e8e4", fg="#5f5e5a",
            activebackground="#d0d0cc",
            relief=tk.FLAT, cursor="hand2",
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=(10, 0))

        # File label
        self.file_label = tk.Label(
            main, text="No file selected.",
            font=("Helvetica", 9),
            bg="#f5f5f3", fg="#888780",
            anchor="w"
        )
        self.file_label.pack(fill=tk.X, pady=(0, 12))

        # ===== Two-column layout: sidebar (feature selection) + results =====
        body = tk.Frame(main, bg="#f5f5f3")
        body.pack(fill=tk.BOTH, expand=True)

        # --- Left: feature selection sidebar ---
        sidebar = tk.Frame(body, bg="white", relief=tk.FLAT, bd=1,
                           highlightbackground="#d3d1c7", highlightthickness=1,
                           width=320)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        sidebar.pack_propagate(False)

        sidebar_header = tk.Frame(sidebar, bg="white")
        sidebar_header.pack(fill=tk.X, padx=12, pady=(10, 6))

        tk.Label(
            sidebar_header, text="Features to display",
            font=("Helvetica", 11, "bold"),
            bg="white", fg="#2c2c2a"
        ).pack(side=tk.LEFT)

        # Select-all / select-none links
        link_frame = tk.Frame(sidebar_header, bg="white")
        link_frame.pack(side=tk.RIGHT)

        tk.Button(
            link_frame, text="All",
            font=("Helvetica", 9, "underline"),
            bg="white", fg="#534AB7", bd=0, cursor="hand2",
            activebackground="white", activeforeground="#3C3489",
            command=self.select_all
        ).pack(side=tk.LEFT, padx=2)

        tk.Label(link_frame, text="|", bg="white", fg="#d3d1c7").pack(side=tk.LEFT)

        tk.Button(
            link_frame, text="None",
            font=("Helvetica", 9, "underline"),
            bg="white", fg="#534AB7", bd=0, cursor="hand2",
            activebackground="white", activeforeground="#3C3489",
            command=self.select_none
        ).pack(side=tk.LEFT, padx=2)

        # Scrollable area for checkboxes
        canvas = tk.Canvas(sidebar, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=canvas.yview)
        checkbox_frame = tk.Frame(canvas, bg="white")

        checkbox_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=checkbox_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=(0, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=(0, 10))

        # Enable mousewheel scrolling on the sidebar
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Build grouped checkboxes
        for group_name, features in FEATURE_GROUPS:
            group_label = tk.Label(
                checkbox_frame, text=group_name,
                font=("Helvetica", 10, "bold"),
                bg="white", fg="#534AB7",
                anchor="w"
            )
            group_label.pack(fill=tk.X, pady=(8, 2), padx=4)

            for feature_name, _note in features:
                var = tk.BooleanVar(value=True)
                self.feature_vars[feature_name] = var

                cb = tk.Checkbutton(
                    checkbox_frame, text=feature_name,
                    variable=var,
                    font=("Helvetica", 9),
                    bg="white", fg="#2c2c2a",
                    activebackground="white",
                    selectcolor="white",
                    anchor="w",
                    command=self._refresh_display,
                    wraplength=260, justify=tk.LEFT
                )
                cb.pack(fill=tk.X, padx=12, anchor="w")

        # --- Right: results table ---
        results_frame = tk.Frame(body, bg="white", relief=tk.FLAT, bd=1,
                                  highlightbackground="#d3d1c7", highlightthickness=1)
        results_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Style the treeview
        style = ttk.Style()
        style.theme_use("default")
        style.configure("VP.Treeview",
                       background="white",
                       foreground="#2c2c2a",
                       rowheight=28,
                       fieldbackground="white",
                       font=("Helvetica", 10),
                       borderwidth=0)
        style.configure("VP.Treeview.Heading",
                       background="#ebebe6",
                       foreground="#2c2c2a",
                       font=("Helvetica", 10, "bold"),
                       borderwidth=0,
                       relief=tk.FLAT)
        style.map("VP.Treeview",
                 background=[("selected", "#e0ddf8")],
                 foreground=[("selected", "#2c2c2a")])

        columns = ("feature", "value", "unit", "domain")
        self.tree = ttk.Treeview(
            results_frame, columns=columns, show="headings",
            style="VP.Treeview", height=18
        )
        self.tree.heading("feature", text="Feature")
        self.tree.heading("value", text="Value")
        self.tree.heading("unit", text="Unit")
        self.tree.heading("domain", text="Physiological system")
        self.tree.column("feature", width=300, anchor="w")
        self.tree.column("value", width=90, anchor="e")
        self.tree.column("unit", width=70, anchor="w")
        self.tree.column("domain", width=220, anchor="w")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Placeholder when empty
        self.placeholder = tk.Label(
            results_frame,
            text="Select an audio file to see extracted voice features.\n\n"
                 "Tips for good recordings:\n"
                 "• Hold microphone 10–20 cm from mouth\n"
                 "• Record in a quiet room (no HVAC or background hum)\n"
                 "• 20–30 seconds of speech, or 5 seconds of sustained \"ahhh\"",
            font=("Helvetica", 10),
            bg="white", fg="#888780",
            justify=tk.LEFT
        )
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")

        # Status bar
        self.status = tk.Label(
            main, text="Ready.",
            font=("Helvetica", 9),
            bg="#f5f5f3", fg="#888780",
            anchor="w"
        )
        self.status.pack(fill=tk.X, pady=(12, 0))

        footer = tk.Label(
            main,
            text="Research tool. Not a diagnostic instrument. "
                 "Interpret values in clinical context and consider recording quality.",
            font=("Helvetica", 8, "italic"),
            bg="#f5f5f3", fg="#888780",
            anchor="w"
        )
        footer.pack(fill=tk.X, pady=(4, 0))

    # ---------- checkbox helpers ----------
    def select_all(self):
        for var in self.feature_vars.values():
            var.set(True)
        self._refresh_display()

    def select_none(self):
        for var in self.feature_vars.values():
            var.set(False)
        self._refresh_display()

    def _get_selected_feature_names(self):
        """Return an ordered list of selected features, preserving group order."""
        selected = []
        for _group_name, features in FEATURE_GROUPS:
            for name, _note in features:
                if self.feature_vars[name].get():
                    selected.append(name)
        return selected

    def _get_domain_for(self, feature_name):
        """Find the domain (group name) for a given feature."""
        for group_name, features in FEATURE_GROUPS:
            for name, _note in features:
                if name == feature_name:
                    return group_name
        return ""

    # ---------- file handling ----------
    def select_file(self):
        filepath = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.wav *.aiff *.flac"),
                ("WAV files", "*.wav"),
                ("All files", "*.*"),
            ]
        )
        if not filepath:
            return

        self.current_audio_path = filepath
        self.file_label.config(text=f"Analysing: {os.path.basename(filepath)}")
        self.status.config(text="Extracting features...")
        self.select_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.root.update()

        for item in self.tree.get_children():
            self.tree.delete(item)
        self.placeholder.place_forget()

        thread = threading.Thread(target=self._run_extraction, args=(filepath,))
        thread.daemon = True
        thread.start()

    def _run_extraction(self, filepath):
        try:
            values = extract_voice_features(filepath)
            self.current_values = values
            self.root.after(0, self._after_extraction)
        except Exception as e:
            self.root.after(0, self._show_error, str(e))

    def _after_extraction(self):
        self._refresh_display()
        self.status.config(
            text=f"Done. Tick/untick features to change the display. Save selected features as CSV."
        )
        self.select_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL, bg="#534AB7", fg="white")

        warnings = self._check_recording_quality()
        if warnings:
            messagebox.showwarning(
                "Recording quality check",
                "Some features suggest your recording may have quality issues:\n\n" +
                "\n".join(f"• {w}" for w in warnings) +
                "\n\nConsider re-recording in a quieter environment, closer to the microphone, "
                "or using a better microphone. Timing features (phonation ratio, voice breaks, "
                "pause duration, speech rate) remain reliable even in noisy conditions."
            )

    def _refresh_display(self):
        """Redraw the results table based on current checkbox state."""
        for item in self.tree.get_children():
            self.tree.delete(item)

        if not self.current_values:
            return

        selected = self._get_selected_feature_names()
        if not selected:
            return

        for name in selected:
            value, unit = self.current_values.get(name, (None, ""))
            display_value = str(value) if value is not None else "—"
            domain = self._get_domain_for(name)
            self.tree.insert("", tk.END, values=(name, display_value, unit, domain))

    def _check_recording_quality(self):
        warnings = []
        if not self.current_values:
            return warnings

        hnr_val = self.current_values.get('HNR (voice clarity)', (None, ""))[0]
        loudness_val = self.current_values.get('Intensity Mean (average loudness)', (None, ""))[0]

        if hnr_val is not None and hnr_val < 10:
            warnings.append(
                f"Voice clarity (HNR) is very low ({hnr_val} dB). Normal is above 20 dB. "
                "Background noise may be dominating the recording."
            )
        if loudness_val is not None and loudness_val < 40:
            warnings.append(
                f"Average loudness is low ({loudness_val} dB). "
                "Try recording closer to the microphone."
            )
        return warnings

    def _show_error(self, error_msg):
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")
        self.status.config(text="Error processing file.")
        self.select_btn.config(state=tk.NORMAL)

        helpful_msg = f"Could not process this file.\n\n"
        if "audio file" in error_msg.lower() or "sound" in error_msg.lower():
            helpful_msg += (
                "The file format may not be supported.\n\n"
                "Try converting to .wav format first using a free online converter "
                "like cloudconvert.com, then try again.\n\n"
            )
        helpful_msg += f"Technical detail: {error_msg}"

        messagebox.showerror("Processing error", helpful_msg)

    # ---------- CSV export ----------
    def save_csv(self):
        if not self.current_values:
            return

        selected = self._get_selected_feature_names()
        if not selected:
            messagebox.showinfo(
                "No features selected",
                "Tick at least one feature checkbox before saving."
            )
            return

        default_name = "voxam_results.csv"
        if self.current_audio_path:
            base = os.path.splitext(os.path.basename(self.current_audio_path))[0]
            default_name = f"{base}_voxam_results.csv"

        filepath = filedialog.asksaveasfilename(
            title="Save results as CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            rows = []
            for name in selected:
                value, unit = self.current_values.get(name, (None, ""))
                rows.append({
                    'Feature': name,
                    'Value': value if value is not None else '',
                    'Unit': unit,
                    'Physiological System': self._get_domain_for(name),
                    'Clinical Note': self.feature_notes.get(name, ''),
                })
            rows.append({
                'Feature': '--- Metadata ---',
                'Value': '', 'Unit': '',
                'Physiological System': '', 'Clinical Note': '',
            })
            rows.append({
                'Feature': 'Audio file',
                'Value': os.path.basename(self.current_audio_path)
                         if self.current_audio_path else '',
                'Unit': '', 'Physiological System': '', 'Clinical Note': '',
            })
            rows.append({
                'Feature': 'Analysis timestamp',
                'Value': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Unit': '', 'Physiological System': '', 'Clinical Note': '',
            })

            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)

            self.status.config(text=f"Saved: {os.path.basename(filepath)}")
            messagebox.showinfo(
                "Saved",
                f"Results saved to:\n{filepath}\n\n"
                f"{len(selected)} features written."
            )
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save file:\n{e}")


def main():
    root = tk.Tk()
    app = VoxamApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
