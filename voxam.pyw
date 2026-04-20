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
Extracts ~13 acoustic features from voice recordings — pitch, voice
quality (jitter/shimmer), clarity (HNR), loudness, timing — and
labels each one with its relevant symptom domain.

Everything runs on your computer. Audio files never leave your machine.

SUPPORTED FORMATS:
------------------
.wav (best), .aiff, .flac
For .mp3 or .m4a, please convert to .wav first using a free online
converter like cloudconvert.com
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
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
# FEATURE EXTRACTION (same logic as the command-line version)
# ============================================================

def safe(val):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def extract_voice_features(audio_path):
    """Extract clinically-labelled voice features from an audio file."""
    sound = parselmouth.Sound(audio_path)
    duration = sound.get_total_duration()

    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    f0_mean = safe(call(pitch, "Get mean", 0, 0, "Hertz"))
    f0_sd = safe(call(pitch, "Get standard deviation", 0, 0, "Hertz"))
    f0_min = safe(call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic"))
    f0_max = safe(call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic"))
    f0_range = (f0_max - f0_min) if (f0_max is not None and f0_min is not None) else None

    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
    jitter = safe(call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
    shimmer = safe(call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))

    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = safe(call(harmonicity, "Get mean", 0, 0))

    intensity = call(sound, "To Intensity", 75, 0.0)
    int_mean = safe(call(intensity, "Get mean", 0, 0, "dB"))
    int_sd = safe(call(intensity, "Get standard deviation", 0, 0))

    n_frames = call(pitch, "Get number of frames")
    voiced = 0
    breaks = 0
    prev_voiced = False
    for i in range(1, n_frames + 1):
        v = call(pitch, "Get value in frame", i, "Hertz")
        is_voiced = (v is not None and
                     not (isinstance(v, float) and math.isnan(v)) and
                     v > 0)
        if is_voiced:
            voiced += 1
        if prev_voiced and not is_voiced:
            breaks += 1
        prev_voiced = is_voiced
    phon_ratio = voiced / n_frames if n_frames > 0 else 0

    formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    f1 = safe(call(formant, "Get mean", 1, 0, 0, "Hertz"))
    f2 = safe(call(formant, "Get mean", 2, 0, 0, "Hertz"))

    def fmt(v, d=2):
        return round(v, d) if v is not None else None

    # Structure: (name, value, unit, domain, note)
    features = [
        ('Average Pitch', fmt(f0_mean), 'Hz', 'Mood / Energy',
         'Lower → fatigue, depression, sedation. Higher → anxiety, pain.'),
        ('Pitch Variability', fmt(f0_sd), 'Hz', 'Mood / Expression',
         'Reduced (monotone) → depression, fatigue, sedation.'),
        ('Pitch Range', fmt(f0_range), 'Hz', 'Mood / Expression',
         'Compressed → flat affect. Wider → expressive speech.'),
        ('Jitter (pitch steadiness)', fmt(jitter * 100 if jitter else None, 3), '%',
         'Neuromuscular Control',
         'Higher → unsteady voice. Normal < 1.04%. Elevated → fatigue, weakness.'),
        ('Shimmer (volume steadiness)', fmt(shimmer * 100 if shimmer else None, 3), '%',
         'Respiratory Support',
         'Higher → unsteady volume. Normal < 3.81%. Elevated → breathlessness, fatigue.'),
        ('Voice Clarity (HNR)', fmt(hnr), 'dB', 'Respiratory / Laryngeal',
         'Higher = clearer. Lower → breathy/hoarse. Normal > 20 dB.'),
        ('Average Loudness', fmt(int_mean), 'dB', 'Energy / Respiratory',
         'Reduced → fatigue, weakness, reduced respiratory capacity.'),
        ('Loudness Variability', fmt(int_sd), 'dB', 'Energy / Expression',
         'Reduced → flat delivery, fatigue, depression.'),
        ('Speaking-to-Silence Ratio', fmt(phon_ratio, 3), 'ratio',
         'Dyspnoea / Cognition',
         'Lower → more silence → breathlessness or cognitive slowing.'),
        ('Voice Breaks', breaks, 'count', 'Respiratory / Fatigue',
         'More breaks → more interruptions in voicing. Breathlessness or vocal effort.'),
        ('Formant F1', fmt(f1), 'Hz', 'Fluid Status / Airway',
         'Changes may reflect oedema, dry mouth, or altered oral cavity.'),
        ('Formant F2', fmt(f2), 'Hz', 'Fluid Status / Airway',
         'Changes over time may reflect fluid shifts or airway changes.'),
        ('Recording Duration', fmt(duration), 'sec', 'Metadata',
         'Total length of the recording.'),
    ]

    return features


# ============================================================
# GUI APPLICATION
# ============================================================

class voxamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("voxam — Voice Feature Extraction")
        self.root.geometry("760x720")
        self.root.configure(bg="#f5f5f3")

        self.current_features = None
        self.current_audio_path = None

        self._build_ui()

    def _build_ui(self):
        # Main container with padding
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
        subtitle.pack(anchor="w", pady=(2, 20))

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

        # Current file label
        self.file_label = tk.Label(
            main, text="No file selected.",
            font=("Helvetica", 9),
            bg="#f5f5f3", fg="#888780",
            anchor="w"
        )
        self.file_label.pack(fill=tk.X, pady=(0, 12))

        # Results area - using a Treeview for a clean table
        results_frame = tk.Frame(main, bg="white", relief=tk.FLAT, bd=1,
                                  highlightbackground="#d3d1c7", highlightthickness=1)
        results_frame.pack(fill=tk.BOTH, expand=True)

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
            style="VP.Treeview", height=14
        )
        self.tree.heading("feature", text="Feature")
        self.tree.heading("value", text="Value")
        self.tree.heading("unit", text="Unit")
        self.tree.heading("domain", text="Symptom domain")
        self.tree.column("feature", width=240, anchor="w")
        self.tree.column("value", width=90, anchor="e")
        self.tree.column("unit", width=70, anchor="w")
        self.tree.column("domain", width=220, anchor="w")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Placeholder text when empty
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

        # Footer note
        footer = tk.Label(
            main,
            text="Research tool. Not a diagnostic instrument. "
                 "Interpret values in clinical context and consider recording quality.",
            font=("Helvetica", 8, "italic"),
            bg="#f5f5f3", fg="#888780",
            anchor="w"
        )
        footer.pack(fill=tk.X, pady=(4, 0))

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

        # Clear previous results
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.placeholder.place_forget()

        # Run extraction in a thread so the UI stays responsive
        thread = threading.Thread(target=self._run_extraction, args=(filepath,))
        thread.daemon = True
        thread.start()

    def _run_extraction(self, filepath):
        try:
            features = extract_voice_features(filepath)
            self.current_features = features
            self.root.after(0, self._display_results, features)
        except Exception as e:
            self.root.after(0, self._show_error, str(e))

    def _display_results(self, features):
        for name, value, unit, domain, note in features:
            display_value = str(value) if value is not None else "—"
            self.tree.insert("", tk.END, values=(name, display_value, unit, domain))

        self.status.config(
            text=f"Done. Extracted {len(features)} features. Hover a row for clinical notes, or save as CSV."
        )
        self.select_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL, bg="#534AB7", fg="white")

        # Check for suspicious values and warn if needed
        warnings = self._check_recording_quality(features)
        if warnings:
            messagebox.showwarning(
                "Recording quality check",
                "Some features suggest your recording may have quality issues:\n\n" +
                "\n".join(f"• {w}" for w in warnings) +
                "\n\nConsider re-recording in a quieter environment, closer to the microphone, "
                "or using a better microphone. The timing features (speaking ratio, voice breaks) "
                "remain reliable even in noisy conditions."
            )

    def _check_recording_quality(self, features):
        """Flag obvious recording quality issues."""
        warnings = []
        feature_dict = {f[0]: f[1] for f in features}

        hnr = feature_dict.get('Voice Clarity (HNR)')
        loudness = feature_dict.get('Average Loudness')

        if hnr is not None and hnr < 10:
            warnings.append(
                f"Voice clarity (HNR) is very low ({hnr} dB). Normal is above 20 dB. "
                "Background noise may be dominating the recording."
            )
        if loudness is not None and loudness < 40:
            warnings.append(
                f"Average loudness is low ({loudness} dB). "
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

    def save_csv(self):
        if not self.current_features:
            return

        # Suggest a filename based on the audio file
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
            for name, value, unit, domain, note in self.current_features:
                rows.append({
                    'Feature': name,
                    'Value': value,
                    'Unit': unit,
                    'Symptom Domain': domain,
                    'Clinical Note': note,
                })
            rows.append({
                'Feature': '--- Metadata ---',
                'Value': '',
                'Unit': '',
                'Symptom Domain': '',
                'Clinical Note': '',
            })
            rows.append({
                'Feature': 'Audio file',
                'Value': os.path.basename(self.current_audio_path) if self.current_audio_path else '',
                'Unit': '',
                'Symptom Domain': '',
                'Clinical Note': '',
            })
            rows.append({
                'Feature': 'Analysis timestamp',
                'Value': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Unit': '',
                'Symptom Domain': '',
                'Clinical Note': '',
            })

            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)

            self.status.config(text=f"Saved: {os.path.basename(filepath)}")
            messagebox.showinfo(
                "Saved",
                f"Results saved to:\n{filepath}"
            )
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save file:\n{e}")


def main():
    root = tk.Tk()
    app = voxamApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
