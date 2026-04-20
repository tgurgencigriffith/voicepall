# voxam

# voxam — Getting Started

A voice feature extraction tool for palliative care research.
All processing happens on your computer — no data leaves your machine.
You need to install Python.
Clinicians: I am a clinician - I promise this is straight forward. If run into trouble, ask Anthropic's Claude LLM for help - but you should be fine.

---

## First-time setup (about 10 minutes, one time only)

### Step 1 — Install Python

**Windows:**
1. Go to https://www.python.org/downloads/
2. Click the big yellow "Download Python" button
3. Run the installer you just downloaded
4. **IMPORTANT:** On the first screen, tick the box that says **"Add Python to PATH"** before clicking Install
5. Click "Install Now" and wait for it to finish

**Mac:**
1. Go to https://www.python.org/downloads/
2. Click the big yellow "Download Python" button
3. Open the downloaded file and follow the installer prompts

### Step 2 — Install the voxam libraries

**Windows:**
1. Press the Windows key, type `cmd`, and press Enter to open the Command Prompt
2. Copy and paste this exact line, then press Enter:
   ```
   pip install praat-parselmouth pandas numpy
   ```
3. Wait for it to finish (about 1 minute). You'll see lots of text scroll by. That's fine.

**Mac:**
1. Open the Terminal app (press Cmd+Space, type "Terminal", press Enter)
2. Copy and paste this exact line, then press Enter:
   ```
   pip3 install praat-parselmouth pandas numpy
   ```
3. Wait for it to finish.

### Step 3 — Download voxam

1. Download `voxam.pyw` (and `voxam.bat` if you're on Windows) to a folder you can find easily, such as your Desktop.

That's it. Setup is done.

---

## Using voxam

### Every time you want to analyse a recording:

1. **Double-click `voxam.bat`** (Windows) or **double-click `voxam.pyw`** (Mac)
2. A window opens titled "voxam — Voice Feature Extraction"
3. Click **"Select audio file..."**
4. Browse to your audio recording and click Open
5. Results appear in the window a few seconds later
6. Click **"Save results as CSV"** to save the table to a spreadsheet file

### Recording tips

For the most reliable results:

- **Get close to the microphone.** 10–20 cm is ideal. Further away = noisier recordings = less reliable voice quality measurements.
- **Find a quiet room.** Air conditioning, heating, fans, oxygen concentrators, and TVs all add noise that degrades voice quality measurements. If you can't eliminate them, record a few seconds of "silence" in the same room so you can estimate the noise floor.
- **Speak normally.** No need to project or articulate carefully — speak as you usually would.
- **Aim for 20–30 seconds of speech.** Longer is fine, but shorter than about 3 seconds makes most features unreliable.
- **For the cleanest voice quality measurements,** try a sustained "ahhhh" for about 5 seconds. This is the gold standard task for jitter/shimmer/HNR.

### Supported file formats

Best: `.wav`, `.aiff`, `.flac`

If your recording is `.mp3` or `.m4a` (common for phone recordings), convert it to `.wav` first. A free online converter like cloudconvert.com works well — takes about 10 seconds.

### What the output means

Each row in the results table is one acoustic feature. The "Symptom domain" column tells you which clinical symptoms that feature is relevant to, based on published research in adjacent fields (heart failure, depression, respiratory disease, cognitive impairment).

Important notes:

- These are research outputs, not diagnostic values
- Normal ranges come from studies of healthy adults and may not apply directly to palliative populations
- Recording quality matters enormously — the app will warn you if voice clarity or loudness suggest a poor recording
- Timing features (speaking ratio, voice breaks) are more robust to noisy environments than voice quality features (jitter, shimmer, HNR)

---

## Troubleshooting

**"Python is not recognized" or similar error on Windows:**
You didn't tick "Add Python to PATH" during installation. Uninstall Python, reinstall, and this time tick the box.

**"Missing libraries" error when opening voxam:**
You haven't run the `pip install` command from Step 2. Go back and run it.

**"Could not process this file" error:**
Your audio is probably in a format Praat doesn't support (like .m4a). Convert it to .wav using a free online converter.

**Results look wrong or show lots of "—" values:**
Your recording is probably too short, too quiet, or too noisy. Record again with the microphone closer to your mouth in a quieter room.

**Still stuck:**
Please contact t.gurgenci at griffith.edu.au with a description of what you tried and what happened.

---

## Citing voxam

If you use voxam in published research, please cite:

[Paper citation to be added]

And please also cite the underlying libraries:

- Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. *Journal of Phonetics*, 71, 1-15.
- Boersma, P., & Weenink, D. (2024). Praat: doing phonetics by computer. Available from http://www.praat.org/
