# Simple Audio Transcription (Whisper)

Transcribe a single audio file with OpenAI Whisper and save a Markdown transcript that includes timestamps and basic metadata. The script favors a lightweight path that reads WAV directly without FFmpeg. If that path fails or the input is not WAV, Whisper's default loading may require FFmpeg.

---

## Features

- Single-file transcription
- Markdown output with timestamps and metadata
- Selectable Whisper model size: `tiny`, `base`, `small`, `medium`, `large`
- Optional SSL verification bypass flag for model downloads during troubleshooting

---

## Requirements

- Python 3.8 or newer
- Python packages:
  - `openai-whisper`
  - `numpy`
  - `scipy`
- FFmpeg is optional but recommended for non-WAV inputs or when Whisper's default loader is used

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install FFmpeg if needed:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install ffmpeg`
- Windows: download from https://ffmpeg.org/download.html and add it to PATH

## Quick Start

```bash
python simple_transcribe.py path/to/audio.wav
```

The script writes a Markdown file next to the input or to a directory you specify with `--output`.

## Command-line Usage

```bash
python simple_transcribe.py [-h] [--model {tiny,base,small,medium,large}] [--output OUTPUT] [--no-ssl-verify] input
```

### Arguments

- `input` - Path to the audio file. WAV is recommended for best compatibility with the direct read path.
- `--model {tiny,base,small,medium,large}` - Whisper model size to use. Default is `tiny`.
- `--output OUTPUT` - Directory for the Markdown transcript. If omitted, the transcript is written next to the input file.
- `--no-ssl-verify` - Disable SSL certificate verification for model downloads. Use only for troubleshooting in controlled environments.

## Output

The script produces a Markdown file named after the input, for example:

```
transcripts/
└── example.md
```

The document contains:
- Title
- Metadata block with source file name and transcription timestamp
- Full transcription text
- Segment list with `[HH:MM:SS - HH:MM:SS]` timestamps

## Recommendations

- For long recordings or limited hardware, choose a smaller model such as `tiny` or `base`.
- Prefer WAV input. Non-WAV formats often require FFmpeg.
- Keep `--no-ssl-verify` for diagnostics only. Restore normal SSL verification after resolving certificate issues.

## License

MIT License. See `LICENSE`.

Copyright (c) 2025 Adeel Ahsan

## Author

Adeel Ahsan - https://www.aeronautyy.com
