# Senior Mortgage Underwriting System - Local App

This is a local `app.py` version of the mortgage underwriting notebook.

## Why JSON config?

This project uses `config.json` instead of `config.py` because JSON is cleaner for settings and secrets:
- it is data-only
- it avoids importing executable Python as config
- it is easy to copy from `config.example.json`

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create your private config file:

```bash
cp config.example.json config.json
```

3. Open `config.json` and add your OpenAI API key:

```json
{
  "OPENAI_API_KEY": "your-key-here",
  "OPENAI_BASE_URL": "https://api.openai.com/v1",
  "MODEL_NAME": "gpt-4o-mini"
}
```

4. Run the app:

```bash
python app.py
```

## Git Safety

`config.json` is included in `.gitignore`, so your API key should not be committed.

This ZIP also includes `gitignore.txt` so you can visibly inspect the ignore rules. The actual Git file is named `.gitignore`.

Before pushing to GitHub, check:

```bash
git status
```

Make sure `config.json` does not appear in staged files.
