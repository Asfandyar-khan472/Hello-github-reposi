# 📊 AI Data Analyst

A Streamlit web app to upload, clean, explore and chat with your data using Google Gemini.

## Features
- Upload CSV and Excel (.xlsx) files, with sheet selection
- Data preview and column profiling
- Missing value and duplicate detection
- One-click cleaning: trim text, remove duplicates, fill or drop missing values, remove outliers, rename columns, change column types
- Interactive Plotly charts: histogram, box, scatter, line, bar, pie
- Statistics, correlation heatmap and strongest relationships
- Trend analysis with moving average
- Gemini AI assistant for questions and insights
- Download cleaned data as CSV (or Excel)

## Files
| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `requirements.txt` | Python dependencies |
| `README.md` | This guide |

## Deploy from your browser (no local Python needed)

### 1. Get a Gemini API key
Create a free key at https://aistudio.google.com/apikey

### 2. Put the files on GitHub
1. Sign in at https://github.com and click **New repository**.
2. Name it `ai-data-analyst`, choose Public, and leave "Add a README" unchecked. Click **Create repository**.
3. Click **creating a new file**, name it `app.py`, paste the code, and click **Commit changes**.
4. Use **Add file → Create new file** to add `requirements.txt` and `README.md` the same way.

### 3. Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app → Deploy a public app from GitHub**.
3. Select your repository, branch `main`, and main file path `app.py`.
4. Open **Advanced settings → Secrets** and paste:

```toml
GEMINI_API_KEY = "paste-your-key-here"
```

5. Click **Deploy**. The first build takes a few minutes.

To change the key later: open your app → **Settings → Secrets**.
To update the app: edit a file on GitHub and commit; the app redeploys automatically.

## Notes
- Never put your API key in `app.py` or commit it to GitHub.
- The default model is `gemini-flash-latest`. If Google renames or retires a model, type a new model name in the sidebar.
- Gemini only sees column names, summary statistics and the first 10 rows, not your full dataset.
- Streamlit's default upload limit is 200 MB per file.

## Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Check that `requirements.txt` is in the repo root and spelled correctly |
| "Gemini API key not found" | Add `GEMINI_API_KEY` in Settings → Secrets |
| Gemini model error | Change the model name in the sidebar |
| Excel file will not open | Save it as `.xlsx` (the older `.xls` format is not supported) |
