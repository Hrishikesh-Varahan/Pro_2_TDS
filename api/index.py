# api/index.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional, Tuple
import base64, io, os, re, csv, zipfile, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import requests

app = FastAPI()

# ----------- Helpers -----------
def read_all_files(files: List[UploadFile]) -> dict:
    """
    Reads multiple uploaded files into a dict[str, str or bytes].
    Text-like files returned as str; images as bytes; zip auto-expanded.
    """
    out = {}
    if not files:
        return out

    os.makedirs("/tmp/uploads", exist_ok=True)
    for f in files:
        name = f.filename or "file"
        path = os.path.join("/tmp/uploads", name)
        content = f.file.read()
        with open(path, "wb") as w:
            w.write(content)

        low = name.lower()
        if low.endswith(".zip"):
            extract_dir = path + "_unzipped"
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(path, "r") as z:
                z.extractall(extract_dir)
            # Read supported files from zip
            for root, _, fnames in os.walk(extract_dir):
                for fn in fnames:
                    p = os.path.join(root, fn)
                    low2 = fn.lower()
                    if low2.endswith(".csv"):
                        out[fn] = open(p, "r", encoding="utf-8", errors="ignore").read()
                    elif low2.endswith(".md") or low2.endswith(".txt"):
                        out[fn] = open(p, "r", encoding="utf-8", errors="ignore").read()
                    elif low2.endswith(".xlsx") or low2.endswith(".xls"):
                        df = pd.read_excel(p)
                        out[fn] = df.to_csv(index=False)
                    else:
                        # keep raw bytes for arbitrary files (e.g., images)
                        out[fn] = open(p, "rb").read()
        elif low.endswith(".csv"):
            out[name] = content.decode("utf-8", errors="ignore")
        elif low.endswith(".md") or low.endswith(".txt"):
            out[name] = content.decode("utf-8", errors="ignore")
        elif low.endswith(".xlsx") or low.endswith(".xls"):
            df = pd.read_excel(path)
            out[name] = df.to_csv(index=False)
        else:
            out[name] = content  # raw bytes (e.g., images)
    return out


def encode_plot_to_data_uri(fig, format="png", max_bytes=100_000) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format=format, bbox_inches="tight", dpi=150)
    data = buf.getvalue()
    plt.close(fig)

    # If too large, try smaller DPI then JPEG/webp
    if len(data) > max_bytes:
        for dpi in (130, 110, 90):
            buf = io.BytesIO()
            fig.savefig(buf, format=format, bbox_inches="tight", dpi=dpi)
            data = buf.getvalue()
            if len(data) <= max_bytes:
                break
        if len(data) > max_bytes and format == "png":
            # Fallback to JPEG to reduce size
            buf = io.BytesIO()
            fig.savefig(buf, format="jpeg", bbox_inches="tight", dpi=110)
            data = buf.getvalue()

    b64 = base64.b64encode(data).decode("utf-8")
    mime = "image/png" if format == "png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


# ----------- Task: Wikipedia highest-grossing films -----------
def answer_highest_grossing_films(questions_text: str) -> list:
    """
    Scrapes the Wikipedia table and answers the 4 demo questions:
    1) How many $2 bn movies before 2000?
    2) Earliest film over $1.5 bn
    3) Correlation between Rank and Peak
    4) Scatterplot Rank vs Peak with dotted RED regression line, returned as base64 data URI (<100kB)
    """
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    tables = pd.read_html(url)  # chooses all tables on the page
    # Heuristic: find table with columns including Rank and Title and Year and Peak
    df = None
    for t in tables:
        cols = [c.lower() for c in t.columns.astype(str).tolist()]
        if any("rank" in c for c in cols) and any("title" in c for c in cols) and any("year" in c for c in cols):
            df = t
            break
    if df is None:
        raise HTTPException(status_code=500, detail="Could not locate the films table.")

    # Standardize columns
    df.columns = [re.sub(r"\W+", "_", str(c).strip().lower()) for c in df.columns]
    # Typical columns: rank, title, worldwide_gross, year, peak, ...
    # Clean numeric columns
    def to_num(x):
        if pd.isna(x): return np.nan
        s = str(x)
        s = re.sub(r"[^0-9.\-]", "", s)
        return float(s) if s else np.nan

    if "rank" not in df.columns and "rank_1" in df.columns:
        df["rank"] = df["rank_1"]
    if "peak" not in df.columns and "peak_1" in df.columns:
        df["peak"] = df["peak_1"]

    for c in ("rank", "peak", "year"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Q1: How many $2 bn movies released before 2000?
    # We need a gross column; try to detect:
    gross_col = None
    for c in df.columns:
        if "gross" in c and "world" in c:
            gross_col = c; break
    if gross_col is None:
        # fallback: any column containing gross
        for c in df.columns:
            if "gross" in c:
                gross_col = c; break
    if gross_col:
        gross_num = df[gross_col].map(to_num)
        two_bn_before_2000 = int(((gross_num >= 2_000_000_000) & (df.get("year", pd.Series([np.nan]*len(df))) < 2000)).sum())
    else:
        two_bn_before_2000 = 0  # unlikely historically, but safe default

    # Q2: Earliest film that grossed over $1.5 bn
    earliest_title = ""
    if gross_col:
        over_15 = df[(df[gross_col].map(to_num) >= 1_500_000_000) & df["year"].notna()]
        if not over_15.empty:
            earliest_row = over_15.sort_values("year", ascending=True).iloc[0]
            earliest_title = str(earliest_row.get("title", ""))

    # Q3: Correlation between Rank and Peak
    corr = np.nan
    if "rank" in df.columns and "peak" in df.columns:
        s = df[["rank", "peak"]].dropna()
        if len(s) >= 2:
            corr = float(s["rank"].corr(s["peak"]))

    # Q4: Scatterplot + dotted RED regression line
    img_uri = ""
    if "rank" in df.columns and "peak" in df.columns:
        data = df[["rank", "peak"]].dropna()
        if len(data) >= 2:
            X = data["rank"].values.reshape(-1, 1)
            y = data["peak"].values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)

            fig = plt.figure()
            plt.scatter(data["rank"], data["peak"])
            plt.plot(data["rank"], y_pred, linestyle=":", color="red")  # dotted red
            plt.xlabel("Rank")
            plt.ylabel("Peak")
            plt.title("Rank vs Peak (with dotted red regression line)")
            img_uri = encode_plot_to_data_uri(fig, "png", max_bytes=100_000)

    return [two_bn_before_2000, earliest_title, 0.0 if np.isnan(corr) else round(corr, 6), img_uri]


def route_task(questions_txt: str, attachments: dict) -> Tuple[object, int]:
    """
    Minimal router that recognizes known demo prompts and returns answers
    in the requested structure. Extend with more handlers as needed.
    """
    q = questions_txt.strip()

    # Demo: highest-grossing films
    if "highest grossing films" in q.lower() and "wikipedia" in q.lower():
        answers = answer_highest_grossing_films(q)
        # Spec: respond as JSON array of 4 elements
        return answers, 200

    # Add more routes here: DuckDB/S3 parquet demo, sales, weather, etc.
    # For unknown tasks, just return the LLM fallback (optional)
    return {"error": "Unrecognized task format in questions.txt"}, 400


# ----------- API endpoint (/api/) -----------
@app.post("/")
async def api_root(
    files: Optional[List[UploadFile]] = File(None)
):
    """
    Accepts multipart/form-data where one of the parts is ALWAYS `questions.txt`.
    There may be zero or more additional files.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Please send multipart form-data with at least questions.txt")

    # Find questions.txt
    questions_part = None
    for f in files:
        if (f.filename or "").lower() == "questions.txt":
            questions_part = f
            break
    if not questions_part:
        raise HTTPException(status_code=400, detail="questions.txt is required and must be named exactly questions.txt")

    questions_txt = (await questions_part.read()).decode("utf-8", errors="ignore")

    # Read all files (including questions again is fine)
    attachments = read_all_files(files)

    # Route to a task-specific solver
    result, status = route_task(questions_txt, attachments)
    return JSONResponse(content=result, status_code=status)
