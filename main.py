from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import zipfile
import os
import csv
import pandas as pd  # For Excel file support
from llm import get_llm_response  # Importing the LLM function

app = FastAPI()

@app.post("/api/")
async def process_question(
    question: str = Form(...),
    file: UploadFile = File(None)
):
    """
    API endpoint to process a question with an optional CSV, Markdown, Excel, or ZIP file.
    """
    try:
        file_content = None  # Default to None if no~ file is uploaded

        if file:
            file_extension = file.filename.split(".")[-1].lower()
            temp_file_path = file.filename  # Save temporarily

            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(await file.read())

            if file_extension == "csv":
                # Process CSV file
                with open(temp_file_path, mode="r") as csvfile:
                    reader = csv.DictReader(csvfile)
                    file_content = "\n".join([str(row) for row in reader])

            elif file_extension == "md":
                # Process Markdown file
                with open(temp_file_path, mode="r", encoding="utf-8") as mdfile:
                    file_content = mdfile.read()

            elif file_extension in ["xls", "xlsx"]:
                # Process Excel file
                df = pd.read_excel(temp_file_path)
                file_content = df.to_csv(index=False)  # Convert to CSV-like format

            elif file_extension == "zip":
                # Process ZIP file (extract CSV or Markdown)
                extract_path = "extracted_files"
                os.makedirs(extract_path, exist_ok=True)  # Ensure directory exists

                with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)

                extracted_files = os.listdir(extract_path)
                content_list = []

                for f in extracted_files:
                    file_path = f"{extract_path}/{f}"

                    if f.endswith(".csv"):
                        with open(file_path, mode="r") as csvfile:
                            reader = csv.DictReader(csvfile)
                            csv_content = "\n".join([str(row) for row in reader])
                            content_list.append(csv_content)

                    elif f.endswith(".md"):
                        with open(file_path, mode="r", encoding="utf-8") as mdfile:
                            md_content = mdfile.read()
                            content_list.append(md_content)

                if content_list:
                    file_content = "\n\n".join(content_list)

            else:
                raise HTTPException(status_code=400, detail="Unsupported file format. Supported: CSV, MD, XLSX, ZIP.")

            # Clean up temporary file
            os.remove(temp_file_path)

        # Get answer from LLM
        response = get_llm_response(question, file_content)
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

