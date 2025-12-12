FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/
# Note: In production we probably wouldn't burn the model into the image if it changes often, 
# but for this assignment we will generate it or copy it if exists.
# Since we already ran train.py, src/model/model.pkl exists and will be copied.

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
