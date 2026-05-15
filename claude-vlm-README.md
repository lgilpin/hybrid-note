# PDF Classifier
Classifies PDFs as scanned or digital using Claude Haiku as a vision model. Each page is rendered as an image and sent to the model for visual inspection.

## Requirements
- Python 3.10+
- An Anthropic API key

## Setup
1. Clone the repository.

2. Create a virtual environment

    Windows/PC:
    ```
    python3 -m venv venv
    venv\Scripts\activate
    ```
    Mac:
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```

33. Install dependencies listed in requirements.txt
    ```
    pip install -r requirements.txt
    ```
4. . Create a `.env` file in the project root with your API key:

    ```
    ANTHROPIC_API_KEY=your-api-key-here
    ```

3. Place the PDF files inside the `docs\` folder.

## Usage

Run program:
    ```
    python pdf_classifier.py
    ```

By default it looks for PDFs in the `docs/` folder. You can also point it at a different folder:

    ```
    python pdf_classifier.py path/to/folder/
    ```

## Output

The script prints a per-page breakdown for each PDF, then a summary table across all documents:

```
  Page   1/4 ... digital   confidence=97%
  Page   2/4 ... digital   confidence=95%
  ...

  FOLDER SUMMARY
  resume.pdf                         digital   97.0%
  CMPM 118 meeting notes.pdf         scanned   88.0%
```