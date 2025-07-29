import os
import io
import pypdf
import logging
import hashlib 
from datetime import datetime
import requests
import pandas as pd
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# Base URL for PDFs
BASE_URL = 'https://raw.githubusercontent.com/DS4SD/docling/refs/heads/main/tests/data/pdf/'
PDF_FILES = [
    '2203.01017v2.pdf', '2305.03393v1-pg9.pdf', '2305.03393v1.pdf',
    'amt_handbook_sample.pdf', 'code_and_formula.pdf', 'picture_classification.pdf',
    'redp5110_sampled.pdf', 'right_to_left_01.pdf', 'right_to_left_02.pdf', 'right_to_left_03.pdf'
]
INPUT_DOC_PATHS = [os.path.join(BASE_URL, pdf_file) for pdf_file in PDF_FILES]

# Configure PDF processing
pipeline_options = PdfPipelineOptions()
pipeline_options.generate_page_images = True

doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

# Load tokenizer and embedding model
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 64  # Small token limit for demonstration
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
embedding_model = SentenceTransformer(EMBED_MODEL_ID)

chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)

def embed_text(text: str) -> list[float]:
    """Generate an embedding for a given text."""
    return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

def generate_document_rows(conv_results):
    """
    Generator that yields one row per chunk from each successfully converted document.
    Each yielded dict contains:
      - file_name: Name of the source file.
      - raw_markdown: Serialized text for the chunk.
      - chunk_embedding: The embedding vector for that chunk.
    """
    processed_docs = 0
    for conv_res in conv_results:
        if conv_res.status != ConversionStatus.SUCCESS:
            continue

        processed_docs += 1
        file_name = conv_res.input.file.stem  # FIX: Use `.file.stem` instead of `.path`

        # Extract the document object (which contains iterate_items)
        document = conv_res.document
        try:
            document_markdown = document.export_to_markdown()
        except:
            document_markdown = ""
        if document is None:
            _log.warning(f"Document conversion failed for {file_name}")
            continue

        # Process each chunk from the document
        for chunk in chunker.chunk(dl_doc=document):  # Use `document` here!
            raw_chunk = chunker.serialize(chunk=chunk)
            embedding = embed_text(raw_chunk)
            yield {
                "file_name": file_name,
                "full_document_markdown": document_markdown,
                "raw_chunk_markdown": raw_chunk,
                "chunk_embedding": embedding,
            }
    _log.info(f"Processed {processed_docs} documents successfully.")

def generate_chunk_id(file_name: str, raw_chunk_markdown: str) -> str:
    """Generate a unique chunk ID based on file_name and raw_chunk_markdown."""
    unique_string = f"{file_name}-{raw_chunk_markdown}"
    return hashlib.sha256(unique_string.encode()).hexdigest()

conv_results = doc_converter.convert_all(INPUT_DOC_PATHS, raises_on_error=False)

# Build a DataFrame where each row is a unique chunk record
rows = list(generate_document_rows(conv_results))
df = pd.DataFrame.from_records(rows)
output_dict = {}
for file_name in PDF_FILES:
    try:
        r = requests.get(BASE_URL + file_name)
        pdf_bytes = io.BytesIO(r.content)
        output_dict[file_name] = pdf_bytes.getvalue()
    except Exception as e:
        print(f"error with {file_name} \n{e}")

odf = pd.DataFrame.from_dict(output_dict, orient='index', columns=['bytes']).reset_index()
odf.rename({"index": "file_name"}, axis=1, inplace=True)
odf['file_name'] = odf['file_name'].str.replace('.pdf', '')
finaldf = df.merge(odf, on='file_name', how='left')
finaldf["chunk_id"] = finaldf.apply(lambda row: generate_chunk_id(row["file_name"], row["raw_chunk_markdown"]), axis=1)

finaldf['created'] = datetime.now()

finaldf['vector'] = finaldf['chunk_embedding']
finaldf['document_id'] = finaldf['file_name']
odf['document_id'] = odf['file_name']

pdf_example = pypdf.PdfReader(io.BytesIO(finaldf['bytes'].values[0]))
finaldf.drop(['full_document_markdown', 'bytes'], axis=1).to_parquet('feature_repo/data/docling_samples.parquet', index=False)
odf.to_parquet('feature_repo/data/metadata_samples.parquet', index=False)
