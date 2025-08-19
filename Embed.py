# Import libraries and modules
from PyPDF2 import PdfReader
import oracledb
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import OracleVS
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import BaseTransformer, Document
print("Successfully imported libraries and modules")

# Declare username, password, and DSN (data connection string)
username = "ADMIN"
password = "HemantGahankari15!"
dsn = '''(description=(retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)(host=adb.us-ashburn-1.oraclecloud.com))(connect_data=(service_name=adb.us-ashburn-1.oraclecloud.com))(security=(ssl_server_dn_match=yes)))'''

# Connect to the database
try:
    conn23c = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connection successful!")
except Exception as e:
    print("Connection failed!")

# Load and transform a document
# RAG Step 1 - Load the document and create pdf reader object
pdf = PdfReader('./pdf-docs/Oracle Cloud Infrastructure AI Foundations.pdf')

# RAG Step 2 - Extract text from the PDF document
text = ""
for page in pdf.pages:
    text += page.extract_text()
print("You have transformed the PDF document to text format")

# RAG Step 3 - Chunk the text document into smaller chunks
text_splitter = CharacterTextSplitter(separator=".", chunk_size=2000, chunk_overlap=100)
chunks = text_splitter.split_text(text)

# Function to format and add metadata to Oracle 23ai Vector Store
def chunks_to_docs_wrapper(row: dict) -> Document:
    """
    Converts text into a Document object suitable for ingestion into Oracle Vector Store.
    - row (dict): A dictionary representing a row of data with keys for 'id', 'link', and 'text'.
    """
    metadata = {'id': row['id'], 'link': row['link']}
    return Document(page_content=row['text'], metadata=metadata)

# RAG Step 4 - Create metadata wrapper to store additional information in the vector store
docs = [chunks_to_docs_wrapper({'id': str(page_num), 'link': f'Page {page_num}', 'text': text}) for page_num, text in enumerate(chunks)]

COMPARTMENT_OCID = "ocid1.compartment.oc1..aaaaaaaah3o77etfcfp2025jxks2pucwvz6veg26z51gpx3q355nikleemg"

# RAG Step 5 - Using an embedding model, embed the chunks as vectors into Oracle Database 23ai.

embed_model = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=COMPARTMENT_OCID
)

# Optional alternative embedding model
# model_4db = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# RAG Step 6 - Configure the vector store with the model, table name, and distance strategy for similarity search

knowledge_base = OracleVS.from_documents(
    docs,
    embed_model,
    client=conn23c,
    table_name="MY_DEMO",
    distance_strategy=DistanceStrategy.DOT_PRODUCT
)