## Industry Classification Code Mapper

This project provides a web interface for mapping industry descriptions to ISIC (International Standard Industrial Classification) codes using Qdrant and SentenceTransformers. It allows users to upload files containing industry descriptions and automatically maps them to relevant ISIC codes using semantic search techniques.

### Features

1. **Batch Processing:** Efficiently encodes industry descriptions and upserts data into the Qdrant vector database using SentenceTransformers.

2. **File Upload:** Supports uploading files in Excel (.xlsx) and CSV formats, which contain industry descriptions for processing.

3. **ISIC Code Mapping:** The application maps industries and descriptions to the top 3 most relevant ISIC codes using cosine similarity.

4. **Download Processed Data:** After processing, users can download the mapped ISIC codes as an Excel or CSV file.

5. **Time Tracking:** Displays the time taken to load ISIC data and process files.

### Technologies Used

1. **Streamlit** - For building the web interface.

2. **Qdrant** - A vector search engine to store and search vector embeddings.

3. **SentenceTransformers** - To encode industry descriptions into vector embeddings.

4. **Pandas** - For data manipulation and processing.

5. **OpenAI** - For semantic search and vector comparisons (optional, if integrated for fine-tuning).

### How It Works

1. **ISIC Data Load and Encode:** The app loads the ISIC structure and definitions from an external source. It uses SentenceTransformers to encode the definitions into vector embeddings, which are then stored in Qdrant for similarity search.

2. **File Upload:** Users can upload their own Excel or CSV files containing two columns: INDUSTRY and DESCRIPTION. These fields are concatenated and then encoded.

3. **ISIC Code Search:** Each industry description is encoded and compared against the pre-stored ISIC definitions using cosine similarity. The app returns the top 3 ISIC codes based on the similarity of vectors.

4. **Download Processed Data:** Once the industry descriptions are processed, users can download the file with the matched ISIC codes.
