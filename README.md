# Key Features of the FileLingo Application

- OCR (Optical Character Recognition) Integration:
The application can extract text from various image formats (jpg, jpeg, png) and PDFs using Tesseract-OCR and PyMuPDF.
Supports multi-page PDFs, ensuring all content is captured and processed.

- Multi-Language Summarization:
Utilizes the bart-large-cnn model for summarizing extracted text.
The summarization is performed in chunks to handle large documents effectively, ensuring concise and relevant summaries.

- Language Translation:
Translates the summarized text between over 50 languages using Facebook’s mBART-large model.
The translation process is capable of handling complex language pairs, making it versatile for various linguistic needs.

- Advanced Text Cleaning and Correction:
The application includes two stages of text cleaning:
  - Basic Cleaning: Removes unnecessary whitespace and structures the text into coherent sentences.
  - Spell Correction: Uses the SpellChecker library to correct common spelling mistakes and tokenization errors.

- User-Friendly Interface:
Built on Streamlit, providing an intuitive and accessible web interface.
Features like file upload and language selection are made easy to use, ensuring a smooth user experience.

- Chunked Processing for Large Texts:
The application intelligently splits large summaries into manageable chunks for translation, ensuring that even lengthy documents are processed without errors.

- Support for Multiple File Formats:
Handles both PDF and image files, making it a versatile tool for various document types.
Capable of processing both text-heavy and image-heavy documents, adapting the OCR and summarization approach accordingly.

- Cross-Platform Compatibility:
The application can be run on different operating systems (Windows, macOS, Linux) with appropriate Python and library installations.

- Scalable and Modular Design:
The code is designed to be easily extended, allowing for the integration of additional languages, models, or functionalities as needed.
These features make FileLingo a powerful tool for multilingual document processing, offering robust capabilities in OCR, text summarization, and translation within a streamlined application.
