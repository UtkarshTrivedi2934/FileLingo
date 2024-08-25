
import streamlit as st
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import unidecode
from contextlib import redirect_stdout
import os
import torch
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast
import PyPDF2
import pytesseract
from PIL import Image
import re
import nltk
from spellchecker import SpellChecker

nltk.download('punkt')

# Load summarization model
model_directory = "/content/drive/MyDrive/ML02/bart-large-cnn"
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model=model_directory, device=device)

# Load translation model
# Language to code mapping
language_code_map = {
    "Arabic": "ar_AR",
    "Czech": "cs_CZ",
    "German": "de_DE",
    "English": "en_XX",
    "Spanish": "es_XX",
    "Estonian": "et_EE",
    "Finnish": "fi_FI",
    "French": "fr_XX",
    "Hindi": "hi_IN",
    "Italian": "it_IT",
    "Japanese": "ja_XX",
    "Kazakh": "kk_KZ",
    "Korean": "ko_KR",
    "Lithuanian": "lt_LT",
    "Latvian": "lv_LV",
    "Burmese": "my_MM",
    "Nepali": "ne_NP",
    "Dutch": "nl_XX",
    "Romanian": "ro_RO",
    "Russian": "ru_RU",
    "Sinhala": "si_LK",
    "Turkish": "tr_TR",
    "Vietnamese": "vi_VN",
    "Chinese": "zh_CN",
    "Afrikaans": "af_ZA",
    "Azerbaijani": "az_AZ",
    "Bengali": "bn_IN",
    "Persian": "fa_IR",
    "Hebrew": "he_IL",
    "Croatian": "hr_HR",
    "Indonesian": "id_ID",
    "Georgian": "ka_GE",
    "Khmer": "km_KH",
    "Macedonian": "mk_MK",
    "Malayalam": "ml_IN",
    "Mongolian": "mn_MN",
    "Marathi": "mr_IN",
    "Polish": "pl_PL",
    "Pashto": "ps_AF",
    "Portuguese": "pt_XX",
    "Swedish": "sv_SE",
    "Swahili": "sw_KE",
    "Tamil": "ta_IN",
    "Telugu": "te_IN",
    "Thai": "th_TH",
    "Tagalog": "tl_XX",
    "Ukrainian": "uk_UA",
    "Urdu": "ur_PK",
    "Xhosa": "xh_ZA",
    "Galician": "gl_ES",
    "Slovene": "sl_SI"
}

@st.cache_data
def extract_text_from_pdf_PyMuPDF(pdf_path):
    extracted_text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            extracted_text += page.get_text() + "\n"
    return extracted_text

def extract_text_from_image(image_path):
    with Image.open(image_path) as img:
        text = pytesseract.image_to_string(img)
    return text

@st.cache_data
def extract_text(file_path):
    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf_PyMuPDF(file_path)
    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        return extract_text_from_image(file_path)
    else:
        return "Unsupported file format."

def clean_text1(text):
    cleaned_text = re.sub(r'\s+', ' ', text.strip())
    sentences = nltk.sent_tokenize(cleaned_text)
    cleaned_text = ' '.join(sentences)
    return cleaned_text

def clean_text2(text):
    words = nltk.word_tokenize(text)
    spell = SpellChecker()
    corrected_words = []
    i = 0
    while i < len(words):
        word = words[i]
        if i < len(words) - 1:
            combined_word = word + words[i + 1]
            if combined_word in spell:
                corrected_words.append(combined_word)
                i += 1
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
        i += 1
    corrected_text = ' '.join(corrected_words)
    corrected_text = re.sub(r'\s([?.!,";:](?:\s|$))', r'\1', corrected_text)
    corrected_text = re.sub(r'([?.!,";:])([a-zA-Z])', r'\1 \2', corrected_text)
    sentences = nltk.sent_tokenize(corrected_text)
    final_text = ' '.join(sentences)
    return final_text

def summarized_text(context_data_string):
    max_tokens_per_chunk = 800
    overlap_tokens = 50
    summaries = []
    start_index = 0
    while start_index < len(context_data_string):
        end_index = min(start_index + max_tokens_per_chunk, len(context_data_string))
        chunk = context_data_string[start_index:end_index]
        max_length = min(len(chunk) - overlap_tokens, 80)
        summary = summarizer(
            chunk,
            max_length=max_length,
            min_length=50,
            do_sample=False
        )
        if summary:
            summaries.append(summary[0]['summary_text'])
        start_index += max_tokens_per_chunk - overlap_tokens
    total_summary = ""
    for summary_text in summaries:
        total_summary += f"{summary_text}\n"
    return total_summary

def load_translation_model():
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    return model, tokenizer

model, tokenizer = load_translation_model()

def translate(source_lang, target_lang, sentence):
    tokenizer.src_lang = source_lang
    encoded_sentence = tokenizer(sentence, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_sentence,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def split_summary_into_chunks(summary):
    sentences = nltk.tokenize.sent_tokenize(summary)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= 60:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

st.title("FileLingo: OCR, Summarization and Translation")

uploaded_file = st.file_uploader("Upload PDF or Image", type=['pdf', 'jpg', 'jpeg', 'png'])
source_lang_name = st.selectbox("Select Source Language", list(language_code_map.keys()))
target_lang_name = st.selectbox("Select Target Language", list(language_code_map.keys()))

if st.button("Run"):
    if uploaded_file and source_lang_name and target_lang_name:
        source_lang = language_code_map.get(source_lang_name)
        target_lang = language_code_map.get(target_lang_name)

        if source_lang is None or target_lang is None:
            st.error("Invalid source or target language.")
        else:
            file_path = "/tmp/" + uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            extracted_text = extract_text(file_path)
            cleaned_text = clean_text1(extracted_text)
            cleaned_text = clean_text2(cleaned_text)
            summary = summarized_text(cleaned_text)
            chunks = split_summary_into_chunks(summary)

            final_translation = ""
            for chunk in chunks:
                translation = translate(source_lang, target_lang, chunk)
                final_translation += translation + " "

            st.subheader("Translated Summary")
            st.write(final_translation)

        # with open('/tmp/summary.txt', 'w') as f:
        #     f.write(final_translation)
        # st.download_button(label="Download Summary", data=open('/tmp/summary.txt', 'r').read(), file_name='summary.txt')
