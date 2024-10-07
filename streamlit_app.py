# streamlit_app.py

import streamlit as st
import polars as pl
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect, DetectorFactory, LangDetectException
import torch
from io import BytesIO
import pandas as pd
from tqdm import tqdm
import logging

# Set seed for langdetect to ensure consistency
DetectorFactory.seed = 0

# ----------------------------
# Logging Configuration
# ----------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ----------------------------
# Function Definitions
# ----------------------------

def detect_language(text: str) -> str:
    """
    Detects the language of a given text string.

    Parameters:
        text (str): The text whose language needs to be detected.

    Returns:
        str: The detected language code (ISO 639-1). Returns 'unknown' if detection fails.

    Example:
        >>> detect_language("Bonjour tout le monde")
        'fr'
    """
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

@st.cache_resource
def load_model(model_name: str = "facebook/m2m100_418M") -> tuple:
    """
    Loads the M2M100 model and tokenizer.

    Parameters:
        model_name (str): The name of the pre-trained M2M100 model.

    Returns:
        tuple: A tuple containing the tokenizer and model.

    Example:
        >>> tokenizer, model = load_model()
    """
    try:
        logger.info(f"Loading tokenizer and model: {model_name}")
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info("Model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load model and tokenizer: {e}")
        raise

def translate_texts(
    texts: list,
    source_langs: list,
    target_lang: str = "en",
    batch_size: int = 8,
    model: M2M100ForConditionalGeneration = None,
    tokenizer: M2M100Tokenizer = None,
    device: torch.device = torch.device("cpu")
) -> list:
    """
    Translates a list of texts from their source languages to a target language.

    Parameters:
        texts (list): List of text strings to be translated.
        source_langs (list): List of source language codes corresponding to each text.
        target_lang (str, optional): The target language code for translation. Defaults to 'en'.
        batch_size (int, optional): Number of texts to process in each batch. Defaults to 8.
        model (M2M100ForConditionalGeneration, optional): The pre-loaded translation model.
        tokenizer (M2M100Tokenizer, optional): The pre-loaded tokenizer.
        device (torch.device, optional): The device to run the model on. Defaults to CPU.

    Returns:
        list: List of translated text strings.

    Example:
        >>> texts = ["Hola, ¿cómo estás?", "Bonjour tout le monde"]
        >>> source_langs = ["es", "fr"]
        >>> translate_texts(texts, source_langs, target_lang="en", batch_size=2, model=model, tokenizer=tokenizer, device=device)
        ['Hello, how are you?', 'Hello everyone']
    """
    translations = []
    total_batches = (len(texts) + batch_size - 1) // batch_size  # Ceiling division for total batches

    for i in tqdm(range(0, len(texts), batch_size), desc="Translating", total=total_batches):
        batch_texts = texts[i:i + batch_size]
        batch_src_langs = source_langs[i:i + batch_size]

        try:
            # Check if all source languages in the batch are the same for efficiency
            if len(set(batch_src_langs)) == 1:
                # Set the source language for the tokenizer
                tokenizer.src_lang = batch_src_langs[0]
                logger.info(f"Translating batch {i//batch_size + 1}/{total_batches} with source language '{batch_src_langs[0]}'.")

                # Tokenize the batch of texts
                encoded = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)

                # Generate translation using the model
                with torch.no_grad():
                    generated_tokens = model.generate(
                        **encoded,
                        forced_bos_token_id=tokenizer.get_lang_id(target_lang)
                    )

                # Decode the generated tokens back to text
                decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                translations.extend(decoded)
            else:
                # For mixed languages in a batch, handle each text individually
                for text, src_lang in zip(batch_texts, batch_src_langs):
                    logger.info(f"Translating text with source language '{src_lang}'.")
                    tokenizer.src_lang = src_lang
                    encoded = tokenizer(
                        [text],
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(device)

                    with torch.no_grad():
                        generated_tokens = model.generate(
                            **encoded,
                            forced_bos_token_id=tokenizer.get_lang_id(target_lang)
                        )

                    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    translations.extend(decoded)
        except Exception as e:
            logger.error(f"Error during translation of batch starting at index {i}: {e}")
            translations.extend(["Error during translation"] * len(batch_texts))

    return translations

def convert_df_to_excel(df: pl.DataFrame) -> BytesIO:
    """
    Converts a Polars DataFrame to an Excel file in memory.

    Parameters:
        df (pl.DataFrame): The Polars DataFrame to convert.

    Returns:
        BytesIO: In-memory bytes buffer containing the Excel file.
    """
    try:
        # Convert Polars DataFrame to Pandas DataFrame for Excel compatibility
        pandas_df = df.to_pandas()
        output = BytesIO()
        # Write the DataFrame to the in-memory buffer
        pandas_df.to_excel(output, index=False, engine='openpyxl')
        # Seek to the beginning of the buffer
        output.seek(0)
        logger.info("DataFrame converted to Excel format successfully.")
        return output
    except Exception as e:
        logger.error(f"Failed to convert DataFrame to Excel: {e}")
        raise

# ----------------------------
# Streamlit App Layout
# ----------------------------

def main():
    """
    Main function to run the Streamlit translation app.
    """
    try:
        st.set_page_config(
            page_title="Excel Column Translator",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        st.title("📊 Excel Column Translator")
        st.markdown("""
        This application allows you to **upload an Excel file**, **preview the data**, **translate a selected column** into a target language, and **download the translated file**.
        
        **Features**:
        - Supports multiple source languages.
        - Utilizes the **M2M100** model for efficient multilingual translation.
        - Ensures data privacy by performing translations locally without API calls.
        """)
        
        # Sidebar for user inputs
        st.sidebar.header("🔧 Configuration")

        # File uploader
        uploaded_file = st.sidebar.file_uploader(
            "Upload your Excel file",
            type=["xlsx", "xls"],
            help="Supported formats: .xlsx, .xls"
        )

        if uploaded_file:
            # Read the Excel file into a Polars DataFrame
            try:
                df = pl.read_excel(uploaded_file)
                logger.info("Excel file uploaded and read successfully.")
                st.success("✅ File uploaded successfully!")
            except Exception as e:
                logger.error(f"Error reading the Excel file: {e}")
                st.error(f"❌ Error reading the Excel file: {e}")
                st.stop()

            # Display the DataFrame preview
            st.subheader("📄 Data Preview")
            try:
                st.dataframe(df.head(10).to_pandas())
                logger.info("Displayed DataFrame preview.")
            except Exception as e:
                logger.error(f"Error displaying DataFrame preview: {e}")
                st.error(f"❌ Error displaying DataFrame preview: {e}")

            # Select column to translate
            column_options = df.columns
            if not column_options:
                logger.warning("Uploaded Excel file contains no columns.")
                st.error("❌ The uploaded Excel file does not contain any columns.")
                st.stop()

            column_to_translate = st.sidebar.selectbox(
                "Select the column to translate",
                options=column_options,
                help="Choose the column containing text to translate."
            )

            # Select target language
            target_language = st.sidebar.selectbox(
                "Select target language",
                options=[
                    ("English", "en"),
                    ("Spanish", "es"),
                    ("French", "fr"),
                    ("German", "de"),
                    ("Italian", "it"),
                    ("Russian", "ru"),
                    ("Japanese", "ja"),
                    ("Chinese", "zh"),
                    # Add more languages as needed
                ],
                index=0,
                help="Choose the language you want to translate the selected column into."
            )

            # Button to start translation
            if st.sidebar.button("🚀 Start Translation"):
                with st.spinner("Translating... This may take a while depending on the size of your data."):
                    try:
                        # Load the model and tokenizer
                        tokenizer, model = load_model()
                        
                        # Extract texts from the selected column
                        texts = df[column_to_translate].to_list()
                        logger.info(f"Extracted {len(texts)} texts from column '{column_to_translate}'.")

                        # Detect languages for each text
                        st.info("🔍 Detecting languages of the texts...")
                        detected_langs = []
                        for text in tqdm(texts, desc="Detecting Languages"):
                            lang = detect_language(text)
                            detected_langs.append(lang)
                        logger.info("Language detection completed.")

                        # Verify lengths
                        logger.info(f"Number of rows in df: {df.height}")
                        logger.info(f"Number of detected languages: {len(detected_langs)}")
                        if df.height != len(detected_langs):
                            logger.error("Mismatch between DataFrame rows and detected languages.")
                            st.error("❌ Mismatch between DataFrame rows and detected languages.")
                            st.stop()

                        # Add detected languages to the DataFrame
                        try:
                            df = df.with_column(pl.Series("detected_lang", detected_langs))
                            logger.info("Added 'detected_lang' column to DataFrame.")
                        except Exception as e:
                            logger.error(f"Error adding 'detected_lang' column: {e}")
                            st.error(f"❌ Error adding 'detected_lang' column: {e}")
                            st.stop()

                        # Handle unknown languages
                        if "unknown" in detected_langs:
                            unknown_count = detected_langs.count("unknown")
                            st.warning(f"There are {unknown_count} texts with undetectable languages. They will not be translated.")
                            logger.warning(f"{unknown_count} texts have undetectable languages.")

                        # Filter out texts with unknown languages
                        valid_indices = [i for i, lang in enumerate(detected_langs) if lang != "unknown"]
                        texts_to_translate = [texts[i] for i in valid_indices]
                        source_langs = [detected_langs[i] for i in valid_indices]
                        ids_to_translate = [df["id"][i] for i in valid_indices] if "id" in df.columns else valid_indices

                        logger.info(f"Translating {len(texts_to_translate)} texts.")

                        # Translate texts
                        if texts_to_translate:
                            translated_texts = translate_texts(
                                texts=texts_to_translate,
                                source_langs=source_langs,
                                target_lang=target_language[1],
                                batch_size=8,
                                model=model,
                                tokenizer=tokenizer,
                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            )
                            logger.info("Translation process completed.")

                            # Create a new column for translations
                            translated_column = [None] * len(df)
                            for idx, translated_text in zip(valid_indices, translated_texts):
                                translated_column[idx] = translated_text

                            # Add the translated column to the DataFrame
                            new_column_name = f"{column_to_translate}_translated_to_{target_language[0]}"
                            try:
                                df = df.with_column(pl.Series(new_column_name, translated_column))
                                logger.info(f"Added '{new_column_name}' column to DataFrame.")
                            except Exception as e:
                                logger.error(f"Error adding translated column '{new_column_name}': {e}")
                                st.error(f"❌ Error adding translated column '{new_column_name}': {e}")
                                st.stop()

                            st.success("✅ Translation completed successfully!")
                        else:
                            logger.error("No texts available for translation.")
                            st.error("❌ No texts available for translation.")
                            st.stop()

                    except Exception as e:
                        logger.error(f"An unexpected error occurred: {e}")
                        st.error(f"❌ An unexpected error occurred: {e}")
                        st.stop()

                # Display the translated DataFrame
                st.subheader("📄 Translated Data Preview")
                try:
                    st.dataframe(
                        df.select([
                            column_to_translate,
                            f"{column_to_translate}_translated_to_{target_language[0]}"
                        ]).head(10).to_pandas()
                    )
                    logger.info("Displayed translated DataFrame preview.")
                except Exception as e:
                    logger.error(f"Error displaying translated DataFrame: {e}")
                    st.error(f"❌ Error displaying translated DataFrame: {e}")

                # Prepare the translated DataFrame for download
                try:
                    translated_excel = convert_df_to_excel(df)
                except Exception as e:
                    logger.error(f"Error converting DataFrame to Excel: {e}")
                    st.error(f"❌ Error converting DataFrame to Excel: {e}")
                    st.stop()

                # Download button
                st.download_button(
                    label="📥 Download Translated Excel",
                    data=translated_excel,
                    file_name="translated_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("📂 Please upload an Excel file to get started.")

if __name__ == "__main__":
    main()
