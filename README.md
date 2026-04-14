# CEFR Text Workshop

A three-tab tool for CEFR-aligned text generation, analysis, and level extension.

## Tabs

### 📝 Generate
- Produces CEFR-levelled texts (A1–B2) using Claude, constrained by EGP grammar profiles and EVP vocabulary profiles
- Targets the *distribution profile* of authentic texts at each level (not hard vocabulary fences)
- Supports: reading passages, dialogues, emails, instructions, narratives, and multiple-choice questions
- Cultural sensitivity filter for school-aged Arabic students
- Optional comprehension questions for reading passages

### 📊 Analyse
- Profiles any English text for grammar and vocabulary complexity
- Grammar analysis via the POLKE API (EGP construct detection)
- Vocabulary analysis via the English Vocabulary Profile (EVP)
- Stacked bar comparison against baseline profiles from 477 CEFR-tagged reference texts
- Ceiling flags highlighting items above the target level

### 🔄 Extension
- Paraphrases generated text one level above and/or below
- For differentiated instruction — same content at different complexity levels
- "Produce Audio" button (placeholder for future TTS integration)

## Deployment

1. Fork this repo
2. Place these files in the root directory:
   - `EVP_Complete_ALL.xlsx` (English Vocabulary Profile data)
   - `egp_list.xlsx` (English Grammar Profile construct list)
   - `grammar_prompts.json` (generated from EGP — included)
   - `vocab_lists.json` (generated from EVP — included)
3. Add your Anthropic API key to Streamlit secrets:
   ```toml
   # .streamlit/secrets.toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Connect the repo to Streamlit Community Cloud
5. Deploy

## Local development

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Data sources

- Grammar constructs: [English Grammar Profile](https://www.englishprofile.org/english-grammar-profile/) via [POLKE](https://polke.kibi.group/)
- Vocabulary levels: [English Vocabulary Profile](https://www.englishprofile.org/wordlists/evp)
- Baseline texts: [UniversalCEFR](https://huggingface.co/UniversalCEFR) (elg-cefr-en dataset, CC BY-NC-SA 4.0)
- Text generation: [Claude](https://www.anthropic.com/) (Anthropic)

## License

Tool code: MIT. Data files are subject to their respective licences.
