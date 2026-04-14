"""
CEFR Text Workshop — Streamlit App
====================================
Three-tab tool for CEFR-aligned text generation, analysis, and extension.

Tab 1 — Generate: Produces level-targeted texts using Claude API
Tab 2 — Analyse: Profiles grammar and vocabulary complexity (POLKE + EVP)
Tab 3 — Extension: Paraphrases generated text one level above/below

Deploy: Streamlit Community Cloud (connect to GitHub repo)
"""

import streamlit as st
import pandas as pd
import requests
import re
import json
import spacy
import anthropic
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CEFR Text Workshop",
    page_icon="📝",
    layout="centered",
)

# ============================================================
# CONSTANTS
# ============================================================

LEVELS = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
TARGET_LEVELS = ['A1', 'A2', 'B1', 'B2']
BANDS = ['A1', 'A2', 'B1', 'B2', 'C']
LEVEL_ORDER = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
BAND_ORDER = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C': 4}

TIER1_FILTER = {67, 68, 69, 70, 275, 277, 278, 279, 586, 587, 588, 860, 1058}
POLKE_URL = "https://polke.kibi.group/extractor"
MAX_TEXT_LENGTH = 6000

GRAMMAR_BASELINES = {
    'A1': {'A1': 46.7, 'A2': 29.9, 'B1': 14.5, 'B2': 7.2, 'C': 1.6},
    'A2': {'A1': 43.2, 'A2': 31.6, 'B1': 16.2, 'B2': 7.2, 'C': 1.8},
    'B1': {'A1': 38.1, 'A2': 32.2, 'B1': 18.8, 'B2': 8.5, 'C': 2.4},
    'B2': {'A1': 33.3, 'A2': 31.8, 'B1': 21.2, 'B2': 10.3, 'C': 3.4},
}

VOCAB_BASELINES = {
    'A1': {'A1': 78.9, 'A2': 13.9, 'B1': 6.0, 'B2': 0.8, 'C': 0.4},
    'A2': {'A1': 70.1, 'A2': 22.1, 'B1': 5.8, 'B2': 1.6, 'C': 0.4},
    'B1': {'A1': 55.9, 'A2': 25.2, 'B1': 13.7, 'B2': 4.1, 'C': 1.1},
    'B2': {'A1': 42.8, 'A2': 22.4, 'B1': 19.3, 'B2': 11.6, 'C': 4.0},
}

BAND_COLORS = {
    'A1': '#5DCAA5',
    'A2': '#97C459',
    'B1': '#EF9F27',
    'B2': '#E24B4A',
    'C':  '#7F77DD',
}

FUNCTION_WORDS = {
    'the','a','an','is','are','was','were','am','be','been','being',
    'do','does','did','done','doing','have','has','had','having',
    'will','would','shall','should','can','could','may','might','must',
    'not','no','nor',"n't",'and','or','but','so','yet',
    'i','me','my','mine','myself','you','your','yours','yourself','yourselves',
    'he','him','his','himself','she','her','hers','herself',
    'it','its','itself','we','us','our','ours','ourselves',
    'they','them','their','theirs','themselves',
    'this','that','these','those',
    'who','whom','whose','which','what','where','when','why','how',
    'to','of','in','on','at','for','with','from','by','about',
    'as','into','through','during','before','after','above','below',
    'between','under','over','up','down','out','off','since','until',
    'if','then','than','because','while','although','though','unless',
    'whether','once',
    'there','here','very','too','also','just','only','more','most',
    'quite','rather',
    'some','any','each','every','all','both','few','many','much',
    'such','own','other','another',
}

TEXT_TYPES = {
    'reading': 'Reading Passage',
    'dialogue': 'Dialogue / Conversation',
    'email': 'Email / Letter',
    'instructions': 'Instructions / How-to',
    'narrative': 'Narrative / Story',
    'mc_grammar': 'MC: Grammar Questions',
    'mc_vocab': 'MC: Vocabulary Questions',
    'mc_phrases': 'MC: Phrase Questions',
}

# ============================================================
# LOAD MODELS AND DATA (cached)
# ============================================================

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_evp():
    df = pd.read_excel("EVP_Complete_ALL.xlsx")
    single_words = df[~df['Part of Speech'].isin(['phrase', 'phrasal verb'])]
    word_min_level = {}
    for _, row in single_words.iterrows():
        word = str(row['Base Word']).lower().strip()
        level = row['Level']
        if word not in word_min_level or LEVEL_ORDER[level] < LEVEL_ORDER[word_min_level[word]]:
            word_min_level[word] = level
    return word_min_level

@st.cache_resource
def load_egp():
    df = pd.read_excel("egp_list.xlsx")
    return {
        'level': dict(zip(df['EGP_ID'], df['Level'])),
        'category': dict(zip(df['EGP_ID'], df['SuperCategory'])),
        'cando': dict(zip(df['EGP_ID'], df['Can-do statement'])),
    }

@st.cache_resource
def load_grammar_prompts():
    with open("grammar_prompts.json") as f:
        return json.load(f)

@st.cache_resource
def load_vocab_lists():
    with open("vocab_lists.json") as f:
        return json.load(f)

def get_anthropic_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)

# ============================================================
# GENERATION FUNCTIONS
# ============================================================

def build_system_prompt(level, text_type, topic, word_count, include_questions, grammar_prompts, vocab_lists):
    levels_order = ['A1', 'A2', 'B1', 'B2']
    li = levels_order.index(level)

    # Cumulative grammar
    grammar_lines = []
    for j in range(li + 1):
        lvl = levels_order[j]
        grammar_lines.append(f"\n--- {lvl} Grammar Structures ---")
        grammar_lines.append(grammar_prompts[lvl])

    # Cumulative vocab
    all_words = set()
    for j in range(li + 1):
        all_words.update(vocab_lists[levels_order[j]])
    vocab_list = ', '.join(sorted(all_words))

    # Target distribution profiles
    g_base = GRAMMAR_BASELINES[level]
    v_base = VOCAB_BASELINES[level]

    questions_instruction = ""
    if text_type == 'reading':
        if include_questions:
            questions_instruction = f"\nAfter the passage, include 3-5 comprehension questions. Use simple question formats appropriate to {level} (true/false, short answer, or multiple choice with the correct answer always as option a)."
        else:
            questions_instruction = "\nDo NOT include comprehension questions, discussion prompts, or follow-up tasks. Produce ONLY the text itself."

    text_type_instructions = {
        'reading': f"Write a reading comprehension passage suitable for language learners.{questions_instruction}",
        'dialogue': "Write a natural dialogue/conversation between 2-3 people. Produce ONLY the dialogue — no questions or tasks.",
        'email': "Write an email or letter appropriate to the level. Produce ONLY the email/letter — no questions or tasks.",
        'instructions': "Write a set of instructions or how-to guide. Produce ONLY the instructions — no questions or tasks.",
        'narrative': "Write a short narrative or story. Produce ONLY the narrative — no questions or tasks.",
        'mc_grammar': f"Create 5 multiple-choice grammar questions (4 options each). Test grammar structures from the permitted list for {level}. The correct answer must ALWAYS be option a).\nFormat:\nQ1. [sentence with gap]\na) correct answer  b) distractor  c) distractor  d) distractor",
        'mc_vocab': f"Create 5 multiple-choice vocabulary questions (4 options each). Test vocabulary appropriate to {level}. The correct answer must ALWAYS be option a).\nFormat:\nQ1. [sentence with gap or definition]\na) correct answer  b) distractor  c) distractor  d) distractor",
        'mc_phrases': f"Create 5 multiple-choice phrase/expression questions (4 options each). Test phrases and collocations at {level}. The correct answer must ALWAYS be option a).\nFormat:\nQ1. [sentence with gap]\na) correct answer  b) distractor  c) distractor  d) distractor",
    }

    prompt = f"""You are a CEFR-aligned text producer. You MUST produce content that matches the {level} profile.

TARGET LEVEL: {level}
TEXT TYPE: {text_type_instructions[text_type]}
{f'TOPIC: {topic}' if topic.strip() else 'TOPIC: Choose an appropriate, engaging topic suitable for school-aged learners.'}
{f'TARGET LENGTH: EXACTLY {word_count} words (tolerance: ±10%). Count carefully. Do NOT exceed {int(word_count * 1.1)} words or go below {int(word_count * 0.9)} words.' if word_count else ''}

═══ TARGET DISTRIBUTION PROFILE ═══

Your text should approximate these distributions found in authentic {level} texts:

GRAMMAR distribution target:
  A1 structures: ~{g_base['A1']:.0f}%  |  A2: ~{g_base['A2']:.0f}%  |  B1: ~{g_base['B1']:.0f}%  |  B2: ~{g_base['B2']:.0f}%  |  C: ~{g_base['C']:.0f}%

VOCABULARY distribution target:
  A1 words: ~{v_base['A1']:.0f}%  |  A2: ~{v_base['A2']:.0f}%  |  B1: ~{v_base['B1']:.0f}%  |  B2: ~{v_base['B2']:.0f}%  |  C: ~{v_base['C']:.0f}%

This means a {level} text is NOT 100% at-level. It is predominantly lower-level vocabulary and grammar with a characteristic proportion of at-level and above-level features. Aim for this natural distribution.

═══ REFERENCE GRAMMAR STRUCTURES (cumulative through {level}) ═══

{chr(10).join(grammar_lines)}

═══ REFERENCE VOCABULARY (cumulative through {level}) ═══

{vocab_list}

═══ GENERATION RULES ═══
- Produce natural, authentic-sounding language — not stilted or robotic
- Use simple, short sentences at A1-A2; allow more complex but controlled sentences at B1-B2
- Match the distribution profile above — include mostly lower-level items with some at-level features
- If a topic requires vocabulary not in the reference list, use it sparingly and naturally
- For MC questions, ensure all options are at or below the target level
- Do NOT include any meta-commentary — produce ONLY the requested text

═══ CULTURAL SENSITIVITY ═══
This content is for school-aged Arabic students. Do NOT include references to:
- Pork, pigs, bacon, ham, or pork products
- Alcohol, beer, wine, or alcoholic beverages
- Drugs, smoking, or substance use
- Sexual content or romantic relationships
- Gambling, betting, or lotteries
- Any content disrespectful to Islamic culture or values
Keep all content age-appropriate, culturally respectful, and suitable for a school environment."""

    return prompt

def generate_text(client, system_prompt, user_message):
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text

def generate_extension(client, text, source_level, target_level, grammar_prompts, vocab_lists):
    li = TARGET_LEVELS.index(target_level)

    grammar_lines = []
    for j in range(li + 1):
        lvl = TARGET_LEVELS[j]
        grammar_lines.append(f"\n--- {lvl} Grammar Structures ---")
        grammar_lines.append(grammar_prompts[lvl])

    all_words = set()
    for j in range(li + 1):
        all_words.update(vocab_lists[TARGET_LEVELS[j]])
    vocab_list = ', '.join(sorted(all_words))

    g_base = GRAMMAR_BASELINES[target_level]
    v_base = VOCAB_BASELINES[target_level]

    system_prompt = f"""You are a CEFR text adapter. You rewrite texts to match a target CEFR level while preserving the original content and meaning as closely as possible.

TARGET LEVEL: {target_level}
SOURCE LEVEL: {source_level}

TARGET DISTRIBUTION:
  Grammar: A1 ~{g_base['A1']:.0f}% | A2 ~{g_base['A2']:.0f}% | B1 ~{g_base['B1']:.0f}% | B2 ~{g_base['B2']:.0f}% | C ~{g_base['C']:.0f}%
  Vocabulary: A1 ~{v_base['A1']:.0f}% | A2 ~{v_base['A2']:.0f}% | B1 ~{v_base['B1']:.0f}% | B2 ~{v_base['B2']:.0f}% | C ~{v_base['C']:.0f}%

REFERENCE GRAMMAR (cumulative through {target_level}):
{chr(10).join(grammar_lines)}

REFERENCE VOCABULARY (cumulative through {target_level}):
{vocab_list}

RULES:
- Preserve the original topic, key information, and structure
- {'Simplify grammar and vocabulary. Use shorter sentences, simpler structures, and more common words.' if LEVEL_ORDER[target_level] < LEVEL_ORDER[source_level] else 'Increase complexity. Use more varied grammar structures, longer sentences, and broader vocabulary.'}
- Match the target distribution profile naturally
- Keep approximately the same length as the original
- Do NOT add meta-commentary. Produce ONLY the rewritten text.

CULTURAL SENSITIVITY:
Content is for school-aged Arabic students. No pork, alcohol, drugs, sexual content, gambling, or content disrespectful to Islamic values."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Rewrite this {source_level} text at {target_level} level:\n\n{text}"}],
    )
    return message.content[0].text

# ============================================================
# ANALYSIS FUNCTIONS (from original profiler)
# ============================================================

def clean_text_for_api(text):
    replacements = {
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-', '\u2026': '...', '\u00a0': ' ',
        '\n': ' ', '\r': ' ', '\t': ' ',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.encode('ascii', errors='ignore').decode('ascii')
    text = ' '.join(text.split())
    if len(text) > MAX_TEXT_LENGTH:
        truncated = text[:MAX_TEXT_LENGTH]
        last_period = truncated.rfind('.')
        if last_period > MAX_TEXT_LENGTH * 0.5:
            text = truncated[:last_period + 1]
        else:
            text = truncated
    return text

def query_polke(text):
    text = clean_text_for_api(text)
    params = {'text': text, 'annotate': 'true', 'tokenize': 'false'}
    response = requests.post(POLKE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

def to_5band(profile):
    return {
        'A1': profile.get('A1', 0),
        'A2': profile.get('A2', 0),
        'B1': profile.get('B1', 0),
        'B2': profile.get('B2', 0),
        'C': profile.get('C1', 0) + profile.get('C2', 0),
    }

def get_grammar_profile(polke_response, egp):
    annotations = polke_response.get('annotationList', [])
    unique_cids = set(a['constructID'] for a in annotations
                      if a['constructID'] not in TIER1_FILTER)
    level_counts = Counter()
    for cid in unique_cids:
        cid_level = egp['level'].get(cid, 'Unknown')
        if cid_level in LEVEL_ORDER:
            level_counts[cid_level] += 1
    total = sum(level_counts.values())
    if total == 0:
        return None, [], 0
    profile = {l: level_counts.get(l, 0) / total * 100 for l in LEVELS}
    ceilings = []
    for cid in sorted(unique_cids):
        level = egp['level'].get(cid, 'Unknown')
        if level in LEVEL_ORDER:
            ceilings.append({
                'id': cid, 'level': level,
                'category': egp['category'].get(cid, '?'),
                'cando': egp['cando'].get(cid, '?'),
                'count': sum(1 for a in annotations if a['constructID'] == cid),
            })
    return to_5band(profile), ceilings, len(unique_cids)

def get_vocab_profile(text, nlp, evp):
    doc = nlp(text)
    level_counts = Counter()
    not_found = []
    words_by_level = defaultdict(list)
    skipped = 0
    for token in doc:
        if token.is_punct or token.is_space or token.like_num:
            continue
        word = token.text.lower().strip()
        lemma = token.lemma_.lower().strip()
        if len(word) <= 1:
            continue
        if word in FUNCTION_WORDS or lemma in FUNCTION_WORDS:
            skipped += 1
            continue
        level = evp.get(lemma) or evp.get(word)
        if level is None and word.endswith("'s"):
            level = evp.get(word[:-2])
        if level is not None:
            level_counts[level] += 1
            words_by_level[level].append(word)
        else:
            not_found.append(word)
    total_profiled = sum(level_counts.values())
    total_content = total_profiled + len(not_found)
    if total_profiled == 0:
        return None, [], 0, 0
    profile = {l: level_counts.get(l, 0) / total_profiled * 100 for l in LEVELS}
    coverage = total_profiled / total_content * 100 if total_content > 0 else 0
    ceilings = []
    for level in ['C2', 'C1', 'B2', 'B1', 'A2']:
        unique_words = sorted(set(words_by_level.get(level, [])))
        for w in unique_words:
            ceilings.append({'word': w, 'level': level, 'count': words_by_level[level].count(w)})
    return to_5band(profile), ceilings, coverage, total_profiled

def classify_band(grammar_profile, vocab_profile):
    if not grammar_profile or not vocab_profile:
        return 'B1'
    g_b1p = grammar_profile['B1'] + grammar_profile['B2'] + grammar_profile['C']
    v_b1p = vocab_profile['B1'] + vocab_profile['B2'] + vocab_profile['C']
    best_band, best_dist = 'B1', float('inf')
    for level in TARGET_LEVELS:
        gb = GRAMMAR_BASELINES[level]
        vb = VOCAB_BASELINES[level]
        g_ref = gb['B1'] + gb['B2'] + gb['C']
        v_ref = vb['B1'] + vb['B2'] + vb['C']
        dist = ((g_b1p - g_ref)**2 + (v_b1p - v_ref)**2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_band = level
    return best_band

def get_qualifier(your_b1p, target_b1p, dimension):
    diff = your_b1p - target_b1p
    abs_diff = abs(diff)
    if abs_diff < 2:
        return "✓", f"{dimension} closely matches the target level.", "green"
    if abs_diff < 6:
        degree = "slightly"
    elif abs_diff < 12:
        degree = "moderately"
    else:
        degree = "considerably"
    if diff > 0:
        return "▲", f"{dimension} is {degree} more complex than the target level.", "orange"
    else:
        return "▼", f"{dimension} is {degree} simpler than the target level.", "blue"

# ============================================================
# RENDERING FUNCTIONS
# ============================================================

def render_bar(profile, highlight=False):
    border = "border: 2px solid #534AB7; border-radius: 6px;" if highlight else "opacity: 0.55;"
    height = "32px" if highlight else "24px"
    segments = ""
    for band in BANDS:
        pct = profile.get(band, 0)
        if pct < 0.3:
            continue
        color = BAND_COLORS[band]
        label = f"{band} {pct:.0f}%" if pct >= 5 else ""
        segments += f'<div style="width:{pct}%;background:{color};display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:500;color:rgba(0,0,0,0.7);overflow:hidden;">{label}</div>'
    return f'<div style="display:flex;height:{height};border-radius:4px;overflow:hidden;{border}">{segments}</div>'

def render_comparison(grammar_profile, vocab_profile, target_level):
    target_idx = TARGET_LEVELS.index(target_level)
    below_level = TARGET_LEVELS[max(0, target_idx - 1)] if target_idx > 0 else None

    st.markdown("**Grammar**")
    if below_level:
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown(f'<div style="text-align:right;font-size:12px;color:gray;padding-top:4px;">Typical {below_level}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(render_bar(GRAMMAR_BASELINES[below_level]), unsafe_allow_html=True)
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown('<div style="text-align:right;font-size:12px;color:#534AB7;font-weight:600;padding-top:6px;">Your text</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(render_bar(grammar_profile, highlight=True), unsafe_allow_html=True)
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown(f'<div style="text-align:right;font-size:12px;color:gray;padding-top:4px;">Typical {target_level}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(render_bar(GRAMMAR_BASELINES[target_level]), unsafe_allow_html=True)

    your_g_b1p = grammar_profile['B1'] + grammar_profile['B2'] + grammar_profile['C']
    target_g_b1p = GRAMMAR_BASELINES[target_level]['B1'] + GRAMMAR_BASELINES[target_level]['B2'] + GRAMMAR_BASELINES[target_level]['C']
    icon, text, color = get_qualifier(your_g_b1p, target_g_b1p, "Grammar")
    st.markdown(f'<div style="font-size:13px;margin:4px 0 16px;"><span style="color:{color};">{icon}</span> {text}</div>', unsafe_allow_html=True)

    st.markdown("**Vocabulary**")
    if below_level:
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown(f'<div style="text-align:right;font-size:12px;color:gray;padding-top:4px;">Typical {below_level}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(render_bar(VOCAB_BASELINES[below_level]), unsafe_allow_html=True)
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown('<div style="text-align:right;font-size:12px;color:#534AB7;font-weight:600;padding-top:6px;">Your text</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(render_bar(vocab_profile, highlight=True), unsafe_allow_html=True)
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown(f'<div style="text-align:right;font-size:12px;color:gray;padding-top:4px;">Typical {target_level}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(render_bar(VOCAB_BASELINES[target_level]), unsafe_allow_html=True)

    your_v_b1p = vocab_profile['B1'] + vocab_profile['B2'] + vocab_profile['C']
    target_v_b1p = VOCAB_BASELINES[target_level]['B1'] + VOCAB_BASELINES[target_level]['B2'] + VOCAB_BASELINES[target_level]['C']
    icon, text, color = get_qualifier(your_v_b1p, target_v_b1p, "Vocabulary")
    st.markdown(f'<div style="font-size:13px;margin:4px 0 0;"><span style="color:{color};">{icon}</span> {text}</div>', unsafe_allow_html=True)

def render_ceiling_flags(grammar_ceilings, vocab_ceilings, target_level):
    target_score = LEVEL_ORDER.get(target_level, 2)
    gram_above = sorted(
        [c for c in grammar_ceilings if LEVEL_ORDER.get(c['level'], 0) > target_score],
        key=lambda x: (-LEVEL_ORDER.get(x['level'], 0), -x['count'])
    )
    vocab_above = sorted(
        [c for c in vocab_ceilings if LEVEL_ORDER.get(c['level'], 0) > target_score],
        key=lambda x: (-LEVEL_ORDER.get(x['level'], 0), -x['count'])
    )
    if not gram_above and not vocab_above:
        st.info("No grammar structures or vocabulary above the target level were found.")
        return
    if gram_above:
        st.markdown('<div style="font-size:12px;color:gray;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;margin:8px 0 4px;">Grammar structures</div>', unsafe_allow_html=True)
        for item in gram_above[:8]:
            badge_bg = "#FFF3E0" if item['level'] == 'B1' else "#FCEBEB" if item['level'] == 'B2' else "#EEEDFE"
            badge_fg = "#854F0B" if item['level'] == 'B1' else "#993C1D" if item['level'] == 'B2' else "#3C3489"
            cando = str(item['cando'])[:80]
            st.markdown(
                f'<div style="display:flex;align-items:baseline;gap:8px;padding:5px 0;border-bottom:0.5px solid #eee;font-size:13px;">'
                f'<span style="font-size:11px;font-weight:500;padding:2px 8px;border-radius:6px;background:{badge_bg};color:{badge_fg};flex-shrink:0;">{item["level"]}</span>'
                f'<span><strong>{item["category"]}</strong> <span style="color:gray;">— {cando} (×{item["count"]})</span></span></div>',
                unsafe_allow_html=True
            )
    if vocab_above:
        st.markdown('<div style="font-size:12px;color:gray;font-weight:500;text-transform:uppercase;letter-spacing:0.5px;margin:12px 0 4px;">Vocabulary</div>', unsafe_allow_html=True)
        for item in vocab_above[:12]:
            badge_bg = "#FFF3E0" if item['level'] == 'B1' else "#FCEBEB" if item['level'] == 'B2' else "#EEEDFE"
            badge_fg = "#854F0B" if item['level'] == 'B1' else "#993C1D" if item['level'] == 'B2' else "#3C3489"
            st.markdown(
                f'<div style="display:flex;align-items:baseline;gap:8px;padding:5px 0;border-bottom:0.5px solid #eee;font-size:13px;">'
                f'<span style="font-size:11px;font-weight:500;padding:2px 8px;border-radius:6px;background:{badge_bg};color:{badge_fg};flex-shrink:0;">{item["level"]}</span>'
                f'<span><strong>{item["word"]}</strong> <span style="color:gray;">(×{item["count"]})</span></span></div>',
                unsafe_allow_html=True
            )

# ============================================================
# MAIN APP
# ============================================================

st.title("CEFR Text Workshop")

tab_generate, tab_analyse, tab_extend = st.tabs(["📝 Generate", "📊 Analyse", "🔄 Extension"])

# ────────────────────────────────────────────────────────────
# TAB 1: GENERATE
# ────────────────────────────────────────────────────────────

with tab_generate:
    st.markdown("Generate CEFR-aligned texts using grammar and vocabulary profiles from the EGP and EVP.")

    col1, col2 = st.columns([1, 1])
    with col1:
        gen_level = st.selectbox("Target level", TARGET_LEVELS, index=2, key="gen_level")
    with col2:
        gen_type = st.selectbox("Text type", list(TEXT_TYPES.keys()), format_func=lambda x: TEXT_TYPES[x], key="gen_type")

    gen_topic = st.text_input("Topic (optional)", placeholder="e.g. daily routines, school life, the environment...", key="gen_topic")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        gen_words = st.number_input("Word count", min_value=50, max_value=1000, value=200, step=50, key="gen_words")
    with col2:
        gen_questions = st.checkbox("Include questions", key="gen_questions", help="Add comprehension questions after reading passages")
    with col3:
        pass

    if st.button("Generate", type="primary", use_container_width=True, key="gen_btn"):
        client = get_anthropic_client()
        if client is None:
            st.error("Anthropic API key not configured. Add ANTHROPIC_API_KEY to your Streamlit secrets.")
        else:
            with st.spinner("Loading constraint data..."):
                grammar_prompts = load_grammar_prompts()
                vocab_lists = load_vocab_lists()

            with st.spinner(f"Generating {gen_level} {TEXT_TYPES[gen_type].lower()}..."):
                try:
                    system_prompt = build_system_prompt(
                        gen_level, gen_type, gen_topic, gen_words,
                        gen_questions, grammar_prompts, vocab_lists
                    )
                    type_label = TEXT_TYPES[gen_type].lower()
                    user_msg = (
                        f"Generate a {gen_level} {type_label} about: {gen_topic}"
                        if gen_topic.strip()
                        else f"Generate a {gen_level} {type_label} on an appropriate topic for school-aged learners."
                    )
                    result = generate_text(client, system_prompt, user_msg)
                    st.session_state['generated_text'] = result
                    st.session_state['generated_level'] = gen_level
                except Exception as e:
                    st.error(f"Generation error: {e}")

    # Display generated text
    if 'generated_text' in st.session_state:
        st.divider()
        st.markdown(st.session_state['generated_text'])
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇ Download .txt",
                data=f"CEFR Level: {st.session_state.get('generated_level', '?')}\nText Type: {TEXT_TYPES.get(gen_type, '?')}\nTopic: {gen_topic or '(auto)'}\n{'─' * 40}\n\n{st.session_state['generated_text']}",
                file_name=f"CEFR_{st.session_state.get('generated_level', 'X')}_{gen_type}.txt",
                mime="text/plain",
            )
        with col2:
            if st.button("📊 Analyse this text", key="analyse_generated"):
                st.session_state['analyse_text'] = st.session_state['generated_text']
                st.session_state['analyse_target'] = st.session_state.get('generated_level', 'B1')
                st.rerun()

# ────────────────────────────────────────────────────────────
# TAB 2: ANALYSE
# ────────────────────────────────────────────────────────────

with tab_analyse:
    st.markdown("Analyse English text for grammar and vocabulary complexity against CEFR levels.")

    prefill = st.session_state.get('analyse_text', '')
    text_input = st.text_area("Paste your text here:", value=prefill, height=200,
                              placeholder="Enter at least 50 words for meaningful results. 300+ words recommended.",
                              key="analyse_input")

    # Clear prefill after use
    if 'analyse_text' in st.session_state:
        del st.session_state['analyse_text']

    col1, col2 = st.columns([1, 3])
    with col1:
        default_idx = TARGET_LEVELS.index(st.session_state.get('analyse_target', 'B1'))
        target_level = st.selectbox("Target level:", TARGET_LEVELS, index=default_idx, key="analyse_level")
        if 'analyse_target' in st.session_state:
            del st.session_state['analyse_target']
    with col2:
        st.markdown("")

    if st.button("Analyse", type="primary", use_container_width=True, key="analyse_btn") and text_input.strip():
        word_count = len(text_input.split())
        if word_count < 20:
            st.error("Please enter at least 20 words for analysis.")
        else:
            if word_count < 300:
                st.warning(f"Text is {word_count} words. Results are more reliable with 300+ words.")

            with st.spinner("Loading language models..."):
                nlp = load_spacy()
                evp = load_evp()
                egp = load_egp()

            with st.spinner("Analysing grammar (querying POLKE API)..."):
                try:
                    polke_response = query_polke(text_input)
                    grammar_profile, grammar_ceilings, unique_constructs = get_grammar_profile(polke_response, egp)
                except requests.exceptions.Timeout:
                    st.error("The POLKE grammar server timed out. Please try again.")
                    st.stop()
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the POLKE grammar server.")
                    st.stop()
                except Exception as e:
                    st.error(f"Grammar analysis error: {e}")
                    st.stop()

            with st.spinner("Analysing vocabulary..."):
                vocab_profile, vocab_ceilings, coverage, total_profiled = get_vocab_profile(text_input, nlp, evp)

            if not grammar_profile or not vocab_profile:
                st.error("Could not generate profiles. The text may be too short.")
                st.stop()

            overall_band = classify_band(grammar_profile, vocab_profile)
            g_b1p = grammar_profile['B1'] + grammar_profile['B2'] + grammar_profile['C']
            v_b1p = vocab_profile['B1'] + vocab_profile['B2'] + vocab_profile['C']
            if g_b1p > v_b1p + 3:
                balance = "Grammar is more complex than vocabulary."
            elif v_b1p > g_b1p + 3:
                balance = "Vocabulary is more complex than grammar."
            else:
                balance = "Grammar and vocabulary are at similar levels."

            st.divider()
            bcol1, bcol2 = st.columns([1, 5])
            with bcol1:
                st.markdown(f'<div style="font-size:48px;font-weight:500;text-align:center;padding:8px 0;">{overall_band}</div>', unsafe_allow_html=True)
            with bcol2:
                st.markdown(f"**This text operates primarily in the {overall_band} range.**")
                st.markdown(balance)

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Words", word_count)
            mcol2.metric("EVP coverage", f"{coverage:.0f}%")
            mcol3.metric("Unique constructs", unique_constructs)

            st.divider()
            render_comparison(grammar_profile, vocab_profile, target_level)
            st.divider()

            st.markdown("**Ceiling flags** — items above the target that may challenge students")
            render_ceiling_flags(grammar_ceilings, vocab_ceilings, target_level)

    elif st.session_state.get('analyse_btn') and not text_input.strip():
        st.warning("Please paste some text to analyse.")

# ────────────────────────────────────────────────────────────
# TAB 3: EXTENSION
# ────────────────────────────────────────────────────────────

with tab_extend:
    st.markdown("Produce paraphrased versions of a text one level above and/or below for differentiated instruction.")

    if 'generated_text' not in st.session_state:
        st.info("Generate a text in the **Generate** tab first, then come here to create level extensions.")
    else:
        source_level = st.session_state.get('generated_level', 'B1')
        source_text = st.session_state['generated_text']

        st.markdown(f"**Source text** ({source_level}):")
        with st.expander("View source text", expanded=False):
            st.markdown(source_text)

        li = TARGET_LEVELS.index(source_level)
        can_go_down = li > 0
        can_go_up = li < len(TARGET_LEVELS) - 1

        if can_go_down:
            lower_level = TARGET_LEVELS[li - 1]
        if can_go_up:
            upper_level = TARGET_LEVELS[li + 1]

        col1, col2 = st.columns(2)

        # Level down
        with col1:
            if can_go_down:
                st.markdown(f"### ⬇ {lower_level} Version")
                if st.button(f"Generate {lower_level} version", key="ext_down", use_container_width=True):
                    client = get_anthropic_client()
                    if client is None:
                        st.error("API key not configured.")
                    else:
                        with st.spinner(f"Adapting to {lower_level}..."):
                            grammar_prompts = load_grammar_prompts()
                            vocab_lists = load_vocab_lists()
                            result = generate_extension(client, source_text, source_level, lower_level, grammar_prompts, vocab_lists)
                            st.session_state[f'ext_{lower_level}'] = result

                if f'ext_{lower_level}' in st.session_state:
                    st.markdown(st.session_state[f'ext_{lower_level}'])
                    st.download_button(
                        f"⬇ Download {lower_level} .txt",
                        data=st.session_state[f'ext_{lower_level}'],
                        file_name=f"extension_{lower_level}.txt",
                        key=f"dl_{lower_level}",
                    )
                    st.button(f"🔊 Produce Audio ({lower_level})", key=f"audio_{lower_level}", disabled=True, help="Coming soon")
            else:
                st.info(f"No lower level available (source is {source_level}).")

        # Level up
        with col2:
            if can_go_up:
                st.markdown(f"### ⬆ {upper_level} Version")
                if st.button(f"Generate {upper_level} version", key="ext_up", use_container_width=True):
                    client = get_anthropic_client()
                    if client is None:
                        st.error("API key not configured.")
                    else:
                        with st.spinner(f"Adapting to {upper_level}..."):
                            grammar_prompts = load_grammar_prompts()
                            vocab_lists = load_vocab_lists()
                            result = generate_extension(client, source_text, source_level, upper_level, grammar_prompts, vocab_lists)
                            st.session_state[f'ext_{upper_level}'] = result

                if f'ext_{upper_level}' in st.session_state:
                    st.markdown(st.session_state[f'ext_{upper_level}'])
                    st.download_button(
                        f"⬇ Download {upper_level} .txt",
                        data=st.session_state[f'ext_{upper_level}'],
                        file_name=f"extension_{upper_level}.txt",
                        key=f"dl_{upper_level}",
                    )
                    st.button(f"🔊 Produce Audio ({upper_level})", key=f"audio_{upper_level}", disabled=True, help="Coming soon")
            else:
                st.info(f"No upper level available (source is {source_level}).")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Grammar analysis powered by [POLKE](https://polke.kibi.group/) (University of Tübingen). "
    "Vocabulary profiled against the [English Vocabulary Profile](https://www.englishprofile.org/wordlists/evp). "
    "Grammar constructs mapped via the [English Grammar Profile](https://www.englishprofile.org/english-grammar-profile/). "
    "Text generation powered by [Claude](https://www.anthropic.com/) (Anthropic). "
    "This tool provides approximate guidance — language complexity cannot be fully quantified."
)
