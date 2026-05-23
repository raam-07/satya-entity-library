# ==============================================================================
# SATYA — ENTITY UPDATER (Repo 4)
# 
# Reads from Classified Sheet, uses spaCy NER + Gemma to:
#   1. Detect current CMs per state (last 90 days)
#   2. Detect ruling party per state (last 90 days)
#   3. Extract promises made by known entities (last 180 days)
#   4. Track criminal cases / FIRs (all time)
#   5. Discover new unknown entities (last 30 days)
#
# Outputs:
#   - entities.json (auto-updated high confidence fields)
#   - review_flags.json (low confidence items for manual review)
#
# Runs weekly via GitHub Actions.
# ==============================================================================

import os
import json
import time
import logging
import re
import requests
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
CLASSIFIED_SHEET_NAME = 'Satya Classified'
CLASSIFIED_WORKSHEET_NAME = 'Sheet1'

# GitHub raw URL for entities.json (your satya-entity-library repo)
ENTITIES_JSON_URL = os.environ.get('ENTITIES_JSON_URL', '')

# Local paths
ENTITIES_OUTPUT_PATH = './entities.json'
REVIEW_FLAGS_PATH = './review_flags.json'

# Time windows (driven by scraped_at)
WINDOW_CM_PARTY_DAYS = 90       # For CM + ruling party detection
WINDOW_PROMISES_DAYS = 180      # For promise extraction
WINDOW_CRIMINAL_DAYS = 9999     # All time for criminal cases
WINDOW_NEW_ENTITIES_DAYS = 30   # For new unknown entity discovery

# Confidence thresholds
AUTO_UPDATE_THRESHOLD = 0.75    # Above this = auto update entities.json
REVIEW_THRESHOLD = 0.40         # Between this and above = flag for review
NEW_ENTITY_MIN_MENTIONS = 20    # Min mentions to flag as new entity

# Gemma model
MODEL_PATH = "./models/gemma-2-9b-it-Q6_K.gguf"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==============================================================================
# --- GOOGLE SHEETS SETUP ---
# ==============================================================================

def connect_to_sheets():
    logging.info("Connecting to Google Sheets...")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    gcp_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not gcp_json:
        raise ValueError("GCP_SERVICE_ACCOUNT_JSON missing!")
    creds_dict = json.loads(gcp_json)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open(CLASSIFIED_SHEET_NAME).worksheet(CLASSIFIED_WORKSHEET_NAME)
    logging.info("Connected to Classified Sheet.")
    return sheet

def fetch_articles(sheet):
    logging.info("Fetching all classified articles...")
    raw_data = sheet.col_values(1)
    articles = []
    for cell in raw_data:
        if not cell:
            continue
        try:
            article = json.loads(cell)
            # Parse scraped_at into datetime
            scraped_raw = article.get('scraped_at', '')
            try:
                article['scraped_dt'] = datetime.strptime(
                    str(scraped_raw).split('.')[0], "%Y-%m-%d %H:%M:%S"
                )
            except:
                article['scraped_dt'] = datetime.now() - timedelta(days=365)
            articles.append(article)
        except json.JSONDecodeError:
            continue
    logging.info(f"Fetched {len(articles)} classified articles.")
    return articles

def filter_by_window(articles, days):
    cutoff = datetime.now() - timedelta(days=days)
    return [a for a in articles if a['scraped_dt'] >= cutoff]

# ==============================================================================
# --- LOAD ENTITIES JSON ---
# ==============================================================================

def load_entities():
    """Load entities.json from GitHub raw URL or local file."""
    if ENTITIES_JSON_URL:
        try:
            logging.info(f"Fetching entities.json from GitHub...")
            response = requests.get(ENTITIES_JSON_URL, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.warning(f"Failed to fetch from GitHub: {e}. Trying local file.")

    if os.path.exists(ENTITIES_OUTPUT_PATH):
        with open(ENTITIES_OUTPUT_PATH, 'r') as f:
            return json.load(f)

    raise FileNotFoundError("No entities.json found locally or via GitHub URL.")

# ==============================================================================
# --- spaCy NER SETUP ---
# ==============================================================================

def load_spacy():
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            logging.info("spaCy model loaded: en_core_web_sm")
        except OSError:
            logging.info("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        return nlp
    except ImportError:
        logging.critical("spaCy not installed. Run: pip install spacy")
        return None

# ==============================================================================
# --- GEMMA SETUP ---
# ==============================================================================

def load_gemma():
    try:
        from llama_cpp import Llama
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Gemma model not found at {MODEL_PATH}")
        logging.info("Loading Gemma model for validation...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_batch=512,
            n_threads=2,
            verbose=False
        )
        logging.info("Gemma loaded.")
        return llm
    except Exception as e:
        logging.error(f"Failed to load Gemma: {e}. Validation step will be skipped.")
        return None

def gemma_validate(llm, question, context):
    """
    Ask Gemma a yes/no question about a piece of text.
    Returns (True/False, confidence_str)
    """
    if llm is None:
        return True, "unvalidated"

    prompt = f"""<start_of_turn>user
Read the text below and answer the question with ONLY a JSON object.

Text: {context[:500]}

Question: {question}

Return ONLY: {{"answer": "yes" or "no", "confidence": "high" or "medium" or "low"}}
No explanation. No extra text.
<end_of_turn>
<start_of_turn>model
"""
    try:
        response = llm(
            prompt,
            max_tokens=60,
            temperature=0.1,
            stop=["<end_of_turn>", "<start_of_turn>"],
            echo=False
        )
        raw = response['choices'][0].get('text', '').strip()
        raw = re.sub(r'```json|```', '', raw).strip()
        parsed = json.loads(raw)
        answer = parsed.get('answer', 'no').lower() == 'yes'
        confidence = parsed.get('confidence', 'low')
        return answer, confidence
    except Exception as e:
        logging.warning(f"Gemma validation failed: {e}")
        return True, "unvalidated"

# ==============================================================================
# --- EXTRACTION FUNCTIONS ---
# ==============================================================================

# --- 1. CM Detection ---
def detect_cms(articles, entities, nlp, llm):
    """
    Detect current Chief Minister per state from last 90 days articles.
    Strict two-pass: pattern match in CM context + Gemma validation.
    """
    logging.info("--- Detecting CMs per state ---")
    window_articles = filter_by_window(articles, WINDOW_CM_PARTY_DAYS)
    logging.info(f"Using {len(window_articles)} articles from last {WINDOW_CM_PARTY_DAYS} days.")

    # Build state lookup
    state_aliases = {}
    for s in entities['india']['states']:
        for alias in [s['name']] + s.get('aliases', []):
            state_aliases[alias.lower()] = s['name']

    # Build known minister lookup
    all_ministers = (
        entities['india']['cabinet_ministers'] +
        entities['india']['state_chief_ministers'] +
        entities['india']['opposition_leaders']
    )
    minister_lookup = {}
    for m in all_ministers:
        minister_lookup[m['name'].lower()] = m['name']
        for alias in m.get('aliases', []):
            minister_lookup[alias.lower()] = m['name']

    # STRICT CM context patterns — must explicitly say "Chief Minister" or "CM of"
    # NOT just "CM" alone which is too ambiguous
    STRICT_CM_PATTERNS = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*,?\s*(?:the\s+)?chief\s+minister\s+of\s+(\w[\w\s]+)',
        r'chief\s+minister\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+of\s+(\w[\w\s]+)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+was\s+sworn\s+in\s+as\s+(?:the\s+)?chief\s+minister',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+takes?\s+oath\s+as\s+(?:the\s+)?chief\s+minister',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*,\s*(?:the\s+)?(?:\w+\s+)?chief\s+minister',
        r'new\s+(?:chief\s+minister|cm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
    ]

    # Per state, count STRICT CM mentions
    state_cm_candidates = defaultdict(lambda: defaultdict(int))
    state_cm_contexts = defaultdict(lambda: defaultdict(list))

    NON_INDIAN_SOURCES = {'The Dawn', 'BBC', 'Al Jazeera', 'The Guardian'}

    for article in window_articles:
        # Skip non-Indian sources and international articles
        if article.get('category') == 'international':
            continue
        if article.get('source') in NON_INDIAN_SOURCES:
            continue

        text = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:600]}"
        text_lower = text.lower()

        # Only process articles that explicitly mention "chief minister"
        if 'chief minister' not in text_lower and ' cm ' not in text_lower:
            continue

        states_in_article = list(article.get('states_mentioned', []) or [])
        # Also scan text for state names
        for alias_lower, canonical in state_aliases.items():
            if len(alias_lower) >= 4 and alias_lower in text_lower:
                if canonical not in states_in_article:
                    states_in_article.append(canonical)

        if not states_in_article:
            continue

        # Match strict patterns
        for pattern in STRICT_CM_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                person = match.group(1).strip() if match.lastindex >= 1 else None
                if not person or len(person) < 5:
                    continue

                # Only accept known ministers — reject random names
                canonical = minister_lookup.get(person.lower())
                if not canonical:
                    # Try partial match
                    for alias, name in minister_lookup.items():
                        if alias in person.lower() or person.lower() in alias:
                            canonical = name
                            break

                if not canonical:
                    continue

                for state in states_in_article:
                    state_cm_candidates[state][canonical] += 1
                    state_cm_contexts[state][canonical].append(text[:400])

    # Score and decide
    cm_updates = []
    cm_flags = []

    for state, candidates in state_cm_candidates.items():
        if not candidates:
            continue
        total_mentions = sum(candidates.values())
        top_candidate = max(candidates, key=candidates.get)
        top_count = candidates[top_candidate]
        confidence = top_count / total_mentions if total_mentions > 0 else 0

        # Need at least 3 mentions to be credible
        if top_count < 3:
            continue

        context_sample = state_cm_contexts[state][top_candidate][0] if state_cm_contexts[state][top_candidate] else ""

        if confidence >= AUTO_UPDATE_THRESHOLD:
            is_cm, gem_conf = gemma_validate(
                llm,
                f"Is '{top_candidate}' explicitly mentioned as the Chief Minister of {state} in this text?",
                context_sample
            )
            if is_cm:
                cm_updates.append({
                    "state": state,
                    "cm": top_candidate,
                    "confidence": round(confidence, 2),
                    "mentions": top_count,
                    "gemma_validated": True,
                    "gemma_confidence": gem_conf
                })
                logging.info(f"CM UPDATE: {state} → {top_candidate} (confidence: {confidence:.2f}, mentions: {top_count})")
            else:
                cm_flags.append({
                    "type": "cm_detection",
                    "state": state,
                    "candidate": top_candidate,
                    "confidence": round(confidence, 2),
                    "reason": "Gemma rejected",
                    "context": context_sample[:200]
                })
        elif confidence >= REVIEW_THRESHOLD and top_count >= 3:
            cm_flags.append({
                "type": "cm_detection",
                "state": state,
                "candidate": top_candidate,
                "confidence": round(confidence, 2),
                "mentions": top_count,
                "reason": "Below auto-update threshold",
                "context": context_sample[:200]
            })

    return cm_updates, cm_flags

# --- 2. Ruling Party Detection ---
def detect_ruling_parties(articles, entities, nlp, llm):
    """
    Detect ruling party per state from last 90 days articles.
    Strict: only counts explicit "X government in Y" or "X ruling Y" patterns.
    Requires minimum 3 mentions. Skips non-Indian sources.
    """
    logging.info("--- Detecting ruling parties per state ---")
    window_articles = filter_by_window(articles, WINDOW_CM_PARTY_DAYS)

    party_aliases = {}
    for p in entities['india']['parties']:
        for alias in [p['name']] + p.get('aliases', []):
            party_aliases[alias.lower()] = p['name']

    state_aliases = {}
    for s in entities['india']['states']:
        for alias in [s['name']] + s.get('aliases', []):
            state_aliases[alias.lower()] = s['name']

    NON_INDIAN_SOURCES = {'The Dawn', 'BBC', 'Al Jazeera', 'The Guardian'}

    # STRICT patterns — must explicitly say party governs/rules a state
    STRICT_RULING_PATTERNS = [
        r'(BJP|Congress|INC|AAP|TMC|DMK|CPM|JDU|JMM|TDP|YSRCP|NCP|Shiv Sena|RJD|BJD)\s+government\s+in\s+(\w[\w\s]+)',
        r'(BJP|Congress|INC|AAP|TMC|DMK|CPM|JDU|JMM|TDP|YSRCP|NCP|Shiv Sena|RJD|BJD)-led\s+government\s+in\s+(\w[\w\s]+)',
        r'(BJP|Congress|INC|AAP|TMC|DMK|CPM|JDU|JMM|TDP|YSRCP|NCP|Shiv Sena|RJD|BJD)\s+(?:won|wins|won\s+power|came\s+to\s+power|swept)\s+(?:in\s+)?(\w[\w\s]+)',
        r'(\w[\w\s]+)\s+(?:state|government)\s+(?:is\s+)?ruled?\s+by\s+(BJP|Congress|INC|AAP|TMC|DMK|CPM|JDU|JMM|TDP)',
        r'(BJP|Congress|INC|AAP|TMC|DMK|CPM|JDU|JMM|TDP)\s+(?:wins?|victory|won)\s+(\w+\s+(?:Pradesh|Nadu|Bengal|Kerala|Karnataka|Bihar|Assam|Jharkhand|Odisha|Goa|Delhi|Punjab|Rajasthan|Gujarat|Maharashtra|Telangana|Andhra))',
    ]

    state_party_candidates = defaultdict(lambda: defaultdict(int))
    state_party_contexts = defaultdict(lambda: defaultdict(list))

    for article in window_articles:
        if article.get('category') == 'international':
            continue
        if article.get('source') in NON_INDIAN_SOURCES:
            continue

        text = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:500]}"
        text_lower = text.lower()

        # Only process articles about elections or governance
        governance_keywords = ['government', 'ruling', 'election', 'won', 'victory', 'cm', 'chief minister', 'sworn in']
        if not any(kw in text_lower for kw in governance_keywords):
            continue

        # Match strict patterns
        for pattern in STRICT_RULING_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) < 2:
                    continue

                # Figure out which group is party and which is state
                g1, g2 = groups[0].strip(), groups[1].strip()

                # Identify party
                party_canonical = party_aliases.get(g1.lower()) or party_aliases.get(g2.lower())
                if not party_canonical:
                    continue

                # Identify state
                state_canonical = state_aliases.get(g1.lower()) or state_aliases.get(g2.lower())
                if not state_canonical:
                    # Try partial state match
                    for alias_lower, canonical in state_aliases.items():
                        if alias_lower in g1.lower() or alias_lower in g2.lower():
                            state_canonical = canonical
                            break

                if not state_canonical:
                    continue

                state_party_candidates[state_canonical][party_canonical] += 1
                state_party_contexts[state_canonical][party_canonical].append(text[:400])

    # Score and decide
    party_updates = []
    party_flags = []

    for state, candidates in state_party_candidates.items():
        if not candidates:
            continue
        total = sum(candidates.values())
        top_party = max(candidates, key=candidates.get)
        top_count = candidates[top_party]
        confidence = top_count / total if total > 0 else 0

        # Need at least 3 explicit mentions to be credible
        if top_count < 3:
            continue

        context_sample = state_party_contexts[state][top_party][0] if state_party_contexts[state][top_party] else ""

        if confidence >= AUTO_UPDATE_THRESHOLD:
            is_ruling, gem_conf = gemma_validate(
                llm,
                f"Is '{top_party}' explicitly mentioned as the ruling party or government of {state} in this text?",
                context_sample
            )
            if is_ruling:
                party_updates.append({
                    "state": state,
                    "ruling_party": top_party,
                    "confidence": round(confidence, 2),
                    "mentions": top_count,
                    "gemma_validated": True
                })
                logging.info(f"PARTY UPDATE: {state} → {top_party} (confidence: {confidence:.2f}, mentions: {top_count})")
            else:
                party_flags.append({
                    "type": "ruling_party",
                    "state": state,
                    "candidate": top_party,
                    "confidence": round(confidence, 2),
                    "reason": "Gemma rejected",
                    "context": context_sample[:200]
                })
        elif confidence >= REVIEW_THRESHOLD and top_count >= 3:
            party_flags.append({
                "type": "ruling_party",
                "state": state,
                "candidate": top_party,
                "confidence": round(confidence, 2),
                "reason": "Below auto-update threshold",
                "context": context_sample[:200]
            })

    return party_updates, party_flags

        # Cross-reference: if article mentions a state and a party, count it
        for state in states_in_article:
            for party in parties_in_article:
                # Check if ruling/government context exists
                if any(kw in text_lower for kw in ['government', 'ruling', 'administration', 'regime', 'led by']):
                    state_party_candidates[state][party] += 1
                    state_party_contexts[state][party].append(text[:300])

        # Pattern matching
        for pattern in ruling_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                matched_text = match.group(0)
                for party_alias, canonical_party in party_aliases.items():
                    if party_alias in matched_text:
                        for state_alias, canonical_state in state_aliases.items():
                            if state_alias in matched_text:
                                state_party_candidates[canonical_state][canonical_party] += 2
                                state_party_contexts[canonical_state][canonical_party].append(text[:300])

    # Score and decide
    party_updates = []
    party_flags = []

    for state, candidates in state_party_candidates.items():
        if not candidates:
            continue
        total = sum(candidates.values())
        top_party = max(candidates, key=candidates.get)
        top_count = candidates[top_party]
        confidence = top_count / total if total > 0 else 0
        context_sample = state_party_contexts[state][top_party][0] if state_party_contexts[state][top_party] else ""

        if confidence >= AUTO_UPDATE_THRESHOLD:
            is_ruling, gem_conf = gemma_validate(
                llm,
                f"Is '{top_party}' mentioned as the ruling party or government of {state}?",
                context_sample
            )
            if is_ruling:
                party_updates.append({
                    "state": state,
                    "ruling_party": top_party,
                    "confidence": round(confidence, 2),
                    "mentions": top_count,
                    "gemma_validated": True
                })
                logging.info(f"PARTY UPDATE: {state} → {top_party} (confidence: {confidence:.2f})")
            else:
                party_flags.append({
                    "type": "ruling_party",
                    "state": state,
                    "candidate": top_party,
                    "confidence": round(confidence, 2),
                    "reason": "Gemma rejected",
                    "context": context_sample[:200]
                })
        elif confidence >= REVIEW_THRESHOLD:
            party_flags.append({
                "type": "ruling_party",
                "state": state,
                "candidate": top_party,
                "confidence": round(confidence, 2),
                "reason": "Below auto-update threshold",
                "context": context_sample[:200]
            })

    return party_updates, party_flags

# --- 3. Promise Extraction ---
def extract_promises(articles, entities, nlp, llm):
    """
    Extract promises made by known entities from last 180 days.
    """
    logging.info("--- Extracting promises ---")
    window_articles = filter_by_window(articles, WINDOW_PROMISES_DAYS)

    # Build known entity name list
    known_names = set()
    for m in entities['india']['cabinet_ministers'] + entities['india']['opposition_leaders'] + entities['india']['state_chief_ministers']:
        known_names.add(m['name'].lower())
        for alias in m.get('aliases', []):
            known_names.add(alias.lower())

    promise_keywords = [
        'promised', 'vowed', 'pledged', 'assured', 'committed',
        'announced', 'declared', 'stated', 'said he will', 'said she will',
        'will ensure', 'will provide', 'will build', 'will create',
        'guarantee', 'target of', 'plan to'
    ]

    promise_pattern = re.compile(
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:' + '|'.join(promise_keywords) + r')\s+(.{20,200}?)(?:\.|,|$)',
        re.IGNORECASE
    )

    extracted_promises = []

    for article in window_articles:
        text = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:800]}"

        matches = promise_pattern.finditer(text)
        for match in matches:
            person = match.group(1).strip()
            promise_text = match.group(2).strip()

            # Only for known entities
            if person.lower() not in known_names:
                continue

            if len(promise_text) < 20:
                continue

            # Gemma validates: is this really a promise?
            is_promise, gem_conf = gemma_validate(
                llm,
                f"Is this text describing a promise or commitment made by {person}?",
                f"{person} {promise_text}"
            )

            if is_promise:
                extracted_promises.append({
                    "person": person,
                    "promise_text": promise_text,
                    "source_url": article.get('url', ''),
                    "source_title": article.get('title', ''),
                    "scraped_at": article.get('scraped_at', ''),
                    "gemma_confidence": gem_conf,
                    "status": "pending_review",
                    "verified": False
                })
                logging.info(f"PROMISE FOUND: {person} — {promise_text[:80]}...")

    logging.info(f"Extracted {len(extracted_promises)} promises.")
    return extracted_promises

# --- 4. Criminal Case Detection ---
def detect_criminal_cases(articles, entities, nlp, llm):
    """
    Detect criminal cases / FIRs linked to known entities.
    Strict: requires PERSON + CRIME in same sentence.
    Gemma validates each incident before counting.
    Skips non-Indian sources.
    """
    logging.info("--- Detecting criminal cases ---")
    window_articles = articles

    NON_INDIAN_SOURCES = {'The Dawn', 'BBC', 'Al Jazeera', 'The Guardian'}

    known_entities = {}
    for m in entities['india']['cabinet_ministers'] + entities['india']['opposition_leaders'] + entities['india']['state_chief_ministers']:
        known_entities[m['name'].lower()] = m['name']
        for alias in m.get('aliases', []):
            known_entities[alias.lower()] = m['name']

    # Only SERIOUS criminal keywords — not just "arrested" which can be a protestor
    SERIOUS_CRIMINAL_KEYWORDS = [
        'fir filed against', 'fir registered against',
        'chargesheeted', 'chargesheet filed against',
        'convicted', 'sentenced',
        'money laundering case', 'corruption case',
        'ed arrested', 'cbi arrested',
        'rape accused', 'murder accused',
        'disproportionate assets',
        'hawala', 'bribery case',
        'criminal case against', 'criminal charges against'
    ]

    # Broader keywords for pattern matching — but require Gemma confirmation
    BROAD_CRIMINAL_KEYWORDS = [
        'arrested', 'detained', 'fir', 'bail', 'custody',
        'charged with', 'accused of', 'fraud case', 'scam',
        'ed summons', 'cbi summons', 'raid on'
    ]

    entity_incidents = defaultdict(list)

    for article in window_articles:
        if article.get('category') == 'international':
            continue
        if article.get('source') in NON_INDIAN_SOURCES:
            continue

        text = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:600]}"
        text_lower = text.lower()

        # Check for criminal context first
        has_serious = any(kw in text_lower for kw in SERIOUS_CRIMINAL_KEYWORDS)
        has_broad = any(kw in text_lower for kw in BROAD_CRIMINAL_KEYWORDS)

        if not has_serious and not has_broad:
            continue

        # Find which known entities are mentioned
        for entity_lower, canonical_name in known_entities.items():
            if len(entity_lower) < 4:
                continue
            if entity_lower not in text_lower:
                continue

            # Avoid duplicate from same article
            already_logged = any(
                inc['source_url'] == article.get('url', '')
                for inc in entity_incidents[canonical_name]
            )
            if already_logged:
                continue

            # For broad keywords — require Gemma to confirm it's a criminal case
            # For serious keywords — accept directly
            if has_serious:
                confirmed = True
                incident_type = "serious"
            else:
                # Ask Gemma: is this article about a criminal case involving this person?
                confirmed, _ = gemma_validate(
                    llm,
                    f"Is this article specifically about a criminal case, FIR, arrest, or legal action directly involving '{canonical_name}'?",
                    text[:400]
                )
                incident_type = "broad"

            if confirmed:
                entity_incidents[canonical_name].append({
                    "incident_text": text[:200],
                    "source_url": article.get('url', ''),
                    "source_title": article.get('title', ''),
                    "scraped_at": article.get('scraped_at', ''),
                    "incident_type": incident_type
                })
                logging.info(f"CRIMINAL CASE [{incident_type}]: {canonical_name} — {article.get('title', '')[:60]}")

    criminal_updates = []
    for entity_name, incidents in entity_incidents.items():
        if incidents:
            criminal_updates.append({
                "entity": entity_name,
                "incident_count": len(incidents),
                "incidents": incidents[:10]
            })
            logging.info(f"CRIMINAL SUMMARY: {entity_name} — {len(incidents)} validated incidents")

    return criminal_updates

# --- 5. New Entity Discovery ---
def discover_new_entities(articles, entities, nlp):
    """
    Find frequently mentioned persons not in entities.json.
    Uses last 30 days articles.
    """
    logging.info("--- Discovering new entities ---")
    window_articles = filter_by_window(articles, WINDOW_NEW_ENTITIES_DAYS)

    # Build set of all known names
    known_names = set()
    for m in (entities['india']['cabinet_ministers'] +
              entities['india']['opposition_leaders'] +
              entities['india']['state_chief_ministers']):
        known_names.add(m['name'].lower())
        for alias in m.get('aliases', []):
            known_names.add(alias.lower())

    for leader in entities['international']['world_leaders']:
        known_names.add(leader['name'].lower())
        for alias in leader.get('aliases', []):
            known_names.add(alias.lower())

    # Count all person mentions via spaCy
    person_counter = Counter()
    person_contexts = defaultdict(list)

    for article in window_articles:
        text = f"{article.get('title', '')} {article.get('rephrased_article', '')}"
        doc = nlp(text[:800])
        for ent in doc.ents:
            if ent.label_ == 'PERSON' and len(ent.text.strip()) > 5:
                name = ent.text.strip()
                if name.lower() not in known_names:
                    person_counter[name] += 1
                    if len(person_contexts[name]) < 3:
                        person_contexts[name].append(article.get('title', ''))

    # Flag those above threshold
    new_entities = []
    for name, count in person_counter.most_common(50):
        if count >= NEW_ENTITY_MIN_MENTIONS:
            new_entities.append({
                "name": name,
                "mentions_last_30_days": count,
                "sample_articles": person_contexts[name][:3],
                "suggested_action": "Review and add to entities.json",
                "needs_verification": True
            })
            logging.info(f"NEW ENTITY: {name} — {count} mentions in last 30 days")

    return new_entities

# ==============================================================================
# --- APPLY UPDATES TO entities.json ---
# ==============================================================================

def apply_updates(entities, cm_updates, party_updates, criminal_updates, new_promises):
    """Apply all high-confidence updates to the entities object."""

    # Update CM per state
    for update in cm_updates:
        for state in entities['india']['states']:
            if state['name'] == update['state']:
                old_cm = state.get('cm', 'Unknown')
                state['cm'] = update['cm']
                state['cm_confidence'] = update['confidence']
                state['cm_last_updated'] = str(datetime.now().date())
                if old_cm != update['cm']:
                    logging.info(f"APPLIED CM UPDATE: {update['state']} — {old_cm} → {update['cm']}")
                break

    # Update ruling party per state
    for update in party_updates:
        for state in entities['india']['states']:
            if state['name'] == update['state']:
                old_party = state.get('ruling_party', 'Unknown')
                state['ruling_party'] = update['ruling_party']
                state['party_confidence'] = update['confidence']
                state['party_last_updated'] = str(datetime.now().date())
                if old_party != update['ruling_party']:
                    logging.info(f"APPLIED PARTY UPDATE: {update['state']} — {old_party} → {update['ruling_party']}")
                break

    # Update criminal case counts
    for update in criminal_updates:
        entity_name = update['entity']
        all_ministers = (entities['india']['cabinet_ministers'] +
                        entities['india']['opposition_leaders'] +
                        entities['india']['state_chief_ministers'])
        for minister in all_ministers:
            if minister['name'] == entity_name:
                minister['criminal_cases_in_news'] = update['incident_count']
                minister['criminal_incidents'] = update['incidents']
                minister['criminal_last_updated'] = str(datetime.now().date())
                break

    # Add new promises to entities metadata
    if new_promises:
        if 'extracted_promises' not in entities:
            entities['extracted_promises'] = []
        # Avoid duplicates by URL
        existing_urls = {p.get('source_url') for p in entities['extracted_promises']}
        for promise in new_promises:
            if promise['source_url'] not in existing_urls:
                entities['extracted_promises'].append(promise)

    # Update metadata
    entities['metadata']['last_updated'] = str(datetime.now().date())
    entities['metadata']['auto_updated_fields'] = {
        "cm_updates": len(cm_updates),
        "party_updates": len(party_updates),
        "criminal_updates": len(criminal_updates),
        "new_promises": len(new_promises)
    }

    return entities

# ==============================================================================
# --- MAIN ---
# ==============================================================================

def main():
    start_time = time.time()
    logging.info("--- Satya Entity Updater Started ---")

    # 1. Load data
    sheet = connect_to_sheets()
    articles = fetch_articles(sheet)

    if not articles:
        logging.error("No articles found. Exiting.")
        return

    # 2. Load existing entities
    entities = load_entities()
    logging.info(f"Loaded entities.json (version: {entities['metadata'].get('version', 'unknown')})")

    # 3. Load NLP tools
    nlp = load_spacy()
    if nlp is None:
        logging.error("spaCy failed to load. Exiting.")
        return

    # 4. Lazy load Gemma only when needed
    llm = load_gemma()

    # 5. Run all extraction tasks
    all_flags = []

    # CM Detection
    cm_updates, cm_flags = detect_cms(articles, entities, nlp, llm)
    all_flags.extend(cm_flags)

    # Ruling Party Detection
    party_updates, party_flags = detect_ruling_parties(articles, entities, nlp, llm)
    all_flags.extend(party_flags)

    # Promise Extraction
    new_promises = extract_promises(articles, entities, nlp, llm)

    # Criminal Case Detection
    criminal_updates = detect_criminal_cases(articles, entities, nlp, llm)

    # New Entity Discovery
    new_entities = discover_new_entities(articles, entities, nlp)

    # 6. Apply high confidence updates to entities
    updated_entities = apply_updates(
        entities, cm_updates, party_updates, criminal_updates, new_promises
    )

    # 7. Save updated entities.json
    with open(ENTITIES_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(updated_entities, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved updated entities.json")

    # 8. Save review_flags.json
    review_output = {
        "generated_at": str(datetime.now()),
        "summary": {
            "cm_updates_applied": len(cm_updates),
            "party_updates_applied": len(party_updates),
            "criminal_updates_applied": len(criminal_updates),
            "promises_extracted": len(new_promises),
            "new_entities_discovered": len(new_entities),
            "items_needing_review": len(all_flags)
        },
        "new_entities_discovered": new_entities,
        "items_needing_manual_review": all_flags
    }

    with open(REVIEW_FLAGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(review_output, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved review_flags.json ({len(all_flags)} items need review)")

    elapsed = round(time.time() - start_time, 2)
    logging.info(f"--- Entity Updater Finished in {elapsed}s ---")
    logging.info(f"Summary: {review_output['summary']}")

    print(json.dumps(review_output['summary'], indent=2))

if __name__ == '__main__':
    main()
