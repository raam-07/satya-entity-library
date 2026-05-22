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
MODEL_PATH = "./models/gemma-2-2b-it-Q6_K_L.gguf"

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
            n_ctx=2048,
            n_batch=256,
            n_threads=4,
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
    Uses spaCy NER + pattern matching + confidence scoring.
    """
    logging.info("--- Detecting CMs per state ---")
    window_articles = filter_by_window(articles, WINDOW_CM_PARTY_DAYS)
    logging.info(f"Using {len(window_articles)} articles from last {WINDOW_CM_PARTY_DAYS} days.")

    # Build state list from entities
    state_names = [s['name'] for s in entities['india']['states']]
    state_aliases = {}
    for s in entities['india']['states']:
        for alias in s.get('aliases', []):
            state_aliases[alias.lower()] = s['name']
        state_aliases[s['name'].lower()] = s['name']

    # CM patterns to scan for
    cm_patterns = [
        r'(\w[\w\s]+?)\s*,?\s*(?:the\s+)?(?:chief\s+minister|cm)\s+of\s+([\w\s]+)',
        r'(?:chief\s+minister|cm)\s+([\w\s]+?)\s+of\s+([\w\s]+)',
        r'([\w\s]+?)\s*,\s*(?:the\s+)?(?:chief\s+minister|cm)',
        r'(?:chief\s+minister|cm)\s+([\w\s]+)',
    ]

    # Per state, count person mentions in CM context
    state_cm_candidates = defaultdict(lambda: defaultdict(int))
    state_cm_contexts = defaultdict(lambda: defaultdict(list))

    for article in window_articles:
        # Skip international articles for Indian entity extraction
        if article.get('category') == 'international':
            continue

        text = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:500]}"
        text_lower = text.lower()
        states_in_article = article.get('states_mentioned', [])

        # Also detect states via text scan
        for state_name_lower, canonical_state in state_aliases.items():
            if state_name_lower in text_lower and canonical_state not in states_in_article:
                states_in_article.append(canonical_state)

        if not states_in_article:
            continue

        # spaCy NER to find persons
        doc = nlp(text[:1000])
        persons_in_article = [ent.text.strip() for ent in doc.ents if ent.label_ == 'PERSON' and len(ent.text.strip()) > 4]

        # Pattern matching for CM context
        for pattern in cm_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                groups = match.groups()
                for state in states_in_article:
                    for person in persons_in_article:
                        if person.lower() in text_lower:
                            state_cm_candidates[state][person] += 1
                            state_cm_contexts[state][person].append(text[:300])

        # Also check if known ministers in entities appear as CM
        for minister in entities['india']['cabinet_ministers'] + entities['india']['state_chief_ministers'] + entities['india']['opposition_leaders']:
            name = minister['name']
            for alias in [name] + minister.get('aliases', []):
                if alias.lower() in text_lower and 'chief minister' in text_lower or 'cm' in text_lower:
                    for state in states_in_article:
                        state_cm_candidates[state][name] += 2  # known entity gets boost
                        state_cm_contexts[state][name].append(text[:300])

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

        context_sample = state_cm_contexts[state][top_candidate][0] if state_cm_contexts[state][top_candidate] else ""

        # Gemma validation for high confidence candidates
        if confidence >= AUTO_UPDATE_THRESHOLD:
            is_cm, gem_conf = gemma_validate(
                llm,
                f"Is '{top_candidate}' mentioned as the Chief Minister in this text?",
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
                logging.info(f"CM UPDATE: {state} → {top_candidate} (confidence: {confidence:.2f})")
            else:
                cm_flags.append({
                    "type": "cm_detection",
                    "state": state,
                    "candidate": top_candidate,
                    "confidence": round(confidence, 2),
                    "reason": "Gemma rejected",
                    "context": context_sample[:200]
                })
        elif confidence >= REVIEW_THRESHOLD:
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
    """
    logging.info("--- Detecting ruling parties per state ---")
    window_articles = filter_by_window(articles, WINDOW_CM_PARTY_DAYS)

    party_names = [p['name'] for p in entities['india']['parties']]
    party_aliases = {}
    for p in entities['india']['parties']:
        for alias in [p['name']] + p.get('aliases', []):
            party_aliases[alias.lower()] = p['name']

    state_aliases = {}
    for s in entities['india']['states']:
        for alias in [s['name']] + s.get('aliases', []):
            state_aliases[alias.lower()] = s['name']

    ruling_patterns = [
        r'([\w\s]+?)\s+government\s+in\s+([\w\s]+)',
        r'([\w\s]+?)\s+ruled?\s+([\w\s]+)',
        r'([\w\s]+?)\s+ruling\s+([\w\s]+)',
        r'([\w\s]+?)-led\s+government\s+in\s+([\w\s]+)',
        r'([\w\s]+?)\s+administration\s+in\s+([\w\s]+)',
    ]

    state_party_candidates = defaultdict(lambda: defaultdict(int))
    state_party_contexts = defaultdict(lambda: defaultdict(list))

    for article in window_articles:
        text = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:500]}"
        text_lower = text.lower()

        states_in_article = article.get('states_mentioned', [])
        parties_in_article = article.get('party_mentioned', [])

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
    Detect criminal cases / FIRs linked to known entities. All time.
    """
    logging.info("--- Detecting criminal cases ---")
    # Use all articles
    window_articles = articles

    known_entities = {}
    for m in entities['india']['cabinet_ministers'] + entities['india']['opposition_leaders'] + entities['india']['state_chief_ministers']:
        known_entities[m['name'].lower()] = m['name']
        for alias in m.get('aliases', []):
            known_entities[alias.lower()] = m['name']

    criminal_keywords = [
        'arrested', 'fir', 'chargesheet', 'convicted', 'bail',
        'custody', 'detained', 'charged with', 'accused of',
        'money laundering', 'corruption case', 'scam', 'fraud case',
        'ed summons', 'cbi summons', 'raid on', 'bribery'
    ]

    criminal_pattern = re.compile(
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:was\s+|has\s+been\s+|is\s+)?(?:' +
        '|'.join(criminal_keywords) + r')',
        re.IGNORECASE
    )

    # Count unique incidents per entity
    entity_incidents = defaultdict(list)

    for article in window_articles:
        text = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:600]}"

        matches = criminal_pattern.finditer(text)
        for match in matches:
            person = match.group(1).strip()
            if person.lower() in known_entities:
                canonical_name = known_entities[person.lower()]
                # Avoid duplicate incidents from same article
                already_logged = any(
                    inc['source_url'] == article.get('url', '')
                    for inc in entity_incidents[canonical_name]
                )
                if not already_logged:
                    entity_incidents[canonical_name].append({
                        "incident_text": match.group(0)[:200],
                        "source_url": article.get('url', ''),
                        "source_title": article.get('title', ''),
                        "scraped_at": article.get('scraped_at', '')
                    })

    criminal_updates = []
    for entity_name, incidents in entity_incidents.items():
        if incidents:
            criminal_updates.append({
                "entity": entity_name,
                "incident_count": len(incidents),
                "incidents": incidents[:10]  # Cap at 10 for JSON size
            })
            logging.info(f"CRIMINAL CASES: {entity_name} — {len(incidents)} incidents found in news")

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
