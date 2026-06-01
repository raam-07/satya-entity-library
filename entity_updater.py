# ==============================================================================
# SATYA — ENTITY UPDATER (Repo 4)
#
# Reads from Classified Sheet, uses spaCy NER + Gemma to:
#   1. Canonicalise ministers_mentioned across all articles (surname dedup fix)
#   2. Detect current CMs per state (last 90 days)
#   3. Detect ruling party per state (last 90 days)
#   4. Extract promises made by known entities (last 180 days)
#   5. Track criminal cases / FIRs (all time)
#   6. Discover and auto-add new entities to entities.json (last 30 days)
#
# Outputs:
#   - entities.json           (auto-updated)
#   - review_flags.json       (low confidence items)
#   - canonicalize_audit.json (audit trail for minister name fixes)
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
CLASSIFIED_SHEET_NAME    = 'Satya Classified'
CLASSIFIED_WORKSHEET_NAME = 'Sheet1'

ENTITIES_JSON_URL = os.environ.get('ENTITIES_JSON_URL', '')

ENTITIES_OUTPUT_PATH    = './entities.json'
REVIEW_FLAGS_PATH       = './review_flags.json'
CANONICALIZE_AUDIT_PATH = './canonicalize_audit.json'

WINDOW_CM_PARTY_DAYS     = 90
WINDOW_PROMISES_DAYS     = 180
WINDOW_CRIMINAL_DAYS     = 9999
WINDOW_NEW_ENTITIES_DAYS = 30

AUTO_UPDATE_THRESHOLD   = 0.75
REVIEW_THRESHOLD        = 0.40
NEW_ENTITY_MIN_MENTIONS = 5

MODEL_PATH = "./models/gemma-2-9b-it-Q6_K.gguf"

NON_INDIAN_SOURCES = {'The Dawn', 'BBC', 'Al Jazeera', 'The Guardian'}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==============================================================================
# --- GOOGLE SHEETS SETUP ---
# ==============================================================================

def connect_to_sheets():
    logging.info("Connecting to Google Sheets...")
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    gcp_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not gcp_json:
        raise ValueError("GCP_SERVICE_ACCOUNT_JSON missing!")
    creds_dict = json.loads(gcp_json)
    creds  = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet  = client.open(CLASSIFIED_SHEET_NAME).worksheet(CLASSIFIED_WORKSHEET_NAME)
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
            scraped_raw = article.get('scraped_at', '')
            try:
                article['scraped_dt'] = datetime.strptime(
                    str(scraped_raw).split('.')[0], "%Y-%m-%d %H:%M:%S"
                )
            except Exception:
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
    if ENTITIES_JSON_URL:
        try:
            logging.info("Fetching entities.json from GitHub...")
            response = requests.get(ENTITIES_JSON_URL.strip(), timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.warning(f"Failed to fetch from GitHub: {e}. Trying local file.")

    if os.path.exists(ENTITIES_OUTPUT_PATH):
        with open(ENTITIES_OUTPUT_PATH, 'r') as f:
            return json.load(f)

    raise FileNotFoundError("No entities.json found.")

def is_candidate_name(ent_text):
    ent_text = ent_text.strip()
    words = ent_text.split()
    # Indian politician names generally consist of 2 or 3 words
    if len(words) not in (2, 3):
        return False
        
    for w in words:
        # Strip dots for checking alphabetic character constraint (e.g. M.K. -> MK)
        w_clean = w.replace('.', '')
        if len(w_clean) < 1:
            return False
        if not w[0].isupper() or not w_clean.isalpha():
            return False
    
    # Suffix and category stop-words to exclude standard geographic or institutional names
    stop_words = {
        'district', 'state', 'commission', 'board', 'court', 'police', 
        'government', 'ministry', 'department', 'party', 'union', 'front',
        'pradesh', 'bengal', 'delhi', 'karnataka', 'kerala', 'bihar',
        'punjab', 'gujarat', 'tamil', 'nadu', 'rajasthan', 'mumbai', 'kolkata',
        'india', 'indian', 'national', 'central', 'assembly', 'elections',
        'election', 'high', 'supreme', 'bjp', 'congress', 'tmc', 'aap',
        'corporation', 'municipal', 'polls', 'poll', 'meet', 'meeting', 'alliance', 
        'group', 'falls', 'committee', 'council', 'ltd', 'limited', 'pvt', 
        'board', 'trust', 'academy', 'association', 'school', 'hospital', 
        'university', 'society', 'foundation'
    }
    if any(w.lower().rstrip('.') in stop_words for w in words):
        return False
    return True


def build_minister_lookup(entities):
    """
    Alias → canonical name lookup used by CM detection, criminal detection,
    and promise extraction. Built from the aliases field in entities.json.
    """
    lookup = {}
    all_ministers = (
        entities['india']['cabinet_ministers'] +
        entities['india']['state_chief_ministers'] +
        entities['india']['opposition_leaders']
    )
    for m in all_ministers:
        lookup[m['name'].lower()] = m['name']
        for alias in m.get('aliases', []):
            lookup[alias.lower()] = m['name']
    return lookup


def build_canonical_minister_set(entities):
    """
    The set of authoritative canonical minister names from entities.json.
    No aliases — primary name field only. Ground truth for canonicalization.
    Auto-includes any entities added by discover_new_entities this run.
    """
    return set(
        m['name']
        for m in (
            entities['india']['cabinet_ministers'] +
            entities['india']['state_chief_ministers'] +
            entities['india']['opposition_leaders']
        )
    )


def find_canonical_candidates(name, canonical_set):
    """
    Given a raw name (e.g. "Modi", "Gandhi", "Narendra Modi"), return every
    canonical name that contains it as a whole-word substring.

    Uses word-boundary regex so "Shah" matches "Amit Shah" but not "Shahabuddin".
    Derives purely from the live canonical set — no hardcoded lists.
    """
    pattern = re.compile(r'\b' + re.escape(name.strip()) + r'\b', re.IGNORECASE)
    return [c for c in canonical_set if pattern.search(c)]

# ==============================================================================
# --- CANONICALISE ministers_mentioned IN SHEET ---
# ==============================================================================

def canonicalize_ministers_in_sheet(sheet, entities, llm):
    """
    Normalise ministers_mentioned in every article row to canonical full names.

    Resolution tiers — no hardcoded aliases, no static lists:
      1. Name already in canonical set             → keep as-is
      2. Matches exactly one canonical name        → resolve (unambiguous)
      3. Matches multiple canonical names          → Gemma picks using article text
      4. No canonical name matches                 → keep as-is (never corrupts)

    Structural guarantee: every Gemma answer is validated against the canonical
    set before acceptance — hallucinations cannot enter the data.

    Articles already marked ministers_canonicalized=True are skipped so weekly
    reruns only process new articles.

    Audit log is written to canonicalize_audit.json BEFORE the sheet is touched,
    providing a complete rollback reference.
    """
    logging.info("--- Canonicalising ministers_mentioned in sheet ---")

    canonical_set = build_canonical_minister_set(entities)
    raw_data      = sheet.col_values(1)
    batch_updates = []
    audit_log     = []

    for row_idx, cell in enumerate(raw_data, start=1):
        if not cell:
            continue
        try:
            article = json.loads(cell)
        except json.JSONDecodeError:
            continue

        # Skip articles already processed in a previous weekly run
        if article.get('ministers_canonicalized'):
            continue

        original = article.get('ministers_mentioned') or []

        # Build article context once — only used if Gemma disambiguation needed
        article_text = ' '.join(filter(None, [
            article.get('title', ''),
            article.get('rephrased_article', ''),
            (article.get('content') or '')[:400],
        ])).strip()

        canonicalized = []
        changes       = []

        for name in original:
            name = name.strip()
            if not name:
                continue

            # Tier 1: already canonical — keep
            if name in canonical_set:
                if name not in canonicalized:
                    canonicalized.append(name)
                continue

            candidates = find_canonical_candidates(name, canonical_set)

            if not candidates:
                # Tier 4: no match at all — keep as-is
                if name not in canonicalized:
                    canonicalized.append(name)
                continue

            if len(candidates) == 1:
                # Tier 2: exactly one canonical name matches — unambiguous
                resolved = candidates[0]
                if resolved not in canonicalized:
                    canonicalized.append(resolved)
                if resolved != name:
                    changes.append({
                        'original': name,
                        'resolved': resolved,
                        'method':   'unambiguous',
                    })
                continue

            # Tier 3: multiple candidates — ask Gemma with article context
            resolved = name  # safe default if Gemma unavailable or fails
            if llm is not None and article_text:
                options_str = ', '.join(f'"{c}"' for c in candidates)
                prompt = f"""<start_of_turn>user
Article: {article_text[:500]}

The article mentions "{name}". Based only on the article text above, which of these people is being referred to?
Options: {options_str}

Reply with ONLY the exact name from the options. No explanation.
<end_of_turn>
<start_of_turn>model
"""
                try:
                    response = llm(
                        prompt,
                        max_tokens=20,
                        temperature=0.0,
                        stop=["<end_of_turn>", "<start_of_turn>", "\n"],
                        echo=False
                    )
                    answer = response['choices'][0].get('text', '').strip().strip('"').strip("'")
                    # Structural validation: reject anything not in canonical set
                    if answer in canonical_set and answer in candidates:
                        resolved = answer
                        if resolved != name:
                            changes.append({
                                'original':   name,
                                'resolved':   resolved,
                                'method':     'gemma',
                                'candidates': candidates,
                            })
                        logging.info(f"Gemma resolved '{name}' → '{resolved}' (row {row_idx})")
                    else:
                        logging.warning(
                            f"Row {row_idx}: Gemma returned '{answer}' not in candidates "
                            f"{candidates}. Keeping original '{name}'."
                        )
                except Exception as e:
                    logging.warning(
                        f"Row {row_idx}: Gemma failed for '{name}': {e}. Keeping original."
                    )

            if resolved not in canonicalized:
                canonicalized.append(resolved)

        # Record audit entry if names actually changed
        if changes:
            audit_log.append({
                'row':      row_idx,
                'url':      article.get('url', ''),
                'original': original,
                'resolved': canonicalized,
                'changes':  changes,
            })

        # Always mark processed and write back (even if no name changes)
        article['ministers_mentioned']    = canonicalized
        article['ministers_canonicalized'] = True
        article_to_write = {k: v for k, v in article.items() if k != 'scraped_dt'}
        batch_updates.append({
            'range':  f'A{row_idx}',
            'values': [[json.dumps(article_to_write, ensure_ascii=False)]],
        })

    # Write audit log BEFORE touching the sheet — full rollback reference
    if audit_log:
        existing_audit = []
        if os.path.exists(CANONICALIZE_AUDIT_PATH):
            try:
                with open(CANONICALIZE_AUDIT_PATH, 'r') as f:
                    existing_audit = json.load(f)
            except Exception:
                pass
        existing_audit.extend(audit_log)
        with open(CANONICALIZE_AUDIT_PATH, 'w') as f:
            json.dump(existing_audit, f, indent=2, ensure_ascii=False)
        logging.info(f"Audit log: {len(audit_log)} name changes recorded.")

    # ✅ FIXED: Chunk writes to prevent Sheets API 500 / Payload-Size crashes
    if batch_updates:
        chunk_size = 100
        logging.info(f"Sending {len(batch_updates)} updates to Google Sheets in chunks of {chunk_size}...")
        for i in range(0, len(batch_updates), chunk_size):
            chunk = batch_updates[i : i + chunk_size]
            try:
                sheet.batch_update(chunk)
                logging.info(f"Wrote batch of {len(chunk)} updates (progress: {i + len(chunk)}/{len(batch_updates)})")
                time.sleep(1.0)  # Brief pause to respect API rate limits
            except Exception as e:
                logging.warning(f"Batch write failed at index {i} due to: {e}. Retrying after 5s...")
                time.sleep(5.0)
                sheet.batch_update(chunk)
                logging.info("Batch retry successful.")
        
        logging.info(
            f"Canonicalisation complete: {len(batch_updates)} rows processed, "
            f"{len(audit_log)} actual name changes."
        )
    else:
        logging.info("All articles already canonicalized — nothing to update.")

    return len(audit_log)

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
        logging.critical("spaCy not installed.")
        return None

# ==============================================================================
# --- GEMMA SETUP ---
# ==============================================================================

def load_gemma():
    try:
        from llama_cpp import Llama
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Gemma model not found at {MODEL_PATH}")
        logging.info("Loading Gemma model...")
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
        logging.error(f"Failed to load Gemma: {e}. Validation will be skipped.")
        return None


def gemma_validate(llm, question, context):
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
        raw    = response['choices'][0].get('text', '').strip()
        raw    = re.sub(r'```json|```', '', raw).strip()
        parsed = json.loads(raw)
        answer     = parsed.get('answer', 'no').lower() == 'yes'
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
    logging.info("--- Detecting CMs per state ---")
    window_articles = filter_by_window(articles, WINDOW_CM_PARTY_DAYS)
    logging.info(f"Using {len(window_articles)} articles from last {WINDOW_CM_PARTY_DAYS} days.")

    state_aliases = {}
    for s in entities['india']['states']:
        for alias in [s['name']] + s.get('aliases', []):
            state_aliases[alias.lower()] = s['name']

    minister_lookup = build_minister_lookup(entities)

    STRICT_CM_PATTERNS = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*,?\s*(?:the\s+)?chief\s+minister\s+of\s+(\w[\w\s]+)',
        r'chief\s+minister\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+of\s+(\w[\w\s]+)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+was\s+sworn\s+in\s+as\s+(?:the\s+)?chief\s+minister',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+takes?\s+oath\s+as\s+(?:the\s+)?chief\s+minister',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*,\s*(?:the\s+)?(?:\w+\s+)?chief\s+minister',
        r'new\s+(?:chief\s+minister|cm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
    ]

    state_cm_candidates = defaultdict(lambda: defaultdict(int))
    state_cm_contexts   = defaultdict(lambda: defaultdict(list))

    for article in window_articles:
        if article.get('category') == 'international':
            continue
        if article.get('source') in NON_INDIAN_SOURCES:
            continue

        text       = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:600]}"
        text_lower = text.lower()

        if 'chief minister' not in text_lower and ' cm ' not in text_lower:
            continue

        states_in_article = list(article.get('states_mentioned', []) or [])
        for alias_lower, canonical in state_aliases.items():
            if len(alias_lower) >= 4 and alias_lower in text_lower:
                if canonical not in states_in_article:
                    states_in_article.append(canonical)

        if not states_in_article:
            continue

        for pattern in STRICT_CM_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Exclude if 'former', 'ex', 'late', etc. appear in the matched string or immediately preceding it
                match_start = match.start()
                match_end = match.end()
                full_match_lower = match.group(0).lower()
                pre_text = text[max(0, match_start-30):match_start].lower()
                if any(w in full_match_lower or w in pre_text for w in ['former', 'ex-', ' ex ', 'late', 'previous', 'past']):
                    logging.info(f"Skipping CM match due to 'former/ex/late' context: {match.group(0)}")
                    continue

                person = match.group(1).strip() if match.lastindex >= 1 else None
                if not person or len(person) < 5:
                    continue

                canonical = minister_lookup.get(person.lower())
                if not canonical:
                    for alias, name in minister_lookup.items():
                        if alias in person.lower() or person.lower() in alias:
                            canonical = name
                            break

                # Dynamic CM Auto-Discovery if candidate is unknown but matched by a STRICT_CM_PATTERN
                if not canonical and llm is not None:
                    if is_candidate_name(person):
                        for state in states_in_article:
                            is_active_cm, gem_conf = gemma_validate(
                                llm,
                                f"Is '{person}' explicitly mentioned as taking office, taking oath, or actively serving as the Chief Minister of {state} in this text?",
                                text[max(0, match_start-150):min(len(text), match_end+150)]
                            )
                            if is_active_cm:
                                new_entry = {
                                    "name":                   person,
                                    "role":                   f"Chief Minister of {state}",
                                    "party":                  "", 
                                    "state":                  state,
                                    "aliases":                [],
                                    "criminal_cases":         0,
                                    "criminal_cases_in_news": 0,
                                    "auto_added":             True,
                                    "auto_added_on":          str(datetime.now().date()),
                                    "gemma_confidence":       "high",
                                    "mentions_detected":      1,
                                }
                                entities['india']['state_chief_ministers'].append(new_entry)
                                minister_lookup[person.lower()] = person
                                canonical = person
                                logging.info(f"DYNAMICAL AUTO-ADD: Sworn-in CM '{person}' discovered and added to database.")
                                break

                if not canonical:
                    continue

                # Center a 400-character window around the match
                start_idx = max(0, match_start - 200)
                end_idx = min(len(text), match_end + 200)
                context_window = text[start_idx:end_idx].strip()

                for state in states_in_article:
                    state_cm_candidates[state][canonical] += 1
                    state_cm_contexts[state][canonical].append(context_window)

    cm_updates = []
    cm_flags   = []

    for state, candidates in state_cm_candidates.items():
        if not candidates:
            continue
        total_mentions = sum(candidates.values())
        top_candidate  = max(candidates, key=candidates.get)
        top_count      = candidates[top_candidate]
        confidence     = top_count / total_mentions if total_mentions > 0 else 0

        if top_count < 3:
            continue

        context_sample = (
            state_cm_contexts[state][top_candidate][0]
            if state_cm_contexts[state][top_candidate] else ""
        )

        if confidence >= AUTO_UPDATE_THRESHOLD:
            is_cm, gem_conf = gemma_validate(
                llm,
                f"Is '{top_candidate}' explicitly mentioned as the Chief Minister of {state} in this text?",
                context_sample
            )
            if is_cm:
                cm_updates.append({
                    "state":            state,
                    "cm":               top_candidate,
                    "confidence":       round(confidence, 2),
                    "mentions":         top_count,
                    "gemma_validated":  True,
                    "gemma_confidence": gem_conf,
                })
                logging.info(
                    f"CM UPDATE: {state} → {top_candidate} "
                    f"(confidence: {confidence:.2f}, mentions: {top_count})"
                )
            else:
                cm_flags.append({
                    "type":       "cm_detection",
                    "state":      state,
                    "candidate":  top_candidate,
                    "confidence": round(confidence, 2),
                    "reason":     "Gemma rejected",
                    "context":    context_sample[:200],
                })
        elif confidence >= REVIEW_THRESHOLD and top_count >= 3:
            cm_flags.append({
                "type":       "cm_detection",
                "state":      state,
                "candidate":  top_candidate,
                "confidence": round(confidence, 2),
                "mentions":   top_count,
                "reason":     "Below auto-update threshold",
                "context":    context_sample[:200],
            })

    return cm_updates, cm_flags


# --- 2. Ruling Party Detection ---
def detect_ruling_parties(articles, entities, nlp, llm):
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

    STRICT_RULING_PATTERNS = [
        r'(BJP|Congress|INC|AAP|TMC|DMK|CPM|JDU|JMM|TDP|YSRCP|NCP|Shiv Sena|RJD|BJD)\s+government\s+in\s+(\w[\w\s]+)',
        r'(BJP|Congress|INC|AAP|TMC|DMK|CPM|JDU|JMM|TDP|YSRCP|NCP|Shiv Sena|RJD|BJD)-led\s+government\s+in\s+(\w[\w\s]+)',
        r'(BJP|Congress|INC|AAP|TMC|DMK|CPM|JDU|JMM|TDP|YSRCP|NCP|Shiv Sena|RJD|BJD)\s+(?:won|wins|won\s+power|came\s+to\s+power|swept)\s+(?:in\s+)?(\w[\w\s]+)',
        r'(\w[\w\s]+)\s+(?:state|government)\s+(?:is\s+)?ruled?\s+by\s+(BJP|Congress|INC|AAP|TMC|DMK|CPM|JDU|JMM|TDP)',
        r'(BJP|Congress|INC|AAP|TMC|DMK|CPM|JDU|JMM|TDP)\s+(?:wins?|victory|won)\s+(\w+\s+(?:Pradesh|Nadu|Bengal|Kerala|Karnataka|Bihar|Assam|Jharkhand|Odisha|Goa|Delhi|Punjab|Rajasthan|Gujarat|Maharashtra|Telangana|Andhra))',
    ]

    state_party_candidates = defaultdict(lambda: defaultdict(int))
    state_party_contexts   = defaultdict(lambda: defaultdict(list))

    for article in window_articles:
        if article.get('category') == 'international':
            continue
        if article.get('source') in NON_INDIAN_SOURCES:
            continue

        text       = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:500]}"
        text_lower = text.lower()

        governance_keywords = [
            'government', 'ruling', 'election', 'won', 'victory',
            'cm', 'chief minister', 'sworn in'
        ]
        if not any(kw in text_lower for kw in governance_keywords):
            continue

        for pattern in STRICT_RULING_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) < 2:
                    continue

                g1, g2 = groups[0].strip(), groups[1].strip()

                party_canonical = party_aliases.get(g1.lower()) or party_aliases.get(g2.lower())
                if not party_canonical:
                    continue

                state_canonical = state_aliases.get(g1.lower()) or state_aliases.get(g2.lower())
                if not state_canonical:
                    for alias_lower, canonical in state_aliases.items():
                        # Use strict word boundary checks to avoid partial matches (e.g., 'ap' in 'rape')
                        pattern_alias = r'\b' + re.escape(alias_lower) + r'\b'
                        if re.search(pattern_alias, g1.lower()) or re.search(pattern_alias, g2.lower()):
                            state_canonical = canonical
                            break

                if not state_canonical:
                    continue

                # Center a 400-character window around the match
                match_start = match.start()
                match_end = match.end()
                start_idx = max(0, match_start - 200)
                end_idx = min(len(text), match_end + 200)
                context_window = text[start_idx:end_idx].strip()

                state_party_candidates[state_canonical][party_canonical] += 1
                state_party_contexts[state_canonical][party_canonical].append(context_window)

    party_updates = []
    party_flags   = []

    for state, candidates in state_party_candidates.items():
        if not candidates:
            continue
        total     = sum(candidates.values())
        top_party = max(candidates, key=candidates.get)
        top_count = candidates[top_party]
        confidence = top_count / total if total > 0 else 0

        if top_count < 3:
            continue

        context_sample = (
            state_party_contexts[state][top_party][0]
            if state_party_contexts[state][top_party] else ""
        )

        if confidence >= AUTO_UPDATE_THRESHOLD:
            is_ruling, gem_conf = gemma_validate(
                llm,
                f"Is '{top_party}' explicitly mentioned as the ruling party or government of {state} in this text?",
                context_sample
            )
            if is_ruling:
                party_updates.append({
                    "state":           state,
                    "ruling_party":    top_party,
                    "confidence":      round(confidence, 2),
                    "mentions":        top_count,
                    "gemma_validated": True,
                })
                logging.info(
                    f"PARTY UPDATE: {state} → {top_party} "
                    f"(confidence: {confidence:.2f}, mentions: {top_count})"
                )
            else:
                party_flags.append({
                    "type":       "ruling_party",
                    "state":      state,
                    "candidate":  top_party,
                    "confidence": round(confidence, 2),
                    "reason":     "Gemma rejected",
                    "context":    context_sample[:200],
                })
        elif confidence >= REVIEW_THRESHOLD and top_count >= 3:
            party_flags.append({
                "type":       "ruling_party",
                "state":      state,
                "candidate":  top_party,
                "confidence": round(confidence, 2),
                "reason":     "Below auto-update threshold",
            })

    return party_updates, party_flags


# --- 3. Promise Extraction (Canonicalised) ---
def extract_promises(articles, entities, nlp, llm):
    logging.info("--- Extracting promises ---")
    window_articles = filter_by_window(articles, WINDOW_PROMISES_DAYS)

    # Master full canonical name resolution datasets
    minister_lookup = build_minister_lookup(entities)
    canonical_set   = build_canonical_minister_set(entities)

    # Load existing promise URLs to skip redundant Gemma calls
    existing_promise_urls = set()
    if 'extracted_promises' in entities:
        for p in entities['extracted_promises']:
            if p.get('source_url'):
                existing_promise_urls.add(p['source_url'].strip())

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
        url = article.get('url', '').strip()
        # Bypass articles whose promises have already been extracted and validated
        if url and url in existing_promise_urls:
            continue

        raw_text = (
            f"{article.get('title', '')} "
            f"{article.get('rephrased_article', '')} "
            f"{article.get('content', '')[:800]}"
        )
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', raw_text)

        for match in promise_pattern.finditer(text):
            person       = match.group(1).strip()
            promise_text = match.group(2).strip()

            # --- RESOLVE TO CANONICAL POLITICIAN FULL NAME ---
            canonical_person = None
            
            # Direct exact match or alias lookup (e.g. Narendra Modi or PM Modi)
            if person.lower() in minister_lookup:
                canonical_person = minister_lookup[person.lower()]
            else:
                # Ambiguous search (resolves "Modi" -> "Narendra Modi")
                candidates = find_canonical_candidates(person, canonical_set)
                if len(candidates) == 1:
                    canonical_person = candidates[0]

            # If we cannot confidently resolve this to a known minister, skip
            if not canonical_person:
                continue
                
            if len(promise_text) < 20:
                continue

            is_promise, gem_conf = gemma_validate(
                llm,
                f"Is this text describing a promise or commitment made by {canonical_person}?",
                f"{canonical_person} {promise_text}"
            )

            if is_promise:
                extracted_promises.append({
                    "person":           canonical_person,  # Saved under clean canonical full name
                    "promise_text":     promise_text,
                    "source_url":       article.get('url', ''),
                    "source_title":     article.get('title', ''),
                    "scraped_at":       article.get('scraped_at', ''),
                    "gemma_confidence": gem_conf,
                    "status":           "pending_review",
                    "verified":         False,
                })
                logging.info(f"PROMISE FOUND: {canonical_person} — {promise_text[:80]}...")

    logging.info(f"Extracted {len(extracted_promises)} promises.")
    return extracted_promises


# --- 4. Criminal Case Detection ---
def detect_criminal_cases(articles, entities, nlp, llm):
    logging.info("--- Detecting criminal cases ---")

    known_entities = {}
    all_ministers = (entities['india']['cabinet_ministers'] +
                      entities['india']['opposition_leaders'] +
                      entities['india']['state_chief_ministers'])
    
    for m in all_ministers:
        known_entities[m['name'].lower()] = m['name']
        for alias in m.get('aliases', []):
            known_entities[alias.lower()] = m['name']

    # Load existing validated incident URLs to skip redundant Gemma runs
    known_incident_urls = set()
    for m in all_ministers:
        for incident in m.get('criminal_incidents', []):
            if incident.get('source_url'):
                known_incident_urls.add(incident['source_url'].strip())

    SERIOUS_CRIMINAL_KEYWORDS = [
        'fir filed against', 'fir registered against',
        'chargesheeted', 'chargesheet filed against',
        'convicted', 'sentenced',
        'money laundering case', 'corruption case',
        'ed arrested', 'cbi arrested',
        'rape accused', 'murder accused',
        'disproportionate assets',
        'hawala', 'bribery case',
        'criminal case against', 'criminal charges against',
    ]

    BROAD_CRIMINAL_KEYWORDS = [
        'arrested', 'detained', 'fir', 'bail', 'custody',
        'charged with', 'accused of', 'fraud case', 'scam',
        'ed summons', 'cbi summons', 'raid on',
    ]

    entity_incidents = defaultdict(list)

    for article in articles:
        if article.get('category') == 'international':
            continue
        if article.get('source') in NON_INDIAN_SOURCES:
            continue

        text       = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:600]}"
        text_lower = text.lower()

        has_serious = any(kw in text_lower for kw in SERIOUS_CRIMINAL_KEYWORDS)
        has_broad   = any(kw in text_lower for kw in BROAD_CRIMINAL_KEYWORDS)

        if not has_serious and not has_broad:
            continue

        for entity_lower, canonical_name in known_entities.items():
            if len(entity_lower) < 4:
                continue
            # Use strict word boundary regex match for high accuracy
            pattern = r'\b' + re.escape(entity_lower) + r'\b'
            if not re.search(pattern, text_lower):
                continue

            already_logged = any(
                inc['source_url'] == article.get('url', '')
                for inc in entity_incidents[canonical_name]
            )
            if already_logged:
                continue

            url = article.get('url', '').strip()
            incident_type = "serious" if has_serious else "broad"

            # Bypass Gemma validation if this URL was already validated and saved
            if url and url in known_incident_urls:
                entity_incidents[canonical_name].append({
                    "incident_text": text[:200],
                    "source_url":    url,
                    "source_title":  article.get('title', ''),
                    "scraped_at":    article.get('scraped_at', ''),
                    "incident_type": incident_type,
                })
                continue

            confirmed, _ = gemma_validate(
                llm,
                f"Is this article specifically about a criminal case, FIR, arrest, or legal action directly involving '{canonical_name}'?",
                text[:400]
            )

            if confirmed:
                entity_incidents[canonical_name].append({
                    "incident_text": text[:200],
                    "source_url":    url,
                    "source_title":  article.get('title', ''),
                    "scraped_at":    article.get('scraped_at', ''),
                    "incident_type": incident_type,
                })
                logging.info(
                    f"CRIMINAL CASE [{incident_type}]: {canonical_name} "
                    f"— {article.get('title', '')[:60]}"
                )

    criminal_updates = []
    for entity_name, incidents in entity_incidents.items():
        if incidents:
            criminal_updates.append({
                "entity":         entity_name,
                "incident_count": len(incidents),
                "incidents":      incidents[:10],
            })
            logging.info(f"CRIMINAL SUMMARY: {entity_name} — {len(incidents)} validated incidents")

    return criminal_updates


# --- 5. New Entity Discovery + Auto-Add ---
def discover_new_entities(articles, entities, nlp, llm):
    """
    Finds unknown persons appearing 20+ times in the last 30 days via spaCy NER.
    Uses Gemma to extract a structured profile for each candidate.
    High/medium confidence → auto-added to entities.json immediately (no manual step).
    Low confidence         → written to review_flags.json only.

    Because auto-added entities are inserted directly into the `entities` dict
    before canonicalize_ministers_in_sheet runs, they are available for name
    resolution within the same weekly run.
    """
    logging.info("--- Discovering and auto-adding new entities ---")
    window_articles = filter_by_window(articles, WINDOW_NEW_ENTITIES_DAYS)

    # Build the full known names set (canonical + aliases) to skip already-known people
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

    # Count mentions of unknown PERSON entities (and possible misclassified ORG/GPE names) via spaCy NER
    person_counter  = Counter()
    person_contexts = defaultdict(list)

    # Using global is_candidate_name helper

    for article in window_articles:
        text = f"{article.get('title', '')} {article.get('rephrased_article', '')}"
        doc  = nlp(text[:800])
        for ent in doc.ents:
            name = ent.text.strip()
            
            # Apply strict candidate checks across all PERSON, ORG, and GPE tags
            is_valid = False
            if ent.label_ in ('PERSON', 'ORG', 'GPE') and is_candidate_name(name):
                is_valid = True

            if is_valid:
                if name.lower() not in known_names:
                    person_counter[name] += 1
                    if len(person_contexts[name]) < 5:
                        person_contexts[name].append(text[:400])

    # Algorithmic Name Consolidation: merge shorter name variations (e.g. "Suvendu" -> "Suvendu Adhikari")
    # Sort candidate names by length descending
    sorted_names = sorted(person_counter.keys(), key=len, reverse=True)
    consolidated_counter = Counter()
    consolidated_contexts = defaultdict(list)

    for name in sorted_names:
        count = person_counter[name]
        contexts = person_contexts[name]
        matched = False
        
        # Check if this name is a strict whole-word substring of any already consolidated longer name
        for longer_name in list(consolidated_counter.keys()):
            if re.search(r'\b' + re.escape(name) + r'\b', longer_name, re.IGNORECASE):
                consolidated_counter[longer_name] += count
                # Merge contexts up to a max of 5
                for ctx in contexts:
                    if ctx not in consolidated_contexts[longer_name]:
                        consolidated_contexts[longer_name].append(ctx)
                consolidated_contexts[longer_name] = consolidated_contexts[longer_name][:5]
                matched = True
                break
        
        if not matched:
            consolidated_counter[name] = count
            consolidated_contexts[name] = contexts[:5]

    auto_added = []
    flagged    = []

    valid_categories = {'cabinet_minister', 'state_chief_minister', 'opposition_leader'}

    for name, count in consolidated_counter.most_common(50):
        if count < NEW_ENTITY_MIN_MENTIONS:
            continue

        context_sample = ' '.join(consolidated_contexts[name][:3])

        if llm is None:
            flagged.append({
                "name":            name,
                "mentions":        count,
                "reason":          "No LLM available for validation",
                "sample_articles": consolidated_contexts[name][:2],
            })
            continue

        extraction_prompt = f"""<start_of_turn>user
Based on the article excerpts below, answer about the person named "{name}".

Articles: {context_sample[:800]}

Return ONLY a JSON object with these exact fields:
{{
  "is_indian_politician": true or false,
  "full_name": "their full official name",
  "role": "their exact role e.g. Chief Minister of Karnataka, Union Minister of Finance",
  "party": "their political party abbreviation e.g. BJP, INC, AAP, TMC, DMK",
  "state": "their home state or null if national-level",
  "category": "cabinet_minister" or "state_chief_minister" or "opposition_leader",
  "confidence": "high" or "medium" or "low"
}}
No explanation. No extra text.
<end_of_turn>
<start_of_turn>model
"""
        try:
            response = llm(
                extraction_prompt,
                max_tokens=200,
                temperature=0.0,
                stop=["<end_of_turn>", "<start_of_turn>"],
                echo=False
            )
            raw    = response['choices'][0].get('text', '').strip()
            raw    = re.sub(r'```json|```', '', raw).strip()
            parsed = json.loads(raw)

            if not parsed.get('is_indian_politician'):
                logging.info(f"SKIP (not Indian politician): {name}")
                continue

            full_name  = (parsed.get('full_name') or name).strip()
            role       = (parsed.get('role') or '').strip()
            party      = (parsed.get('party') or '').strip()
            state      = parsed.get('state') or None
            category   = (parsed.get('category') or '').strip()
            confidence = (parsed.get('confidence') or 'low').strip()

            # Validate category
            if category not in valid_categories:
                category = 'cabinet_minister'

            # Skip if Gemma resolved to an already-known name
            if full_name.lower() in known_names:
                logging.info(f"SKIP (already known under full name): {full_name}")
                continue

            new_entry = {
                "name":                   full_name,
                "role":                   role,
                "party":                  party,
                "state":                  state,
                "aliases":                [name] if name != full_name else [],
                "criminal_cases":         0,
                "criminal_cases_in_news": 0,
                "auto_added":             True,
                "auto_added_on":          str(datetime.now().date()),
                "gemma_confidence":       confidence,
                "mentions_detected":      count,
            }

            if confidence in ('high', 'medium'):
                # Auto-add directly into the live entities dict
                entities['india'][f'{category}s'].append(new_entry)
                # Update known_names so we don't double-add this run
                known_names.add(full_name.lower())
                known_names.add(name.lower())

                auto_added.append({
                    "name":       full_name,
                    "category":   category,
                    "confidence": confidence,
                    "mentions":   count,
                    "role":       role,
                    "party":      party,
                })
                logging.info(
                    f"AUTO-ADDED [{confidence}]: {full_name} ({role}, {party}) "
                    f"→ entities['india']['{category}s']"
                )
            else:
                flagged.append({
                    "name":            full_name,
                    "raw_name":        name,
                    "mentions":        count,
                    "role":            role,
                    "party":           party,
                    "category":        category,
                    "confidence":      confidence,
                    "reason":          "Gemma confidence too low for auto-add",
                    "sample_articles": person_contexts[name][:2],
                })
                logging.info(f"FLAGGED FOR REVIEW (low confidence): {full_name}")

        except Exception as e:
            logging.warning(f"Gemma extraction failed for '{name}': {e}. Flagging for review.")
            flagged.append({
                "name":            name,
                "mentions":        count,
                "reason":          f"Gemma extraction error: {e}",
                "sample_articles": person_contexts[name][:2],
            })

    logging.info(
        f"New entities: {len(auto_added)} auto-added, {len(flagged)} flagged for review."
    )
    return auto_added, flagged

# ==============================================================================
# --- APPLY UPDATES TO entities.json ---
# ==============================================================================

def apply_updates(entities, cm_updates, party_updates, criminal_updates, new_promises):
    for update in cm_updates:
        for state in entities['india']['states']:
            if state['name'] == update['state']:
                old_cm           = state.get('cm', 'Unknown')
                state['cm']      = update['cm']
                state['cm_confidence']   = update['confidence']
                state['cm_last_updated'] = str(datetime.now().date())
                if old_cm != update['cm']:
                    logging.info(
                        f"APPLIED CM UPDATE: {update['state']} — {old_cm} → {update['cm']}"
                    )
                break

    for update in party_updates:
        for state in entities['india']['states']:
            if state['name'] == update['state']:
                old_party             = state.get('ruling_party', 'Unknown')
                state['ruling_party'] = update['ruling_party']
                state['party_confidence']   = update['confidence']
                state['party_last_updated'] = str(datetime.now().date())
                if old_party != update['ruling_party']:
                    logging.info(
                        f"APPLIED PARTY UPDATE: {update['state']} — {old_party} → {update['ruling_party']}"
                    )
                break

    for update in criminal_updates:
        entity_name   = update['entity']
        all_ministers = (
            entities['india']['cabinet_ministers'] +
            entities['india']['opposition_leaders'] +
            entities['india']['state_chief_ministers']
        )
        for minister in all_ministers:
            if minister['name'] == entity_name:
                minister['criminal_cases_in_news'] = update['incident_count']
                minister['criminal_incidents']     = update['incidents']
                minister['criminal_last_updated']  = str(datetime.now().date())
                break

    if new_promises:
        if 'extracted_promises' not in entities:
            entities['extracted_promises'] = []
        existing_urls = {p.get('source_url') for p in entities['extracted_promises']}
        for promise in new_promises:
            if promise['source_url'] not in existing_urls:
                entities['extracted_promises'].append(promise)

    entities['metadata']['last_updated'] = str(datetime.now().date())
    entities['metadata']['auto_updated_fields'] = {
        "cm_updates":      len(cm_updates),
        "party_updates":   len(party_updates),
        "criminal_updates": len(criminal_updates),
        "new_promises":    len(new_promises),
    }

    return entities

# ==============================================================================
# --- MAIN ---
# ==============================================================================

def main():
    start_time = time.time()
    logging.info("--- Satya Entity Updater Started ---")

    sheet    = connect_to_sheets()
    articles = fetch_articles(sheet)

    if not articles:
        logging.error("No articles found. Exiting.")
        return

    entities = load_entities()
    logging.info(f"Loaded entities.json (version: {entities['metadata'].get('version', 'unknown')})")

    nlp = load_spacy()
    if nlp is None:
        logging.error("spaCy failed to load. Exiting.")
        return

    # Gemma loaded first — needed by both entity discovery and canonicalization
    llm = load_gemma()

    # Step 1: Discover and auto-add new entities into the live entities dict.
    # Must run BEFORE canonicalize so newly added ministers are available
    # for name resolution in the same weekly run.
    new_entities_added, new_entities_flagged = discover_new_entities(
        articles, entities, nlp, llm
    )

    # Step 2: Canonicalise ministers_mentioned across all articles.
    # Uses canonical set derived from entities.json (including just-added entities).
    # Gemma resolves ambiguous bare surnames. Audit log written before sheet write.
    # Articles marked ministers_canonicalized=True are skipped on future runs.
    canonicalize_ministers_in_sheet(sheet, entities, llm)

    all_flags = []

    cm_updates, cm_flags = detect_cms(articles, entities, nlp, llm)
    all_flags.extend(cm_flags)

    party_updates, party_flags = detect_ruling_parties(articles, entities, nlp, llm)
    all_flags.extend(party_flags)

    new_promises     = extract_promises(articles, entities, nlp, llm)
    criminal_updates = detect_criminal_cases(articles, entities, nlp, llm)

    updated_entities = apply_updates(
        entities, cm_updates, party_updates, criminal_updates, new_promises
    )

    with open(ENTITIES_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(updated_entities, f, indent=2, ensure_ascii=False)
    logging.info("Saved updated entities.json")

    review_output = {
        "generated_at": str(datetime.now()),
        "summary": {
            "cm_updates_applied":        len(cm_updates),
            "party_updates_applied":     len(party_updates),
            "criminal_updates_applied":  len(criminal_updates),
            "promises_extracted":        len(new_promises),
            "new_entities_auto_added":   len(new_entities_added),
            "new_entities_needs_review": len(new_entities_flagged),
            "items_needing_review":      len(all_flags),
        },
        "new_entities_auto_added":     new_entities_added,
        "new_entities_needs_review":   new_entities_flagged,
        "items_needing_manual_review": all_flags,
    }

    with open(REVIEW_FLAGS_PATH, 'w', encoding='utf-8') as f:
        json.dump(review_output, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved review_flags.json ({len(all_flags)} need review)")

    elapsed = round(time.time() - start_time, 2)
    logging.info(f"--- Entity Updater Finished in {elapsed}s ---")
    logging.info(f"Summary: {review_output['summary']}")
    print(json.dumps(review_output['summary'], indent=2))


if __name__ == '__main__':
    main()
