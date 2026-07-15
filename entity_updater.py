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
import sqlite3
import zlib

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
ENTITIES_JSON_URL = os.environ.get('ENTITIES_JSON_URL', '')

ENTITIES_OUTPUT_PATH    = './entities.json'
REVIEW_FLAGS_PATH       = './review_flags.json'
CANONICALIZE_AUDIT_PATH = './canonicalize_audit.json'

WINDOW_CM_PARTY_DAYS     = 30
WINDOW_PROMISES_DAYS     = 180
WINDOW_CRIMINAL_DAYS     = 9999
WINDOW_NEW_ENTITIES_DAYS = 30

AUTO_UPDATE_THRESHOLD   = 0.75
REVIEW_THRESHOLD        = 0.40
NEW_ENTITY_MIN_MENTIONS = 20
# 100 per run: smaller chunks + the existing has_more self-loop = same daily
# throughput, but a failed run loses at most 100 articles' work, not 500.
MAX_ARTICLES_PER_RUN    = 100

# Qwen 14B — same gate model (and same GHA cache) as the timeline and promise
# pipelines. Entity canonicalization is upstream of everything; it deserves
# the strongest judgment we run anywhere.
MODEL_PATH = os.environ.get('MODEL_GATE_PATH', "./models/Qwen2.5-14B-Instruct-Q5_K_M.gguf")

NON_INDIAN_SOURCES = {'The Dawn', 'BBC', 'Al Jazeera', 'The Guardian'}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==============================================================================
# --- DATABASE CONFIGURATION ---
# ==============================================================================
def load_env():
    env_paths = [
        os.path.join(os.path.dirname(__file__), '.env'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, val = line.split('=', 1)
                        os.environ[key.strip()] = val.strip()

load_env()

default_db_path = '/Users/mac/Downloads/Code/Satya/satya.db'
if not os.path.exists(os.path.dirname(default_db_path)):
    default_db_path = os.path.join(os.path.dirname(__file__), 'satya.db')

DB_PATH = os.environ.get('SATYA_DB_PATH', default_db_path)

def get_db_connection():
    db_url = os.environ.get('SATYA_DB_URL')
    db_token = os.environ.get('SATYA_DB_TOKEN')
    
    if db_url and (db_url.startswith('libsql://') or db_url.startswith('https://')):
        try:
            import libsql
            return libsql.connect(database=db_url, auth_token=db_token)
        except ImportError:
            logging.error("libsql package not installed. Falling back to local sqlite3.")
            
    return sqlite3.connect(DB_PATH)

PARTY_SLUG_ALIASES = {
  'bharatiya_janata_party': 'bjp',
  'bhartiya_janata_party': 'bjp',
  'bharatiya_janata': 'bjp',
  'indian_national_congress': 'inc',
  'congress': 'inc',
  'congress_party': 'inc',
  'grand_old_party': 'inc',
  'aam_aadmi_party': 'aap',
  'aam_aadmi': 'aap',
  'common_man_party': 'aap',
  'all_india_trinamool_congress': 'tmc',
  'trinamool': 'tmc',
  'aitc': 'tmc',
  'trinamool_congress': 'tmc',
  'samajwadi_party': 'sp',
  'samajwadi': 'sp',
  'bahujan_samaj_party': 'bsp',
  'bahujan_samaj': 'bsp',
  'dravida_munnetra_kazhagam': 'dmk',
  'dravidam': 'dmk',
  'communist_party_of_india_marxist': 'cpm',
  'cpim': 'cpm',
  'left_front': 'cpm',
  'marxist': 'cpm',
  'janata_dal_united': 'jdu',
  'nitish_party': 'jdu',
  'nationalist_congress_party': 'ncp',
  'nationalist_congress': 'ncp',
  'telugu_desam_party': 'tdp',
  'telugu_desam': 'tdp',
  'jharkhand_mukti_morcha': 'jmm',
  'jharkhand_mukti': 'jmm',
  'rashtriya_janata_dal': 'rjd',
  'rashtriya_janata': 'rjd',
  'all_india_majlis_e_ittehadul_muslimeen': 'aimim',
  'majlis': 'aimim',
  'mim': 'aimim',
  'shiv_sena_eknath_shinde': 'shiv_sena',
  'shinde_sena': 'shiv_sena',
  'balasahebanchi_shiv_sena': 'shiv_sena',
  'viduthalai_chiruthaigal_katchi': 'vck',
  'jammu_and_kashmir_peoples_democratic_party': 'pdp',
  'peoples_democratic_party': 'pdp',
  'all_india_anna_dravida_munnetra_kazhagam': 'aiadmk',
  'all_india_anna_dmk': 'aiadmk',
  'marumalarchi_dravida_munnetra_kazhagam': 'mdmk',
}

def slugify(name):
    if not name:
        return ""
    s = name.lower()
    s = s.replace(' ', '_')
    s = s.replace('.', '')
    s = re.sub(r'[^a-z0-9_]', '', s)
    return s

def party_slugify(name):
    s = slugify(name)
    return PARTY_SLUG_ALIASES.get(s, s)

def refresh_bridge_rows(cursor, article_id, kind, values, slugify_fn):
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            values = []
    cursor.execute("DELETE FROM article_entities WHERE article_id = ? AND kind = ?", (article_id, kind))
    for v in set(values or []):
        if v:
            cursor.execute(
                "INSERT OR IGNORE INTO article_entities (article_id, kind, slug) VALUES (?, ?, ?)",
                (article_id, kind, slugify_fn(v))
            )

def fetch_articles(conn):
    rescan_requested = os.environ.get('RESCAN_HISTORY', '').lower() in ('1', 'true', 'yes')
    logging.info(f"Fetching classified articles from SQLite database (rescan_requested={rescan_requested})...")
    articles = []
    try:
        cursor = conn.cursor()
        if rescan_requested:
            cursor.execute("""
                SELECT id, cluster_id, source_id, title, url, content, image_url, scraped_at, 
                       category, sentiment, sentiment_target, rephrased_article, 
                       party_mentioned, ministers_mentioned, states_mentioned, cities_mentioned, 
                       topic_tags, civic_flag, civic_flag_score, civic_flag_category, civic_flag_reason, 
                       classified_at, status 
                FROM articles 
                WHERE status IN ('classified', 'entity_processed', 'processed')
            """)
        else:
            # Optimize: Only fetch unprocessed articles OR processed articles from the last 30 days
            import time
            cutoff_timestamp = int(time.time()) - 30 * 24 * 3600
            cursor.execute("""
                SELECT id, cluster_id, source_id, title, url, content, image_url, scraped_at, 
                       category, sentiment, sentiment_target, rephrased_article, 
                       party_mentioned, ministers_mentioned, states_mentioned, cities_mentioned, 
                       topic_tags, civic_flag, civic_flag_score, civic_flag_category, civic_flag_reason, 
                       classified_at, status 
                FROM articles INDEXED BY idx_articles_status_scraped
                WHERE status = 'classified' AND scraped_at >= ?
                UNION ALL
                SELECT id, cluster_id, source_id, title, url, content, image_url, scraped_at, 
                       category, sentiment, sentiment_target, rephrased_article, 
                       party_mentioned, ministers_mentioned, states_mentioned, cities_mentioned, 
                       topic_tags, civic_flag, civic_flag_score, civic_flag_category, civic_flag_reason, 
                       classified_at, status 
                FROM articles INDEXED BY idx_articles_scraped
                WHERE status IN ('entity_processed', 'processed') AND scraped_at >= ?
            """, (cutoff_timestamp, cutoff_timestamp))
        rows = cursor.fetchall()
    except Exception as e:
        logging.error(f"Failed to query articles from database: {e}")
        return []

    for r in rows:
        article_id = r[0]
        cluster_id = r[1]
        source_id = r[2]
        title = r[3]
        url = r[4]
        compressed_content = r[5]
        image_url = r[6]
        scraped_timestamp = r[7]
        category = r[8]
        sentiment = r[9]
        sentiment_target = r[10]
        compressed_rephrased = r[11]
        party_mentioned_str = r[12]
        ministers_mentioned_str = r[13]
        states_mentioned_str = r[14]
        cities_mentioned_str = r[15]
        topic_tags_str = r[16]
        civic_flag_val = r[17]
        civic_flag_score_val = r[18]
        civic_flag_category_val = r[19]
        civic_flag_reason_val = r[20]
        classified_at_val = r[21]
        status_val = r[22]

        try:
            content = zlib.decompress(compressed_content).decode('utf-8') if compressed_content else ""
        except Exception:
            content = ""

        try:
            rephrased = zlib.decompress(compressed_rephrased).decode('utf-8') if compressed_rephrased else content
        except Exception:
            rephrased = content

        scraped_at_str = ""
        if scraped_timestamp:
            try:
                scraped_at_str = datetime.fromtimestamp(scraped_timestamp).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        # Maintain exact backward compatibility keys
        article = {
            'id': article_id,
            'row_idx': article_id,  # Map row_idx to id for database write-backs
            'cluster_id': cluster_id,
            'source_id': source_id,
            'title': title,
            'url': url,
            'content': content,
            'image_url': image_url,
            'scraped_at': scraped_at_str,
            'category': category,
            'sentiment': sentiment,
            'sentiment_target': sentiment_target,
            'rephrased_article': rephrased if rephrased else content,
            'party_mentioned': json.loads(party_mentioned_str) if party_mentioned_str else [],
            'ministers_mentioned': json.loads(ministers_mentioned_str) if ministers_mentioned_str else [],
            'states_mentioned': json.loads(states_mentioned_str) if states_mentioned_str else [],
            'cities_mentioned': json.loads(cities_mentioned_str) if cities_mentioned_str else [],
            'topic_tags': json.loads(topic_tags_str) if topic_tags_str else [],
            'civic_flag': civic_flag_val == 1,
            'civic_flag_score': civic_flag_score_val or 0,
            'civic_flag_category': civic_flag_category_val,
            'civic_flag_reason': civic_flag_reason_val,
            'classified_at': classified_at_val,
            'status': status_val,
            'ministers_canonicalized': (status_val in ('entity_processed', 'processed'))
        }

        try:
            article['scraped_dt'] = datetime.fromtimestamp(scraped_timestamp) if scraped_timestamp else datetime.now()
        except Exception:
            article['scraped_dt'] = datetime.now() - timedelta(days=365)

        articles.append(article)

    logging.info(f"Fetched {len(articles)} classified/processed articles from database.")
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
    # Indian politician names generally consist of 1 to 4 words
    if not (1 <= len(words) <= 4):
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
        'university', 'society', 'foundation',
        'sabha', 'lok', 'rajya', 'vidhan', 'bhavan', 'yojana', 'nigam',
        'samiti', 'morcha', 'panchayat', 'zilla', 'mandal', 'aayog',
        'sena', 'dal', 'parishad', 'sangh', 'seva'
    }
    if any(w.lower().rstrip('.') in stop_words for w in words):
        return False
    return True


def dedupe_entities(entities):
    """
    Removes duplicate politician profiles (same name across or within groups).
    Keeps the first-seen profile and merges aliases, criminal_incidents,
    and controversies from duplicates into it.
    """
    groups = ['cabinet_ministers', 'state_chief_ministers',
              'opposition_leaders', 'generic_politicians']
    seen = {}
    removed = 0
    for g in groups:
        kept = []
        for m in entities['india'].get(g, []):
            key = m.get('name', '').strip().lower()
            if not key:
                continue
            if key not in seen:
                seen[key] = m
                kept.append(m)
                continue
            primary = seen[key]
            for alias in m.get('aliases', []):
                if alias and alias not in primary.setdefault('aliases', []):
                    primary['aliases'].append(alias)
            for field in ('criminal_incidents', 'controversies'):
                urls = {i.get('source_url') for i in primary.get(field, [])}
                for inc in m.get(field, []):
                    if inc.get('source_url') not in urls:
                        primary.setdefault(field, []).append(inc)
                        urls.add(inc.get('source_url'))
            for field in ('party', 'state', 'role', 'wikipedia', 'constituency'):
                if not primary.get(field) and m.get(field):
                    primary[field] = m[field]
            if primary.get('criminal_incidents') is not None:
                primary['criminal_cases_in_news'] = len(primary['criminal_incidents'])
            removed += 1
            logging.info(f"DEDUPE: merged duplicate profile '{m.get('name')}' (from {g})")
        entities['india'][g] = kept
    # Second pass: fuzzy duplicates (spelling variants like 'Mamta'/'Mamata')
    import difflib
    def _norm(name):
        return re.sub(r'[^a-z ]', '', name.lower().replace('.', ' ')).strip()

    all_profiles = [(g, m) for g in groups for m in entities['india'].get(g, [])]
    to_remove = set()
    for i, (g1, m1) in enumerate(all_profiles):
        if id(m1) in to_remove:
            continue
        for g2, m2 in all_profiles[i + 1:]:
            if id(m2) in to_remove:
                continue
            n1, n2 = _norm(m1.get('name', '')), _norm(m2.get('name', ''))
            if not n1 or not n2:
                continue
            ratio = difflib.SequenceMatcher(None, n1, n2).ratio()
            same_context = (
                (m1.get('state') and m1.get('state') == m2.get('state')) or
                (m1.get('party') and m1.get('party') == m2.get('party'))
            )
            if ratio >= 0.85 and same_context:
                # Merge m2 into m1 (m1 = first/primary)
                for alias in [m2.get('name', '')] + m2.get('aliases', []):
                    if alias and alias not in m1.setdefault('aliases', []) and alias != m1.get('name'):
                        m1['aliases'].append(alias)
                for field in ('criminal_incidents', 'controversies'):
                    urls = {x.get('source_url') for x in m1.get(field, [])}
                    for inc in m2.get(field, []):
                        if inc.get('source_url') not in urls:
                            m1.setdefault(field, []).append(inc)
                            urls.add(inc.get('source_url'))
                for field in ('party', 'state', 'role', 'wikipedia', 'constituency'):
                    if not m1.get(field) and m2.get(field):
                        m1[field] = m2[field]
                if m1.get('criminal_incidents') is not None:
                    m1['criminal_cases_in_news'] = len(m1['criminal_incidents'])
                to_remove.add(id(m2))
                removed += 1
                logging.info(f"FUZZY DEDUPE: merged '{m2.get('name')}' into '{m1.get('name')}' (ratio {ratio:.2f})")
            elif 0.75 <= ratio < 0.85 and same_context:
                logging.warning(f"POSSIBLE DUPLICATE (not merged): '{m1.get('name')}' vs '{m2.get('name')}' (ratio {ratio:.2f})")
    if to_remove:
        for g in groups:
            entities['india'][g] = [m for m in entities['india'][g] if id(m) not in to_remove]

    if removed:
        logging.info(f"Deduplicated {removed} duplicate profiles.")
    return entities


def build_minister_lookup(entities):
    """
    Alias → canonical name lookup used by CM detection, criminal detection,
    and promise extraction. Built from the aliases field in entities.json.
    """
    lookup = {}
    all_ministers = (
        entities['india']['cabinet_ministers'] +
        entities['india']['state_chief_ministers'] +
        entities['india']['opposition_leaders'] +
        entities['india'].get('generic_politicians', [])
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
            entities['india']['opposition_leaders'] +
            entities['india'].get('generic_politicians', [])
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

def canonicalize_ministers_in_sheet(sheet, unprocessed_chunk, entities, llm):
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
    batch_updates = []
    audit_log     = []

    for article in unprocessed_chunk:
        row_idx = article.get('row_idx')
        if not row_idx:
            continue

        original = article.get('ministers_mentioned') or []

        # Build article context once — only used if Gemma disambiguation needed
        article_text = ' '.join(filter(None, [
            article.get('title', ''),
            article.get('rephrased_article', ''),
            (article.get('content') or '')[:1000],
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
                prompt = f"""<|im_start|>user
Article: {article_text[:1800]}

The article mentions "{name}". Based only on the article text above, which of these people is being referred to?
Options: {options_str}

Reply with ONLY the exact name from the options. No explanation.
<|im_end|>
<|im_start|>assistant
"""
                try:
                    response = llm(
                        prompt,
                        max_tokens=20,
                        temperature=0.0,
                        stop=["<|im_end|>", "<|im_start|>", "\n"],
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
        batch_updates.append({
            'id': article['id'],
            'ministers_mentioned': canonicalized
        })

    # Write audit log BEFORE returning — keeps log on disk
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

    return batch_updates


def commit_sheet_updates(conn, batch_updates):
    """
    Writes a list of batch updates back to the SQLite/Turso database.

    Chunked commits with reconnect-retry: one long transaction over a remote
    Hrana stream dies after a few minutes ("stream not found") and rolls back
    everything. Small transactions commit as we go; a dropped stream costs one
    chunk retry, not the whole batch. Returns the (possibly new) connection.
    """
    if not batch_updates:
        logging.info("No updates to commit to Database.")
        return conn

    logging.info(f"Committing {len(batch_updates)} updates to Database...")
    CHUNK = 50
    committed = 0
    for i in range(0, len(batch_updates), CHUNK):
        chunk = batch_updates[i:i + CHUNK]
        last_err = None
        for attempt in range(3):
            try:
                cursor = conn.cursor()
                for update in chunk:
                    article_id = update['id']
                    ministers = json.dumps(update['ministers_mentioned'])
                    cursor.execute("""
                        UPDATE articles
                        SET ministers_mentioned = ?, status = 'entity_processed'
                        WHERE id = ?
                    """, (ministers, article_id))
                    refresh_bridge_rows(cursor, article_id, 'minister', update['ministers_mentioned'], slugify)
                conn.commit()
                committed += len(chunk)
                last_err = None
                logging.info(f"Committed {committed}/{len(batch_updates)} updates...")
                break
            except Exception as e:
                last_err = e
                logging.error(f"Chunk commit attempt {attempt + 1} failed: {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
                if attempt < 2:
                    logging.info("Reconnecting database and retrying chunk...")
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = get_db_connection()
        if last_err is not None:
            logging.error(f"Giving up after 3 attempts at chunk starting index {i} ({committed} updates already committed and safe).")
            raise last_err

    logging.info("Successfully committed all updates to Database.")
    return conn

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
    # Name kept for call-site compatibility; loads the Qwen gate model.
    try:
        from llama_cpp import Llama
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Gate model not found at {MODEL_PATH}")
        logging.info(f"Loading gate model from {MODEL_PATH}...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=8192,
            n_batch=512,
            n_threads=int(os.environ.get('LLM_THREADS', 4)),  # match promise tracker; runner reports 4 vCPUs
            verbose=False
        )
        logging.info("Gate model loaded.")
        return llm
    except Exception as e:
        logging.error(f"Failed to load gate model: {e}. Validation will be skipped.")
        return None


def benchmark_threads():
    """Time identical short generations at n_threads=2 vs 4 on this runner,
    so the thread setting is decided by measurement, not Azure SMT guesswork."""
    from llama_cpp import Llama
    prompt = ("<|im_start|>user\nName the capital of India and one neighbouring "
              "country. Answer in one short sentence.<|im_end|>\n<|im_start|>assistant\n")
    results = {}
    for threads in (2, 4):
        llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_batch=512, n_threads=threads, verbose=False)
        t0 = time.time()
        for _ in range(3):
            llm(prompt, max_tokens=40, temperature=0.0, stop=["<|im_end|>"])
        results[threads] = (time.time() - t0) / 3
        del llm
        import gc
        gc.collect()
    for threads, secs in results.items():
        logging.info(f"[BENCH] n_threads={threads}: {secs:.1f}s per call (avg of 3)")
    faster = min(results, key=results.get)
    logging.info(f"[BENCH] Winner: n_threads={faster} "
                 f"({results[max(results, key=results.get)] / results[faster]:.2f}x faster than the other)")


def gemma_validate(llm, question, context):
    if llm is None:
        return True, "unvalidated"

    prompt = f"""<|im_start|>user
Read the text below and answer the question with ONLY a JSON object.

Text: {context[:2000]}

Question: {question}

Return ONLY: {{"answer": "yes" or "no", "confidence": "high" or "medium" or "low", "evidence": "for yes answers, the EXACT phrase copied verbatim from the text that proves it; empty string for no"}}
No explanation. No extra text.
<|im_end|>
<|im_start|>assistant
"""
    try:
        response = llm(
            prompt,
            max_tokens=140,
            temperature=0.1,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False
        )
        raw    = response['choices'][0].get('text', '').strip()
        raw    = re.sub(r'```json|```', '', raw).strip()
        parsed = json.loads(raw)
        answer     = parsed.get('answer', 'no').lower() == 'yes'
        confidence = parsed.get('confidence', 'low')

        # Anti-hallucination guard: a "yes" must cite a span actually present
        # in the text. Reject yes-answers whose evidence cannot be located.
        if answer:
            evidence = str(parsed.get('evidence', '')).strip()
            norm = lambda s: re.sub(r'\s+', ' ', s.lower()).strip()
            if not evidence or norm(evidence) not in norm(context):
                logging.info(f"Gemma 'yes' rejected — evidence span not found in text: {evidence[:80]!r}")
                return False, 'low'
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
        r'((?:[A-Z]\.)+\s*[A-Z][a-z]+|[A-Z][a-z]+(?:\s+(?:(?:[A-Z]\.)+\s*)?[A-Z][a-z]+)+)\s*,?\s*(?:the\s+)?chief\s+minister\s+of\s+(\w[\w\s]+)',
        r'chief\s+minister\s+((?:[A-Z]\.)+\s*[A-Z][a-z]+|[A-Z][a-z]+(?:\s+(?:(?:[A-Z]\.)+\s*)?[A-Z][a-z]+)+)\s+of\s+(\w[\w\s]+)',
        r'((?:[A-Z]\.)+\s*[A-Z][a-z]+|[A-Z][a-z]+(?:\s+(?:(?:[A-Z]\.)+\s*)?[A-Z][a-z]+)+)\s+was\s+sworn\s+in\s+as\s+(?:the\s+)?chief\s+minister',
        r'((?:[A-Z]\.)+\s*[A-Z][a-z]+|[A-Z][a-z]+(?:\s+(?:(?:[A-Z]\.)+\s*)?[A-Z][a-z]+)+)\s+takes?\s+oath\s+as\s+(?:the\s+)?chief\s+minister',
        r'((?:[A-Z]\.)+\s*[A-Z][a-z]+|[A-Z][a-z]+(?:\s+(?:(?:[A-Z]\.)+\s*)?[A-Z][a-z]+)+)\s*,\s*(?:the\s+)?(?:\w+\s+)?chief\s+minister',
        r'new\s+(?:chief\s+minister|cm)\s+((?:[A-Z]\.)+\s*[A-Z][a-z]+|[A-Z][a-z]+(?:\s+(?:(?:[A-Z]\.)+\s*)?[A-Z][a-z]+)+)',
    ]

    state_cm_candidates = defaultdict(lambda: defaultdict(int))
    state_cm_contexts   = defaultdict(lambda: defaultdict(list))

    for article in window_articles:
        if article.get('category') == 'international':
            continue
        if article.get('source') in NON_INDIAN_SOURCES:
            continue

        text       = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:1200]}"
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
                    person_lower = person.lower()
                    for alias, name in minister_lookup.items():
                        # Whole-word match only — 'shah' must not match 'shahid khan'
                        if re.search(r'\b' + re.escape(alias) + r'\b', person_lower) or \
                           re.search(r'\b' + re.escape(person_lower) + r'\b', alias):
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
                                    "affidavit_url":          "https://affidavit.eci.gov.in",
                                    "wikipedia":              f"https://en.wikipedia.org/wiki/{person.replace(' ', '_')}",
                                    "image_placeholder":      person.lower().replace(' ', '_').replace('.', ''),
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

        text       = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:1000]}"
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


# --- 2b. Central Government Detection (PM / President) ---
# A national office change is rare and saturates the news, so the evidence bar
# is deliberately much higher than for state CMs: more mentions, higher
# consensus, and multi-context LLM validation. Unknown candidates are never
# auto-added at this level — they go to review.
CENTRAL_MIN_MENTIONS   = 10    # vs 3 for state CMs
CENTRAL_AUTO_THRESHOLD = 0.85  # vs 0.75 for state CMs
CENTRAL_VALIDATE_CONTEXTS = 3  # distinct contexts sampled; majority must pass

# Strictly capitalized person-name capture (patterns run WITHOUT IGNORECASE so
# 'was'/'of India' can never be swallowed into a name). Office keywords use
# explicit case classes instead.
_NM = r'((?:[A-Z]\.\s*)*[A-Z][a-z]+(?:\s+(?:[A-Z]\.\s*)*[A-Z][a-z]+){1,3})'
_PM = r'[Pp]rime\s+[Mm]inister'
_PR = r'[Pp]resident\s+of\s+India'
_SW = r'(?:was\s+)?(?:sworn\s+in|[Tt]akes?\s+oath|took\s+oath)\s+as\s+(?:the\s+)?(?:new\s+)?'

CENTRAL_OFFICES = {
    "prime_minister": {
        "label": "Prime Minister of India",
        "keywords": ["prime minister"],
        "patterns": [
            _NM + r'\s+' + _SW + _PM,
            r'[Nn]ew\s+' + _PM + r'\s+' + _NM,
            _NM + r'\s*,\s*(?:the\s+)?' + _PM + r'\s+of\s+India',
            _PM + r'\s+' + _NM + r'\s+of\s+India',
        ],
    },
    "president": {
        "label": "President of India",
        "keywords": ["president of india", "rashtrapati"],
        "patterns": [
            _NM + r'\s+' + _SW + _PR,
            _NM + r'\s*,\s*(?:the\s+)?' + _PR,
        ],
    },
}

_CAPTURE_STOPWORDS = {'of', 'india', 'the', 'new', 'was', 'at', 'in', 'as',
                      'minister', 'president', 'prime', 'rashtrapati', 'bhavan'}

def _valid_person_capture(person):
    words = person.split()
    if not (2 <= len(words) <= 4):
        return False
    return not any(w.lower().strip('.') in _CAPTURE_STOPWORDS for w in words)

def _lookup_person_party(entities, person):
    groups = (
        entities['india'].get('cabinet_ministers', []) +
        entities['india'].get('opposition_leaders', []) +
        entities['india'].get('state_chief_ministers', []) +
        entities['india'].get('generic_politicians', [])
    )
    for m in groups:
        names = [m.get('name', '')] + m.get('aliases', [])
        if any(n and n.lower() == person.lower() for n in names):
            return m.get('party', '') or None
    return None

def detect_central_government(articles, entities, nlp, llm):
    logging.info("--- Detecting central government (PM / President) ---")
    window_articles = filter_by_window(articles, WINDOW_CM_PARTY_DAYS)

    minister_lookup = build_minister_lookup(entities)
    office_candidates = {office: defaultdict(int) for office in CENTRAL_OFFICES}
    office_contexts   = {office: defaultdict(list) for office in CENTRAL_OFFICES}

    for article in window_articles:
        if article.get('source') in NON_INDIAN_SOURCES:
            continue
        text = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:1200]}"
        text_lower = text.lower()

        # Foreign-leader guard: a US/foreign 'president/PM' story must not vote.
        if re.search(r'\bus president\b|\bu\.s\. president\b|white house|president of the united states|president of (?:pakistan|china|russia|france|sri lanka|nepal|bangladesh)', text_lower):
            continue

        for office, cfg in CENTRAL_OFFICES.items():
            if not any(k in text_lower for k in cfg['keywords']):
                continue
            for pattern in cfg['patterns']:
                for match in re.finditer(pattern, text):
                    full_match_lower = match.group(0).lower()
                    pre_text = text[max(0, match.start()-30):match.start()].lower()
                    if any(w in full_match_lower or w in pre_text for w in ['former', 'ex-', ' ex ', 'late', 'previous', 'past', 'outgoing']):
                        continue
                    person = match.group(1).strip() if match.lastindex >= 1 else None
                    if not person or len(person) < 5 or not _valid_person_capture(person):
                        continue
                    canonical = minister_lookup.get(person.lower())
                    if not canonical:
                        person_lower = person.lower()
                        for alias, name in minister_lookup.items():
                            if re.search(r'\b' + re.escape(alias) + r'\b', person_lower) or \
                               re.search(r'\b' + re.escape(person_lower) + r'\b', alias):
                                canonical = name
                                break
                    # National office: unknown people are review-only, but still
                    # counted under their raw name so the flag carries evidence.
                    key = canonical or person
                    start_idx = max(0, match.start() - 200)
                    end_idx = min(len(text), match.end() + 200)
                    office_candidates[office][key] += 1
                    office_contexts[office][key].append(text[start_idx:end_idx].strip())

    central_updates = []
    central_flags   = []

    for office, candidates in office_candidates.items():
        if not candidates:
            continue
        label          = CENTRAL_OFFICES[office]['label']
        total_mentions = sum(candidates.values())
        top_candidate  = max(candidates, key=candidates.get)
        top_count      = candidates[top_candidate]
        confidence     = top_count / total_mentions if total_mentions > 0 else 0
        contexts       = office_contexts[office][top_candidate]

        current = (entities['india'].get('central_government', {}) or {}).get(office, '')
        if current and current.strip().lower() == top_candidate.strip().lower():
            continue  # no change — nothing to do

        if top_count < CENTRAL_MIN_MENTIONS:
            if top_count >= 3:
                central_flags.append({
                    "type": "central_government", "office": office, "candidate": top_candidate,
                    "confidence": round(confidence, 2), "mentions": top_count,
                    "reason": f"Below {CENTRAL_MIN_MENTIONS}-mention floor for a national office",
                    "context": contexts[0][:200] if contexts else "",
                })
            continue

        known = top_candidate.lower() in minister_lookup or any(
            top_candidate.lower() == v.lower() for v in minister_lookup.values())

        if confidence >= CENTRAL_AUTO_THRESHOLD and known:
            # Majority vote across up to 3 distinct contexts — one lucky
            # sentence must not change the PM of record.
            sample = contexts[:CENTRAL_VALIDATE_CONTEXTS]
            passes = 0
            for ctx in sample:
                ok, _conf = gemma_validate(
                    llm,
                    f"Is '{top_candidate}' explicitly described as currently serving as (or newly sworn in as) the {label} in this text?",
                    ctx
                )
                if ok:
                    passes += 1
            if passes * 2 > len(sample):
                central_updates.append({
                    "office": office, "person": top_candidate,
                    "confidence": round(confidence, 2), "mentions": top_count,
                    "validated_contexts": f"{passes}/{len(sample)}",
                })
                logging.info(f"CENTRAL UPDATE: {label} → {top_candidate} (confidence {confidence:.2f}, {top_count} mentions, {passes}/{len(sample)} contexts validated)")
            else:
                central_flags.append({
                    "type": "central_government", "office": office, "candidate": top_candidate,
                    "confidence": round(confidence, 2), "mentions": top_count,
                    "reason": f"LLM validated only {passes}/{len(sample)} contexts",
                    "context": sample[0][:200] if sample else "",
                })
        else:
            central_flags.append({
                "type": "central_government", "office": office, "candidate": top_candidate,
                "confidence": round(confidence, 2), "mentions": top_count,
                "reason": "Unknown candidate (needs manual add)" if not known else "Below auto-update consensus threshold",
                "context": contexts[0][:200] if contexts else "",
            })

    return central_updates, central_flags


# --- 2c. Union Cabinet Portfolio Changes (reshuffles) ---
PORTFOLIO_MIN_MENTIONS   = 5
PORTFOLIO_AUTO_THRESHOLD = 0.70

_PORTFOLIO = r'([A-Z][A-Za-z]+(?:\s+(?:and\s+|&\s+)?[A-Z][A-Za-z]+){0,3})'

PORTFOLIO_PATTERNS = [
    _NM + r'\s+' + _SW + r'(?:Union\s+)?[Mm]inister\s+(?:of|for)\s+' + _PORTFOLIO,
    _NM + r'\s+takes?\s+charge\s+(?:of|as)\s+(?:the\s+)?(?:Union\s+)?' + _PORTFOLIO + r'\s+[Mm]inist(?:ry|er)',
    _NM + r'\s+(?:was\s+)?appointed\s+(?:as\s+)?(?:the\s+)?(?:new\s+)?(?:Union\s+)?' + _PORTFOLIO + r'\s+[Mm]inister',
    _NM + r'\s+(?:gets|was\s+allocated|was\s+assigned)\s+(?:the\s+)?' + _PORTFOLIO + r'\s+portfolio',
]

def detect_portfolio_changes(articles, entities, nlp, llm):
    """Cabinet reshuffles: an EXISTING minister's role changes. Only people
    already in the entity library are eligible — new faces are the job of
    discover_new_entities. Same evidence-aggregation discipline as CM/PM."""
    logging.info("--- Detecting Union cabinet portfolio changes ---")
    window_articles = filter_by_window(articles, WINDOW_CM_PARTY_DAYS)
    minister_lookup = build_minister_lookup(entities)

    pairs    = defaultdict(int)            # (person, portfolio) -> mentions
    contexts = defaultdict(list)

    for article in window_articles:
        if article.get('source') in NON_INDIAN_SOURCES or article.get('category') == 'international':
            continue
        text = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:1200]}"
        low = text.lower()
        if 'minister' not in low and 'portfolio' not in low:
            continue
        for pattern in PORTFOLIO_PATTERNS:
            for match in re.finditer(pattern, text):
                pre = text[max(0, match.start()-30):match.start()].lower()
                if any(w in match.group(0).lower() or w in pre for w in ['former', 'ex-', ' ex ', 'late', 'previous', 'outgoing']):
                    continue
                person = (match.group(1) or '').strip()
                portfolio = (match.group(2) or '').strip() if match.lastindex >= 2 else ''
                if not person or not portfolio or not _valid_person_capture(person):
                    continue
                # Chief/Prime ministers are handled by their own detectors.
                if portfolio.lower() in ('chief', 'prime', 'union', 'state'):
                    continue
                canonical = minister_lookup.get(person.lower())
                if not canonical:
                    person_lower = person.lower()
                    for alias, name in minister_lookup.items():
                        if re.search(r'\b' + re.escape(alias) + r'\b', person_lower) or \
                           re.search(r'\b' + re.escape(person_lower) + r'\b', alias):
                            canonical = name
                            break
                if not canonical:
                    continue  # unknown people: not this detector's job
                key = (canonical, portfolio.title())
                pairs[key] += 1
                s, e = max(0, match.start()-200), min(len(text), match.end()+200)
                contexts[key].append(text[s:e].strip())

    # Consolidate per person: strongest portfolio must dominate that person's evidence
    per_person = defaultdict(dict)
    for (person, portfolio), count in pairs.items():
        per_person[person][portfolio] = count

    role_updates, role_flags = [], []
    for person, folios in per_person.items():
        total = sum(folios.values())
        top = max(folios, key=folios.get)
        count = folios[top]
        confidence = count / total if total else 0
        if count < PORTFOLIO_MIN_MENTIONS:
            continue
        new_role = f"Union Minister of {top}"
        ctxs = contexts[(person, top)][:2]
        if confidence >= PORTFOLIO_AUTO_THRESHOLD:
            passes = 0
            for ctx in ctxs:
                ok, _c = gemma_validate(
                    llm,
                    f"Does this text explicitly say that '{person}' now holds (was sworn in, appointed, or took charge of) the {top} ministry/portfolio in the Union government?",
                    ctx)
                if ok:
                    passes += 1
            if passes * 2 > len(ctxs):
                role_updates.append({"person": person, "role": new_role,
                                     "confidence": round(confidence, 2), "mentions": count})
                logging.info(f"PORTFOLIO UPDATE: {person} → {new_role} ({count} mentions, conf {confidence:.2f})")
                continue
        role_flags.append({"type": "portfolio_change", "person": person, "candidate_role": new_role,
                           "confidence": round(confidence, 2), "mentions": count,
                           "reason": "Below threshold or LLM validation failed",
                           "context": ctxs[0][:200] if ctxs else ""})
    return role_updates, role_flags


# --- 2d. Enrich auto-added entities (party + safe aliases) ---
COMMON_SURNAMES = {'singh', 'kumar', 'sharma', 'yadav', 'khan', 'patel', 'reddy',
                   'gandhi', 'shah', 'rao', 'das', 'devi', 'verma', 'gupta',
                   'joshi', 'mishra', 'pandey', 'nair', 'menon', 'iyer'}

def enrich_auto_added_entities(articles, entities, llm):
    """Fill in empty party/aliases on auto-discovered politicians using the
    same news evidence that discovered them. Party needs >=3 co-mentions and a
    70% share, validated by the LLM; aliases are mechanical and collision-safe."""
    logging.info("--- Enriching auto-added entities (party/aliases) ---")
    groups = (entities['india'].get('cabinet_ministers', []) +
              entities['india'].get('opposition_leaders', []) +
              entities['india'].get('state_chief_ministers', []) +
              entities['india'].get('generic_politicians', []))
    targets = [m for m in groups if m.get('auto_added') and (not m.get('party') or not m.get('aliases'))]
    if not targets:
        return 0, []

    # Party alias map from the canonical party list
    party_aliases = {}
    for p in entities['india'].get('parties', []):
        for alias in [p.get('name', '')] + p.get('aliases', []):
            if alias:
                party_aliases[alias.lower()] = p.get('name', alias)

    minister_lookup = build_minister_lookup(entities)
    window_articles = filter_by_window(articles, WINDOW_CM_PARTY_DAYS)
    enriched, flags = 0, []

    for m in targets:
        name = m.get('name', '')
        if not name:
            continue
        name_re = re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE)

        # --- Party inference from co-mentions near the name ---
        if not m.get('party'):
            votes = defaultdict(int)
            sample_ctx = {}
            for article in window_articles:
                text = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:1200]}"
                for hit in name_re.finditer(text):
                    nearby = text[max(0, hit.start()-120):hit.end()+120]
                    nearby_lower = nearby.lower()
                    for alias_lower, canonical in party_aliases.items():
                        if len(alias_lower) < 3:
                            continue
                        if re.search((r'\b' + re.escape(alias_lower) + r'\b') if alias_lower.isupper() else r'\b' + re.escape(alias_lower) + r'\b', nearby_lower):
                            votes[canonical] += 1
                            sample_ctx.setdefault(canonical, nearby)
            if votes:
                total = sum(votes.values())
                top = max(votes, key=votes.get)
                share = votes[top] / total
                if votes[top] >= 3 and share >= 0.70:
                    ok, _c = gemma_validate(
                        llm,
                        f"Does this text indicate that '{name}' belongs to or represents the party '{top}'?",
                        sample_ctx.get(top, ''))
                    if ok:
                        m['party'] = top
                        m['party_source'] = 'auto_evidence'
                        enriched += 1
                        logging.info(f"ENRICHED: {name} party → {top} ({votes[top]} co-mentions, {share:.0%} share)")
                    else:
                        flags.append({"type": "entity_enrichment", "person": name, "candidate_party": top,
                                      "reason": "LLM rejected party evidence", "context": sample_ctx.get(top, '')[:200]})
                elif votes[top] >= 2:
                    flags.append({"type": "entity_enrichment", "person": name, "candidate_party": top,
                                  "mentions": votes[top], "share": round(share, 2),
                                  "reason": "Party evidence below auto threshold"})

        # --- Mechanical, collision-safe aliases ---
        if not m.get('aliases'):
            aliases = []
            words = name.split()
            def safe(alias):
                a = alias.lower()
                return (len(a) >= 5 and a != name.lower()
                        and a not in COMMON_SURNAMES
                        and a not in minister_lookup)
            if len(words) >= 3:
                short = f"{words[0]} {words[-1]}"      # drop middle names
                if safe(short):
                    aliases.append(short)
            if len(words) >= 2 and safe(words[-1]):     # bare surname, only if rare
                aliases.append(words[-1])
            if aliases:
                m['aliases'] = aliases
                for a in aliases:
                    minister_lookup[a.lower()] = name
                enriched += 1
                logging.info(f"ENRICHED: {name} aliases → {aliases}")

    return enriched, flags


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
            f"{article.get('content', '')[:1500]}"
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
                      entities['india']['state_chief_ministers'] +
                      entities['india'].get('generic_politicians', []))
    
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

        text       = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:1000]}"
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
                text
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


def extract_controversy_statement(llm, canonical_name, context):
    if llm is None:
        return "Controversial statement reported in the news."

    prompt = f"""<|im_start|>user
Analyze the news text below. Find the controversial statement, unscientific claim, or verbal gaffe directly made by "{canonical_name}".
Extract and rephrase it into a single, highly concise, objective sentence (maximum 15 words) starting with their name.
(Example: "Narendra Modi claimed cloud cover could help jets evade radar.")

Text: {context[:1800]}

Return ONLY a JSON object with this exact field:
{{"statement": "the single concise sentence summary"}}
No explanation. No extra text.
<|im_end|>
<|im_start|>assistant
"""
    try:
        response = llm(
            prompt,
            max_tokens=100,
            temperature=0.1,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False
        )
        raw    = response['choices'][0].get('text', '').strip()
        raw    = re.sub(r'```json|```', '', raw).strip()
        parsed = json.loads(raw)
        return parsed.get('statement', '').strip()
    except Exception as e:
        logging.warning(f"Gemma gaffe extraction failed: {e}")
        return "Controversial remark or gaffe reported in the news."


# --- 5. Controversies and Gaffes Detection ---
def detect_controversies_and_gaffes(articles, entities, nlp, llm):
    logging.info("--- Detecting controversies and verbal gaffes ---")

    known_entities = {}
    all_ministers = (entities['india']['cabinet_ministers'] +
                      entities['india']['opposition_leaders'] +
                      entities['india']['state_chief_ministers'] +
                      entities['india'].get('generic_politicians', []))
    
    for m in all_ministers:
        known_entities[m['name'].lower()] = m['name']
        for alias in m.get('aliases', []):
            known_entities[alias.lower()] = m['name']

    # Load existing validated controversy URLs to skip redundant Gemma runs
    known_controversy_urls = set()
    for m in all_ministers:
        for controversy in m.get('controversies', []):
            if controversy.get('source_url'):
                known_controversy_urls.add(controversy['source_url'].strip())

    entity_controversies = defaultdict(list)

    for article in articles:
        if article.get('category') == 'international':
            continue
        if article.get('source') in NON_INDIAN_SOURCES:
            continue

        # Rely 100% on the semantic AI classifier topic tags (from Repository 3)
        if "political_gaffe" not in article.get('topic_tags', []):
            continue

        text       = f"{article.get('title', '')} {article.get('rephrased_article', '')} {article.get('content', '')[:1000]}"
        text_lower = text.lower()

        for entity_lower, canonical_name in known_entities.items():
            if len(entity_lower) < 4:
                continue
            pattern = r'\b' + re.escape(entity_lower) + r'\b'
            if not re.search(pattern, text_lower):
                continue

            already_logged = any(
                inc['source_url'] == article.get('url', '')
                for inc in entity_controversies[canonical_name]
            )
            if already_logged:
                continue

            url = article.get('url', '').strip()

            if url and url in known_controversy_urls:
                # Find the existing controversy description from entities.json
                existing_text = text[:200]
                for m in all_ministers:
                    for controversy in m.get('controversies', []):
                        if controversy.get('source_url', '').strip() == url:
                            existing_text = controversy.get('incident_text', text[:200])
                            break
                entity_controversies[canonical_name].append({
                    "incident_text": existing_text,
                    "source_url":    url,
                    "source_title":  article.get('title', ''),
                    "scraped_at":    article.get('scraped_at', ''),
                })
                continue

            confirmed, _ = gemma_validate(
                llm,
                f"Does this article specifically report on a controversial statement, verbal gaffe, unscientific claim, or highly mocked/criticized remark directly made by '{canonical_name}'?",
                text
            )

            if confirmed:
                extracted_text = extract_controversy_statement(llm, canonical_name, text)
                entity_controversies[canonical_name].append({
                    "incident_text": extracted_text,
                    "source_url":    url,
                    "source_title":  article.get('title', ''),
                    "scraped_at":    article.get('scraped_at', ''),
                })
                logging.info(
                    f"GAFFE/CONTROVERSY DETECTED: {canonical_name} "
                    f"— {extracted_text[:60]}"
                )

    gaffe_updates = []
    for entity_name, controversies in entity_controversies.items():
        if controversies:
            gaffe_updates.append({
                "entity":    entity_name,
                "incidents": controversies[:10],
            })
            logging.info(f"GAFFE SUMMARY: {entity_name} — {len(controversies)} validated gaffes")

    return gaffe_updates


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
              entities['india']['state_chief_ministers'] +
              entities['india'].get('generic_politicians', [])):
        known_names.add(m['name'].lower())
        for alias in m.get('aliases', []):
            known_names.add(alias.lower())
    for leader in entities['international']['world_leaders']:
        known_names.add(leader['name'].lower())
        for alias in leader.get('aliases', []):
            known_names.add(alias.lower())

    # Blocklist: institutions and parties must never be auto-added as politicians
    blocked_names = set()
    for inst in entities['india'].get('institutions', []):
        n = inst.get('name') if isinstance(inst, dict) else inst
        if n:
            blocked_names.add(str(n).lower())
        if isinstance(inst, dict):
            for alias in inst.get('aliases', []):
                blocked_names.add(alias.lower())
    for p in entities['india'].get('parties', []):
        blocked_names.add(p['name'].lower())
        if p.get('full_name'):
            blocked_names.add(p['full_name'].lower())
        for alias in p.get('aliases', []):
            blocked_names.add(alias.lower())

    # Build party registry lookup dynamically from entities.json
    party_registry = {}
    for p in entities['india'].get('parties', []):
        p_name = p['name']
        party_registry[p_name.lower()] = p_name
        if p.get('full_name'):
            party_registry[p['full_name'].lower()] = p_name
        for alias in p.get('aliases', []):
            party_registry[alias.lower()] = p_name

    # Build valid states set dynamically from entities.json
    valid_states = {s['name'].lower() for s in entities['india'].get('states', [])}

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
                if name.lower() in blocked_names:
                    continue
                if name.lower() not in known_names:
                    person_counter[name] += 1
                    if len(person_contexts[name]) < 5:
                        person_contexts[name].append(text)

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

    valid_categories = {'cabinet_minister', 'state_chief_minister', 'opposition_leader', 'generic_politician'}
    candidates_processed = []

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

        extraction_prompt = f"""<|im_start|>user
You are extracting a profile for ONE specific person named "{name}".
Use ONLY facts about {name} themselves — never facts about people they meet, criticise,
attack, succeed, praise, or merely refer to.

CRITICAL RULES:
1. "role" must be {name}'s OWN current position. If the text only states SOMEONE ELSE'S role
   (e.g. "{name} attacked the former Chief Minister"), that role belongs to the other person —
   do NOT assign it to {name}.
2. Never record a "former"/"ex" title as the current role.
3. "role_evidence" must be the exact phrase from the article that states {name}'s OWN role,
   and it must contain {name}'s name. If no such phrase exists, set role AND role_evidence to "".
4. Only judge {name}: are they an Indian politician?

Example Trap Avoidance:
Article text: "BJP Union Minister Pabitra Margherita attacked the former Chief Minister of Assam, Tarun Gogoi."
Extraction for "Pabitra Margherita":
- "role": "Union Minister" (Not "Former Chief Minister of Assam")
- "role_evidence": "BJP Union Minister Pabitra Margherita" (the phrase showing their own role)

Article excerpts: {context_sample[:1800]}

Return ONLY this JSON (no extra text):
{{
  "is_indian_politician": true or false,
  "full_name": "official full name",
  "role": "{name}'s own current role, or '' if not clearly stated",
  "role_evidence": "exact phrase from the article naming {name} and their role, or ''",
  "party": "party abbreviation as written in the article, or '' if not stated",
  "state": "home state, or null if national-level",
  "category": "cabinet_minister" or "state_chief_minister" or "opposition_leader" or "generic_politician",
  "confidence": "high" or "medium" or "low"
}}
<|im_end|>
<|im_start|>assistant
"""
        try:
            response = llm(
                extraction_prompt,
                max_tokens=200,
                temperature=0.0,
                stop=["<|im_end|>", "<|im_start|>"],
                echo=False
            )
            raw    = response['choices'][0].get('text', '').strip()
            raw    = re.sub(r'```json|```', '', raw).strip()
            parsed = json.loads(raw)

            if not parsed.get('is_indian_politician'):
                logging.info(f"SKIP (not Indian politician): {name}")
                continue

            full_name     = (parsed.get('full_name') or name).strip()
            role          = (parsed.get('role') or '').strip()
            role_evidence = (parsed.get('role_evidence') or '').strip()
            party         = (parsed.get('party') or '').strip()
            state         = parsed.get('state') or None
            category      = (parsed.get('category') or '').strip()
            confidence    = (parsed.get('confidence') or 'low').strip()

            # Clean/normalize helper for role evidence backstop
            def normalize_text(t):
                if not t:
                    return ""
                return re.sub(r'[^a-z0-9]', '', t.lower())

            norm_evidence  = normalize_text(role_evidence)
            norm_name      = normalize_text(name)
            norm_full_name = normalize_text(full_name)
            norm_context   = normalize_text(context_sample)

            is_evidence_valid = False
            if norm_evidence:
                contains_name = (norm_name in norm_evidence) or (norm_full_name in norm_evidence)
                in_context    = norm_evidence in norm_context
                if contains_name and in_context:
                    is_evidence_valid = True

            if not is_evidence_valid:
                logging.info(f"Role evidence validation failed for {full_name} (evidence: '{role_evidence}'). Discarding role.")
                role = ""
                role_evidence = ""

            # Drop former/ex roles
            if role:
                role_lower = role.lower()
                if "former" in role_lower or "ex-" in role_lower or "ex " in role_lower:
                    logging.info(f"Former/ex role detected for {full_name} ('{role}'). Discarding role.")
                    role = ""
                    role_evidence = ""

            # Force all auto-added entities to generic_politician category
            category = 'generic_politician'

            # Dynamically derive ruling coalition
            ruling_party_val = ''
            ruling_coalition_val = ''
            # Fallback for central_government path lookup
            if 'central_government' not in entities['india'] and 'central_govt' in entities['india']:
                ruling_party_val = entities['india']['central_govt'].get('ruling_party', '').strip().lower()
                ruling_coalition_val = entities['india']['central_govt'].get('ruling_coalition', '').strip().lower()
            elif 'central_government' in entities['india']:
                ruling_party_val = entities['india']['central_government'].get('ruling_party', '').strip().lower()
                ruling_coalition_val = entities['india']['central_government'].get('ruling_coalition', '').strip().lower()
                
            union_coalition = {ruling_party_val, ruling_coalition_val}
            for p in entities['india'].get('parties', []):
                p_coalition = p.get('coalition', '').strip().lower()
                if p_coalition and p_coalition == ruling_coalition_val:
                    union_coalition.add(p['name'].lower())
                    for alias in p.get('aliases', []):
                        union_coalition.add(alias.lower())

            # Party verification check
            is_recognized_party = False
            resolved_party = ""
            if party:
                party_lower = party.strip().lower()
                if party_lower in party_registry:
                    resolved_party = party_registry[party_lower]
                    is_recognized_party = True
                else:
                    resolved_party = party.strip()
            else:
                resolved_party = "unconfirmed"

            # Parse and canonicalize state if valid
            has_valid_state = False
            if state:
                state_lower = state.strip().lower()
                if state_lower in valid_states:
                    has_valid_state = True
                    canonical_state = next(s['name'] for s in entities['india']['states'] if s['name'].lower() == state_lower)
                    state = canonical_state

            new_entry = {
                "name":                   full_name,
                "role":                   role,
                "party":                  resolved_party,
                "state":                  state,
                "aliases":                [name] if name != full_name else [],
                "criminal_cases":         0,
                "criminal_cases_in_news": 0,
                "affidavit_url":          "https://affidavit.eci.gov.in",
                "wikipedia":              "", # keep empty per user requirement
                "image_placeholder":      full_name.lower().replace(' ', '_').replace('.', ''),
                "auto_added":             True,
                "auto_added_on":          str(datetime.now().date()),
                "gemma_confidence":       confidence,
                "mentions_detected":      count,
                "party_verified":         is_recognized_party,
            }
            if role_evidence:
                new_entry["role_evidence"] = role_evidence

            candidates_processed.append({
                "profile": new_entry,
                "confidence": confidence,
                "count": count,
                "raw_name": name,
                "has_valid_state": has_valid_state,
                "is_recognized_party": is_recognized_party,
                "raw_party": resolved_party,
                "contexts": person_contexts[name]
            })

        except Exception as e:
            logging.warning(f"Gemma extraction failed for '{name}': {e}. Flagging for review.")
            flagged.append({
                "name":            name,
                "mentions":        count,
                "reason":          f"Gemma extraction error: {e}",
                "sample_articles": person_contexts[name][:2],
            })

    # Build a set of all lowercase names and aliases of all existing parties in entities.json for deduplication
    existing_party_aliases = set()
    for p in entities['india'].get('parties', []):
        existing_party_aliases.add(p['name'].lower())
        if p.get('full_name'):
            existing_party_aliases.add(p['full_name'].lower())
        for alias in p.get('aliases', []):
            existing_party_aliases.add(alias.lower())

    # Tally occurrences of unrecognized parties in current discovery run
    unrecognized_party_counts = Counter()
    unrecognized_party_candidates = defaultdict(list)

    for c in candidates_processed:
        party_val = c["raw_party"]
        if party_val != "unconfirmed" and not c["is_recognized_party"]:
            party_upper = party_val.upper()
            unrecognized_party_counts[party_upper] += 1
            unrecognized_party_candidates[party_upper].append(c)

    # Process unrecognized parties and auto-promote if they qualify
    for party_upper, candidates in unrecognized_party_candidates.items():
        count_seen = len(candidates)
        if count_seen >= 3:
            # Check Indian context: at least one candidate carrying it must have a valid Indian state
            has_indian_state = any(c["has_valid_state"] for c in candidates)
            
            # Check shape: 2-8 char uppercase abbreviation OR contains standard Indian political party suffix/token
            is_abbrev = re.match(r'^[A-Z0-9\(\)\-&]{2,8}$', party_upper) is not None
            p_lower = party_upper.lower()
            suffix_tokens = ["party", "dal", "sena", "katchi", "kazhagam", "congress", "morcha", "samaj", "union", "front", "parishad", "league", "movement"]
            has_suffix = any(token in p_lower for token in suffix_tokens)
            has_valid_shape = is_abbrev or has_suffix
            
            # Deduplication against existing aliases
            is_duplicate = p_lower in existing_party_aliases
            
            if has_indian_state and has_valid_shape and not is_duplicate:
                logging.info(f"Auto-promoting party '{party_upper}' to registry (seen {count_seen} times)")
                new_party_obj = {
                    "name": party_upper,
                    "full_name": party_upper,
                    "aliases": [],
                    "ideology": "Regionalism, state rights",
                    "founded": "Unknown",
                    "president": "Unknown",
                    "coalition": "Unknown",
                    "ruling_states": [],
                    "wikipedia": "",
                    "color": "#808080",
                    "auto_added": True
                }
                entities['india']['parties'].append(new_party_obj)
                existing_party_aliases.add(p_lower)
                
                # Mark all candidates of this party as verified
                for c in candidates:
                    c["profile"]["party_verified"] = True
                    c["passed_backstop_override"] = True
            else:
                logging.info(f"Party '{party_upper}' did not qualify for auto-promotion. "
                             f"State context: {has_indian_state}, Shape: {has_valid_shape}, Duplicate: {is_duplicate}")
                flagged.append({
                    "type": "unrecognized_party",
                    "party": party_upper,
                    "mentions": count_seen,
                    "reason": f"Failed auto-promotion. Indian state context: {has_indian_state}, valid shape: {has_valid_shape}, duplicate: {is_duplicate}",
                    "sample_candidates": [c["profile"]["name"] for c in candidates]
                })
        else:
            logging.info(f"Party '{party_upper}' seen {count_seen} times (< 3). Flagging party.")
            flagged.append({
                "type": "unrecognized_party",
                "party": party_upper,
                "mentions": count_seen,
                "reason": "Seen less than 3 times in this run",
                "sample_candidates": [c["profile"]["name"] for c in candidates]
            })

    # Now filter candidates into auto_added and flagged lists based on final backstop check
    for c in candidates_processed:
        profile = c["profile"]
        confidence = c["confidence"]
        count = c["count"]
        raw_name = c["raw_name"]
        has_valid_state = c["has_valid_state"]
        is_recognized_party = c["is_recognized_party"]
        passed_backstop = (
            is_recognized_party or 
            has_valid_state or 
            c.get("passed_backstop_override", False)
        )
        
        if not passed_backstop:
            logging.info(f"Nationality backstop failed for {profile['name']} (party: {profile['party']}, state: {profile['state']}). Flagging for review.")
            flagged.append({
                "name":            profile['name'],
                "raw_name":        raw_name,
                "mentions":        count,
                "role":            profile['role'],
                "party":           profile['party'],
                "state":           profile['state'],
                "category":        profile['category'],
                "confidence":      confidence,
                "reason":          "Nationality backstop failed: unrecognized party and state/UT",
                "sample_articles": c["contexts"][:2],
            })
            continue

        if profile['name'].lower() in blocked_names or raw_name.lower() in blocked_names:
            logging.info(f"SKIP (institution/party, not a person): {profile['name']}")
            continue

        if profile['name'].lower() in known_names:
            logging.info(f"SKIP (already known under full name): {profile['name']}")
            continue

        if confidence in ('high', 'medium'):
            # Auto-add directly into the live entities dict
            category = profile['category']
            entities['india'][f'{category}s'].append(profile)
            dest_str = f"entities['india']['{category}s']"

            known_names.add(profile['name'].lower())
            known_names.add(raw_name.lower())

            auto_added.append({
                "name":       profile['name'],
                "category":   category,
                "confidence": confidence,
                "mentions":   count,
                "role":       profile['role'],
                "party":      profile['party'],
            })
            logging.info(
                f"AUTO-ADDED [{confidence}]: {profile['name']} ({profile['role']}, {profile['party']}) "
                f"→ {dest_str}"
            )
        else:
            flagged.append({
                "name":            profile['name'],
                "raw_name":        raw_name,
                "mentions":        count,
                "role":            profile['role'],
                "party":           profile['party'],
                "category":        profile['category'],
                "confidence":      confidence,
                "reason":          "Gemma confidence too low for auto-add",
                "sample_articles": c["contexts"][:2],
            })
            logging.info(f"FLAGGED FOR REVIEW (low confidence): {profile['name']}")

    logging.info(
        f"New entities: {len(auto_added)} auto-added, {len(flagged)} flagged for review."
    )
    return auto_added, flagged

# ==============================================================================
# --- APPLY UPDATES TO entities.json ---
# ==============================================================================

def enforce_database_consistency(entities):
    logging.info("--- Enforcing Database Consistency & Referential Integrity ---")
    
    cms = entities['india']['state_chief_ministers']
    party_list = entities['india']['parties']
    states = entities['india']['states']
    
    # Build party lookup map (resolves aliases like "Congress" -> "INC")
    party_lookup = {}
    for p in party_list:
        p_name = p['name']
        party_lookup[p_name.upper()] = p_name
        for alias in p.get('aliases', []):
            party_lookup[alias.upper()] = p_name

    def resolve_party_name(party_raw):
        if not party_raw:
            return ""
        party_raw_upper = party_raw.strip().upper()
        return party_lookup.get(party_raw_upper, party_raw.strip())

    def matches_cm(profile, auth_cm):
        if not auth_cm or auth_cm == "N/A":
            return False
        
        def clean(n):
            return re.sub(r'[^a-z0-9]', '', n.lower())
        
        auth_cm_clean = clean(auth_cm)
        if clean(profile.get('name', '')) == auth_cm_clean:
            return True
        
        for alias in profile.get('aliases', []):
            if clean(alias) == auth_cm_clean:
                return True
                
        # Punctuation/space normalized whole-word match fallback
        # e.g., 'Stalin' resolves to 'M. K. Stalin' or 'M.K. Stalin'
        def norm_words(n):
            return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', n.lower().replace('.', ' '))).strip()
            
        p_norm = norm_words(profile.get('name', ''))
        auth_norm = norm_words(auth_cm)
        
        if p_norm and auth_norm:
            if re.search(r'\b' + re.escape(auth_norm) + r'\b', p_norm) or \
               re.search(r'\b' + re.escape(p_norm) + r'\b', auth_norm):
                return True
                
        return False

    # Build CM profile resolver helper (checks name and aliases in both lists)
    def find_cm_profile(cm_name):
        if not cm_name or cm_name == "N/A":
            return None
        # Check in state_chief_ministers first
        for profile in cms:
            if matches_cm(profile, cm_name):
                return profile
        # Check in generic_politicians next
        for profile in entities['india'].get('generic_politicians', []):
            if matches_cm(profile, cm_name):
                return profile
        return None

    # 1. Synchronize State CMs with profiles and demote former/non-matching CMs
    active_cms_by_state = {}
    for state in states:
        state_name = state['name']
        active_cm = state.get('cm')
        if active_cm and active_cm.strip() and active_cm.strip() != "N/A":
            active_cms_by_state[state_name.lower()] = active_cm.strip()

    cms_to_keep = []
    cms_to_demote = []
    matched_states = set()
    promoted_profiles = set()

    # Resolve active CM profiles for states that have them
    for state in states:
        state_name = state['name']
        state_lower = state_name.lower()
        if state_lower in active_cms_by_state:
            auth_cm = active_cms_by_state[state_lower]
            profile = find_cm_profile(auth_cm)
            if profile:
                profile['role'] = f"Chief Minister of {state_name}"
                profile['state'] = state_name
                # Remove "Former" or "Former Chief Minister" from role if it was there
                if "Former" in profile['role']:
                    profile['role'] = profile['role'].replace("Former Chief Minister", "Chief Minister").replace("Former ", "")
                
                cms_to_keep.append(profile)
                matched_states.add(state_lower)
                promoted_profiles.add(id(profile))
                
                # If it was in generic_politicians, remove it
                if 'generic_politicians' in entities['india']:
                    entities['india']['generic_politicians'] = [gp for gp in entities['india']['generic_politicians'] if id(gp) != id(profile)]
            else:
                logging.warning(f"Active CM '{auth_cm}' for state {state_name} has no profile. Creating basic profile.")
                ruling_party = resolve_party_name(state.get('ruling_party'))
                new_profile = {
                    "name": auth_cm,
                    "role": f"Chief Minister of {state_name}",
                    "party": ruling_party or "",
                    "state": state_name,
                    "aliases": [],
                    "criminal_cases": 0,
                    "criminal_cases_in_news": 0,
                    "affidavit_url": "https://affidavit.eci.gov.in",
                    "wikipedia": f"https://en.wikipedia.org/wiki/{auth_cm.replace(' ', '_')}",
                    "image_placeholder": auth_cm.lower().replace(' ', '_').replace('.', ''),
                    "auto_added": True,
                    "auto_added_on": str(datetime.now().date()),
                    "gemma_confidence": "high",
                    "criminal_incidents": [],
                    "controversies": []
                }
                cms_to_keep.append(new_profile)
                matched_states.add(state_lower)

    # For states with empty states[].cm, keep their existing CM profiles intact
    for state in states:
        state_name = state['name']
        state_lower = state_name.lower()
        if state_lower not in active_cms_by_state:
            for profile in cms:
                if profile.get('state') == state_name:
                    logging.warning(f"Guard triggered: state '{state_name}' has no CM in states[].cm. Leaving existing CM entry '{profile.get('name')}' intact.")
                    cms_to_keep.append(profile)
                    promoted_profiles.add(id(profile))

    # Demote all remaining CM profiles that were not matched/promoted
    for profile in cms:
        if id(profile) not in promoted_profiles:
            cms_to_demote.append(profile)

    # Sync state fields (cm, ruling_party) for keeping profiles
    for state in states:
        state_name = state['name']
        state_lower = state_name.lower()
        if state_lower in active_cms_by_state:
            auth_cm = active_cms_by_state[state_lower]
            profile = next((p for p in cms_to_keep if p.get('state') == state_name and matches_cm(p, auth_cm)), None)
            if profile:
                state['cm'] = profile['name']
                ruling_party = resolve_party_name(state.get('ruling_party'))
                cm_party = resolve_party_name(profile.get('party'))
                if cm_party and cm_party != ruling_party:
                    logging.info(f"Syncing state {state_name} ruling_party '{ruling_party}' to match CM's party '{cm_party}'")
                    state['ruling_party'] = cm_party
                elif ruling_party and not cm_party:
                    profile['party'] = ruling_party

    # Apply demotions: modify role, remove from state_chief_ministers, and append to generic_politicians
    if 'generic_politicians' not in entities['india']:
        entities['india']['generic_politicians'] = []

    for profile in cms_to_demote:
        role = profile.get('role', '')
        if "Chief Minister" in role and "Former" not in role:
            profile['role'] = role.replace("Chief Minister", "Former Chief Minister")
        elif not role.lower().startswith("former"):
            profile['role'] = f"Former {role}"
            
        if not any(g['name'].lower() == profile['name'].lower() for g in entities['india']['generic_politicians']):
            entities['india']['generic_politicians'].append(profile)

    # Overwrite the state_chief_ministers list
    entities['india']['state_chief_ministers'] = cms_to_keep

    # 2. Recompute Party ruling_states dynamically
    party_map = {p['name'].upper(): p for p in party_list}
    
    # Initialize all party ruling states to empty
    for party in party_list:
        party['ruling_states'] = []
        
    for state in states:
        state_name = state['name']
        ruling_party = resolve_party_name(state.get('ruling_party'))
        if ruling_party:
            ruling_party_upper = ruling_party.upper()
            if ruling_party_upper in party_map:
                party_map[ruling_party_upper]['ruling_states'].append(state_name)
            else:
                logging.info(f"State {state_name} is ruled by party '{ruling_party}' which is not in parties list. Appending basic party entry.")
                new_party = {
                    "name": ruling_party,
                    "full_name": ruling_party,
                    "aliases": [],
                    "ideology": "Regionalism",
                    "founded": "Unknown",
                    "president": "Unknown",
                    "coalition": "Unknown",
                    "ruling_states": [state_name],
                    "wikipedia": f"https://en.wikipedia.org/wiki/{ruling_party}",
                    "color": "#808080"
                }
                party_list.append(new_party)
                party_map[ruling_party_upper] = new_party
                
    # 3. Validate and normalize party fields for all auto-added politicians
    all_categories = ['cabinet_ministers', 'opposition_leaders', 'state_chief_ministers', 'generic_politicians']
    for cat in all_categories:
        for p in entities['india'].get(cat, []):
            if p.get('auto_added'):
                party = p.get('party')
                if party:
                    party_lower = party.strip().lower()
                    party_upper = party_lower.upper()
                    if party_upper in party_lookup:
                        p['party'] = party_lookup[party_upper]
                    else:
                        p['party'] = "unconfirmed"

    logging.info("--- Referential Integrity Enforced successfully ---")


def apply_updates(entities, cm_updates, party_updates, criminal_updates, new_promises, gaffe_updates=None, central_updates=None, role_updates=None):
    for update in (role_updates or []):
        groups = (entities['india'].get('cabinet_ministers', []) +
                  entities['india'].get('opposition_leaders', []) +
                  entities['india'].get('state_chief_ministers', []) +
                  entities['india'].get('generic_politicians', []))
        for m in groups:
            if m.get('name', '').lower() == update['person'].lower():
                old_role = m.get('role', 'Unknown')
                m['role'] = update['role']
                m['role_confidence']   = update['confidence']
                m['role_last_updated'] = str(datetime.now().date())
                if old_role != update['role']:
                    logging.info(f"APPLIED ROLE UPDATE: {update['person']} — {old_role} → {update['role']}")
                break

    for update in (central_updates or []):
        cg = entities['india'].setdefault('central_government', {})
        office = update['office']
        old = cg.get(office, 'Unknown')
        cg[office] = update['person']
        cg[f'{office}_confidence']   = update['confidence']
        cg[f'{office}_last_updated'] = str(datetime.now().date())
        if old != update['person']:
            logging.info(f"APPLIED CENTRAL UPDATE: {office} — {old} → {update['person']}")
        # A new PM implies the ruling party at the Centre follows their party.
        if office == 'prime_minister':
            pm_party = _lookup_person_party(entities, update['person'])
            if pm_party:
                old_rp = cg.get('ruling_party', 'Unknown')
                cg['ruling_party'] = pm_party
                cg['ruling_party_last_updated'] = str(datetime.now().date())
                if old_rp != pm_party:
                    logging.info(f"APPLIED CENTRAL UPDATE: ruling_party — {old_rp} → {pm_party} (derived from new PM)")
            else:
                logging.warning("New PM applied but party unknown — ruling_party left unchanged, review recommended.")

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
            entities['india']['state_chief_ministers'] +
            entities['india'].get('generic_politicians', [])
        )
        for minister in all_ministers:
            if minister['name'] == entity_name:
                # MERGE by source_url — never wipe previously recorded history
                existing = minister.get('criminal_incidents', []) or []
                known_urls = {i.get('source_url') for i in existing}
                for inc in update['incidents']:
                    if inc.get('source_url') not in known_urls:
                        existing.append(inc)
                        known_urls.add(inc.get('source_url'))
                minister['criminal_incidents']     = existing
                minister['criminal_cases_in_news'] = len(existing)
                minister['criminal_last_updated']  = str(datetime.now().date())
                break

    if gaffe_updates:
        all_ministers = (
            entities['india']['cabinet_ministers'] +
            entities['india']['opposition_leaders'] +
            entities['india']['state_chief_ministers'] +
            entities['india'].get('generic_politicians', [])
        )
        for update in gaffe_updates:
            entity_name = update['entity']
            for minister in all_ministers:
                if minister['name'] == entity_name:
                    # MERGE by source_url — preserve controversy history
                    existing = minister.get('controversies', []) or []
                    known_urls = {i.get('source_url') for i in existing}
                    for inc in update['incidents']:
                        if inc.get('source_url') not in known_urls:
                            existing.append(inc)
                            known_urls.add(inc.get('source_url'))
                    minister['controversies'] = existing
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
        "gaffe_updates":   len(gaffe_updates) if gaffe_updates else 0,
    }

    # Enforce referential consistency and dynamic fields
    enforce_database_consistency(entities)

    return entities

# ==============================================================================
# --- MAIN ---
# ==============================================================================

def main():
    import sys
    args = sys.argv[1:]
    
    # Process mode options: "--process", "--commit-sheet", "--only-consistency",
    # "--benchmark-threads", or default (both)
    mode = "both"
    if "--benchmark-threads" in args:
        logging.info("--- Thread benchmark only ---")
        benchmark_threads()
        return
    if "--process" in args:
        mode = "process"
    elif "--commit-sheet" in args:
        mode = "commit-sheet"
    elif "--only-consistency" in args:
        mode = "only-consistency"
        logging.info("--- Running Consistency Enforcer Only ---")
        entities = load_entities()
        entities = dedupe_entities(entities)
        enforce_database_consistency(entities)
        with open(ENTITIES_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(entities, f, indent=2, ensure_ascii=False)
            f.write("\n")
        logging.info("Saved consistency-enforced entities.json")
        return

    start_time = time.time()
    logging.info(f"--- Satya Entity Updater Started (Mode: {mode}) ---")

    try:
        conn = get_db_connection()
    except Exception as e:
        logging.critical(f"Failed to connect to database: {e}")
        sys.exit(1)

    if mode == "commit-sheet":
        pending_file = './pending_sheet_updates.json'
        if not os.path.exists(pending_file):
            logging.info("No pending Database updates found. Exiting.")
            conn.close()
            return
        try:
            with open(pending_file, 'r') as f:
                batch_updates = json.load(f)
            logging.info(f"Loaded {len(batch_updates)} pending updates from disk.")
            conn = commit_sheet_updates(conn, batch_updates)
            # Remove the file so we don't apply it again
            os.remove(pending_file)
            logging.info("Removed pending_sheet_updates.json")
        except Exception as e:
            logging.error(f"Failed to commit database updates: {e}")
            conn.close()
            sys.exit(1)
        conn.close()
        return

    # Else: process or both
    articles = fetch_articles(conn)

    if not articles:
        logging.error("No articles found. Exiting.")
        conn.close()
        return

    entities = load_entities()
    entities = dedupe_entities(entities)
    logging.info(f"Loaded entities.json (version: {entities['metadata'].get('version', 'unknown')})")

    nlp = load_spacy()
    if nlp is None:
        logging.error("spaCy failed to load. Exiting.")
        conn.close()
        return

    # Gemma loaded first — needed by both entity discovery and canonicalization
    llm = load_gemma()

    # Step 1: Separate all fetched articles into unprocessed and processed.
    # An article is re-queued (even if previously canonicalized) when it still
    # contains a non-canonical name that the now-grown entity library CAN resolve —
    # so names unresolvable in the past get fixed as the library learns.
    canonical_set_now = build_canonical_minister_set(entities)

    def needs_canonicalization(a):
        if not a.get('ministers_canonicalized'):
            return True
        for n in (a.get('ministers_mentioned') or []):
            if n not in canonical_set_now and find_canonical_candidates(n, canonical_set_now):
                return True
        return False

    unprocessed_articles = [a for a in articles if needs_canonicalization(a)]
    processed_articles   = [a for a in articles if not needs_canonicalization(a)]
    
    unprocessed_chunk = unprocessed_articles[:MAX_ARTICLES_PER_RUN]
    logging.info(
        f"Partitioned articles: {len(processed_articles)} already processed, "
        f"{len(unprocessed_articles)} unprocessed. Selected a chunk of "
        f"{len(unprocessed_chunk)} unprocessed articles to run."
    )

    new_entities_added   = []
    new_entities_flagged = []
    new_promises         = []
    criminal_updates     = []
    gaffe_updates        = []
    all_flags            = []
    batch_updates        = []

    if unprocessed_chunk:
        # Step 1.1: Discover and auto-add new entities into the live entities dict.
        # Must run BEFORE canonicalize so newly added ministers are available
        # for name resolution in the same weekly run.
        new_entities_added, new_entities_flagged = discover_new_entities(
            unprocessed_chunk, entities, nlp, llm
        )

        # Step 2: Canonicalise ministers_mentioned across the unprocessed chunk in memory.
        # Returns batch updates list.
        batch_updates = canonicalize_ministers_in_sheet(conn, unprocessed_chunk, entities, llm)

        # Step 3: Run promise extraction on newly canonicalized chunk
        new_promises = extract_promises(unprocessed_chunk, entities, nlp, llm)

    # Step 3.5: Criminal/controversy detection.
    # Normally scans only the new chunk (incidents now MERGE, so history persists).
    #
    # Full-history rescan (recovers incidents lost to the old replace bug):
    #   - Started by setting RESCAN_HISTORY=true on any run.
    #   - Runs in chunks of RESCAN_CHUNK_SIZE articles to stay inside CI timeouts.
    #   - Progress is checkpointed in entities.json metadata, so subsequent runs
    #     (triggered by the workflow's existing has_more self-loop, or the next
    #     scheduled run) resume automatically until complete — no manual babysitting.
    RESCAN_CHUNK_SIZE = int(os.environ.get('RESCAN_CHUNK_SIZE', 2000))
    rescan_requested = os.environ.get('RESCAN_HISTORY', '').lower() in ('1', 'true', 'yes')
    rescan_state = entities['metadata'].get('rescan_history', {})

    if rescan_requested and not rescan_state.get('in_progress'):
        rescan_state = {'in_progress': True, 'next_index': 0, 'started_on': str(datetime.now().date())}
        logging.info("RESCAN_HISTORY: starting full-history rescan from article 0.")

    rescan_active = rescan_state.get('in_progress', False)
    rescan_has_more = False

    if rescan_active:
        start = int(rescan_state.get('next_index', 0))
        end = min(start + RESCAN_CHUNK_SIZE, len(articles))
        crime_scan_articles = articles[start:end]
        logging.info(f"RESCAN_HISTORY: scanning articles {start}..{end} of {len(articles)}.")
        if end >= len(articles):
            rescan_state = {'in_progress': False, 'completed_on': str(datetime.now().date()),
                            'articles_scanned': len(articles)}
            logging.info("RESCAN_HISTORY: final chunk — rescan complete after this run.")
        else:
            rescan_state['next_index'] = end
            rescan_has_more = True
        entities['metadata']['rescan_history'] = rescan_state
    else:
        crime_scan_articles = unprocessed_chunk

    if crime_scan_articles:
        criminal_updates = detect_criminal_cases(crime_scan_articles, entities, nlp, llm)
        gaffe_updates    = detect_controversies_and_gaffes(crime_scan_articles, entities, nlp, llm)

    # Step 4: Run CM and ruling party aggregations using both historical and chunk articles
    # Combine processed articles with the newly processed chunk for correct aggregate confidence calculation
    cm_party_articles = processed_articles + unprocessed_chunk
    cm_updates, cm_flags = detect_cms(cm_party_articles, entities, nlp, llm)
    all_flags.extend(cm_flags)

    party_updates, party_flags = detect_ruling_parties(cm_party_articles, entities, nlp, llm)
    all_flags.extend(party_flags)

    central_updates, central_flags = detect_central_government(cm_party_articles, entities, nlp, llm)
    all_flags.extend(central_flags)

    role_updates, role_flags = detect_portfolio_changes(cm_party_articles, entities, nlp, llm)
    all_flags.extend(role_flags)

    enriched_count, enrich_flags = enrich_auto_added_entities(cm_party_articles, entities, llm)
    all_flags.extend(enrich_flags)

    updated_entities = apply_updates(
        entities, cm_updates, party_updates, criminal_updates, new_promises, gaffe_updates,
        central_updates=central_updates, role_updates=role_updates
    )

    with open(ENTITIES_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(updated_entities, f, indent=2, ensure_ascii=False)
    logging.info("Saved updated entities.json")

    review_output = {
        "generated_at": str(datetime.now()),
        "summary": {
            "cm_updates_applied":        len(cm_updates),
            "central_updates_applied":   len(central_updates),
            "role_updates_applied":      len(role_updates),
            "entities_enriched":         enriched_count,
            "party_updates_applied":     len(party_updates),
            "criminal_updates_applied":  len(criminal_updates),
            "gaffe_updates_applied":     len(gaffe_updates),
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

    # Save pending sheet updates to a local disk file in 'process' mode
    if mode == "process":
        pending_file = './pending_sheet_updates.json'
        with open(pending_file, 'w', encoding='utf-8') as f:
            json.dump(batch_updates, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved {len(batch_updates)} pending Database updates to {pending_file}")

        # Write GITHUB_OUTPUT for self-triggering recursive loops in GitHub Actions
        github_output = os.environ.get('GITHUB_OUTPUT')
        if github_output:
            try:
                with open(github_output, 'a') as f:
                    if len(unprocessed_articles) > MAX_ARTICLES_PER_RUN or rescan_has_more:
                        f.write("has_more=true\n")
                        logging.info("Set GITHUB_OUTPUT: has_more=true")
                    else:
                        f.write("has_more=false\n")
                        logging.info("Set GITHUB_OUTPUT: has_more=false")
            except Exception as e:
                logging.warning(f"Failed to write GITHUB_OUTPUT: {e}")
    elif mode == "both":
        # Commit immediately for manual/local testing
        conn = commit_sheet_updates(conn, batch_updates)

    conn.close()
    elapsed = round(time.time() - start_time, 2)
    logging.info(f"--- Entity Updater Finished in {elapsed}s ---")
    logging.info(f"Summary: {review_output['summary']}")
    print(json.dumps(review_output['summary'], indent=2))


if __name__ == '__main__':
    main()
