import sqlite3
import json
import sys
import os

# Adjust path so we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entity_updater import refresh_bridge_rows, slugify, party_slugify

def setup_test_db():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    # Create mock articles table
    cursor.execute("""
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY,
            title TEXT,
            ministers_mentioned TEXT,
            party_mentioned TEXT,
            states_mentioned TEXT,
            status TEXT
        );
    """)
    
    # Create mock article_entities bridge table
    cursor.execute("""
        CREATE TABLE article_entities (
            article_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            slug TEXT NOT NULL,
            PRIMARY KEY (article_id, kind, slug)
        );
    """)
    
    # Insert Test Articles
    cursor.execute("""
        INSERT INTO articles (id, title, ministers_mentioned, party_mentioned, states_mentioned, status)
        VALUES (1, 'Test Article 1', '["Narendra Modi", "Amit Shah"]', '["BJP"]', '["Delhi"]', 'classified');
    """)
    cursor.execute("""
        INSERT INTO articles (id, title, ministers_mentioned, party_mentioned, states_mentioned, status)
        VALUES (2, 'Control Article 2', '["Nitin Gadkari"]', '["BJP"]', '["Maharashtra"]', 'classified');
    """)
    
    # Insert initial bridge rows
    initial_entities = [
        (1, 'minister', 'narendra_modi'),
        (1, 'minister', 'amit_shah'),
        (1, 'party', 'bjp'),
        (1, 'state', 'delhi'),
        (2, 'minister', 'nitin_gadkari'),
        (2, 'party', 'bjp'),
        (2, 'state', 'maharashtra')
    ]
    for row in initial_entities:
        cursor.execute("INSERT INTO article_entities VALUES (?, ?, ?)", row)
        
    conn.commit()
    return conn

def run_tests():
    conn = setup_test_db()
    cursor = conn.cursor()
    
    print("Running bridge table synchronization unit tests...")
    
    # ----------------------------------------------------
    # TEST 1: Updating ministers list updates only ministers kind for this article
    # ----------------------------------------------------
    new_ministers = ["Narendra Modi", "Rajnath Singh"]
    refresh_bridge_rows(cursor, 1, 'minister', new_ministers, slugify)
    conn.commit()
    
    # Assert minister slugs for article 1 updated
    cursor.execute("SELECT slug FROM article_entities WHERE article_id = 1 AND kind = 'minister' ORDER BY slug")
    ministers = [r[0] for r in cursor.fetchall()]
    assert ministers == ['narendra_modi', 'rajnath_singh'], f"Expected ['narendra_modi', 'rajnath_singh'], got {ministers}"
    print("  Test 1 (ministers slugs update) Passed.")
    
    # Assert other kinds for article 1 remain untouched
    cursor.execute("SELECT slug FROM article_entities WHERE article_id = 1 AND kind = 'party'")
    parties = [r[0] for r in cursor.fetchall()]
    assert parties == ['bjp'], f"Expected ['bjp'], got {parties}"
    
    cursor.execute("SELECT slug FROM article_entities WHERE article_id = 1 AND kind = 'state'")
    states = [r[0] for r in cursor.fetchall()]
    assert states == ['delhi'], f"Expected ['delhi'], got {states}"
    print("  Test 2 (untouched kinds on same article) Passed.")
    
    # Assert other articles remain untouched
    cursor.execute("SELECT slug FROM article_entities WHERE article_id = 2 AND kind = 'minister'")
    control_ministers = [r[0] for r in cursor.fetchall()]
    assert control_ministers == ['nitin_gadkari'], f"Expected ['nitin_gadkari'], got {control_ministers}"
    print("  Test 3 (control article untouched) Passed.")
    
    # ----------------------------------------------------
    # TEST 2: Emptying ministers list deletes all minister bridge rows for that article
    # ----------------------------------------------------
    refresh_bridge_rows(cursor, 1, 'minister', [], slugify)
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM article_entities WHERE article_id = 1 AND kind = 'minister'")
    count = cursor.fetchone()[0]
    assert count == 0, f"Expected 0 minister rows for article 1, got {count}"
    print("  Test 4 (empty list deletes rows) Passed.")
    
    # Assert other kinds/articles still untouched
    cursor.execute("SELECT slug FROM article_entities WHERE article_id = 1 AND kind = 'party'")
    assert [r[0] for r in cursor.fetchall()] == ['bjp']
    
    cursor.execute("SELECT slug FROM article_entities WHERE article_id = 2 AND kind = 'minister'")
    assert [r[0] for r in cursor.fetchall()] == ['nitin_gadkari']
    print("  Test 5 (control verification after deletion) Passed.")
    
    # ----------------------------------------------------
    # TEST 3: Handling string (JSON) inputs robustly
    # ----------------------------------------------------
    json_input = '["Amit Shah", "Nirmala Sitharaman"]'
    refresh_bridge_rows(cursor, 1, 'minister', json_input, slugify)
    conn.commit()
    
    cursor.execute("SELECT slug FROM article_entities WHERE article_id = 1 AND kind = 'minister' ORDER BY slug")
    ministers = [r[0] for r in cursor.fetchall()]
    assert ministers == ['amit_shah', 'nirmala_sitharaman'], f"Expected ['amit_shah', 'nirmala_sitharaman'], got {ministers}"
    print("  Test 6 (JSON string values parsing) Passed.")
    
    print("\nAll entity updater bridge synchronization tests passed successfully!")
    conn.close()

if __name__ == '__main__':
    try:
        run_tests()
        sys.exit(0)
    except AssertionError as e:
        print(f"\nAssertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
