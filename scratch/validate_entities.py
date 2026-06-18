import json
import re
import sys

file_path = "entities.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

states = data["india"]["states"]
cms = data["india"]["state_chief_ministers"]
parties = data["india"]["parties"]

# Build valid state names and their active CM
active_cm_by_state = {}
for s in states:
    name = s["name"]
    cm = s.get("cm")
    if cm and cm.strip() and cm.strip() != "N/A":
        active_cm_by_state[name.lower()] = cm.strip()

# Helper for normalized matching
def clean(n):
    return re.sub(r'[^a-z0-9]', '', n.lower())

def norm_words(n):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', n.lower().replace('.', ' '))).strip()

def matches_cm(profile, auth_cm):
    auth_cm_clean = clean(auth_cm)
    if clean(profile.get('name', '')) == auth_cm_clean:
        return True
    for alias in profile.get('aliases', []):
        if clean(alias) == auth_cm_clean:
            return True
    p_norm = norm_words(profile.get('name', ''))
    auth_norm = norm_words(auth_cm)
    if p_norm and auth_norm:
        if re.search(r'\b' + re.escape(auth_norm) + r'\b', p_norm) or \
           re.search(r'\b' + re.escape(p_norm) + r'\b', auth_norm):
            return True
    return False

# Build party registry lookup
party_registry = set()
for p in parties:
    party_registry.add(p['name'].lower())
    if p.get('full_name'):
        party_registry.add(p['full_name'].lower())
    for alias in p.get('aliases', []):
        party_registry.add(alias.lower())

print("--- RUNNING OFFLINE ENTITIES VALIDATION ---")
failures = 0

# 1. Assert no Sheikh Abdullah profile exists in state_chief_ministers
abdullah_profiles = [c for c in cms if c["name"] == "Sheikh Abdullah"]
if abdullah_profiles:
    print("FAIL: Sheikh Abdullah profile still exists in state_chief_ministers!")
    failures += 1
else:
    print("PASS: Sheikh Abdullah removed from state_chief_ministers.")

# 2. Assert at most 1 Chief Minister per state in state_chief_ministers
state_cm_counts = {}
for c in cms:
    state_name = c.get("state")
    if state_name:
        state_cm_counts[state_name] = state_cm_counts.get(state_name, 0) + 1

multiple_cms = {state: count for state, count in state_cm_counts.items() if count > 1}
if multiple_cms:
    print(f"FAIL: States with multiple active CM profiles in state_chief_ministers: {multiple_cms}")
    failures += 1
else:
    print("PASS: No states have multiple active CM profiles.")

# 3. Assert every CM in state_chief_ministers matches state['cm'] if populated
for c in cms:
    state_name = c.get("state")
    if not state_name:
        print(f"FAIL: CM profile {c['name']} has no state field!")
        failures += 1
        continue
    
    state_lower = state_name.lower()
    if state_lower in active_cm_by_state:
        auth_cm = active_cm_by_state[state_lower]
        if not matches_cm(c, auth_cm):
            print(f"FAIL: CM profile '{c['name']}' does not match authoritative CM '{auth_cm}' for state '{state_name}'!")
            failures += 1
    else:
        print(f"INFO: Guard check - CM profile '{c['name']}' exists for state '{state_name}' which has empty/N/A active CM.")

all_politicians = (
    cms + 
    data["india"].get("cabinet_ministers", []) + 
    data["india"].get("opposition_leaders", []) + 
    data["india"].get("generic_politicians", [])
)

# 4. Assert no politician entry has "former" or "ex-" or "ex " in their role
for p in all_politicians:
    role = p.get("role", "")
    role_lower = role.lower()
    if "former" in role_lower or "ex-" in role_lower or "ex " in role_lower:
        p_name = p["name"]
        is_auto = p.get("auto_added", False)
        if is_auto:
            print(f"FAIL: Auto-added profile '{p_name}' has former/ex in role: '{role}'!")
            failures += 1
        else:
            # Log warning/info for legacy profiles
            print(f"INFO: Legacy profile '{p_name}' has former/ex in role: '{role}'")

if failures == 0:
    print("PASS: No former/ex roles in auto-added profiles.")

# 5. Assert all auto-added entries have valid party or state, and valid party values
auto_added_count = 0
for p in all_politicians:
    if p.get("auto_added"):
        auto_added_count += 1
        p_name = p["name"]
        party = p.get("party")
        state = p.get("state")
        
        # Check nationality backstop
        has_valid_party = party and party.lower() in party_registry
        has_valid_state = state and state.lower() in [s["name"].lower() for s in states]
        if not (has_valid_party or has_valid_state):
            print(f"FAIL: Auto-added entry '{p_name}' failed nationality backstop (party: '{party}', state: '{state}')")
            failures += 1
            
        # Check party normalization
        if party and party.lower() not in party_registry and party != "unconfirmed":
            print(f"FAIL: Auto-added entry '{p_name}' has unrecognized party: '{party}'")
            failures += 1

print(f"Checked {auto_added_count} auto-added entries.")

print("-------------------------------------------")
if failures > 0:
    print(f"VALIDATION FAILED: {failures} errors found.")
    sys.exit(1)
else:
    print("VALIDATION SUCCESSFUL! All integrity checks passed.")
    sys.exit(0)
