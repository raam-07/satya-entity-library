import json
import os

entities_path = "entities.json"

with open(entities_path, "r", encoding="utf-8") as f:
    data = json.load(f)

parties = data["india"]["parties"]
existing_names = {p["name"].lower() for p in parties}
print("Existing parties:")
for p in parties:
    print(f"  - {p['name']}: {p.get('full_name', '')}")

# The list of 27 parties to backfill
backfill_list = [
    {"name": "VCK", "full_name": "Viduthalai Chiruthaigal Katchi", "aliases": ["Liberation Panthers Party"], "color": "#0000FF"},
    {"name": "PDP", "full_name": "Jammu and Kashmir Peoples Democratic Party", "aliases": ["Peoples Democratic Party"], "color": "#006400"},
    {"name": "AIADMK", "full_name": "All India Anna Dravida Munnetra Kazhagam", "aliases": [], "color": "#008000"},
    {"name": "MDMK", "full_name": "Marumalarchi Dravida Munnetra Kazhagam", "aliases": [], "color": "#FF0000"},
    {"name": "PMK", "full_name": "Pattali Makkal Katchi", "aliases": [], "color": "#FFFF00"},
    {"name": "AIMIM", "full_name": "All India Majlis-e-Ittehadul Muslimeen", "aliases": [], "color": "#008080"},
    {"name": "RLD", "full_name": "Rashtriya Lok Dal", "aliases": [], "color": "#00FF00"},
    {"name": "JD(S)", "full_name": "Janata Dal (Secular)", "aliases": ["JDS"], "color": "#00FF7F"},
    {"name": "INLD", "full_name": "Indian National Lok Dal", "aliases": [], "color": "#2E8B57"},
    {"name": "AGP", "full_name": "Asom Gana Parishad", "aliases": [], "color": "#32CD32"},
    {"name": "IUML", "full_name": "Indian Union Muslim League", "aliases": [], "color": "#228B22"},
    {"name": "KC(M)", "full_name": "Kerala Congress (M)", "aliases": ["Kerala Congress"], "color": "#FFD700"},
    {"name": "NPP", "full_name": "National People's Party", "aliases": [], "color": "#DAA520"},
    {"name": "NPF", "full_name": "Naga People's Front", "aliases": [], "color": "#4B0082"},
    {"name": "NDPP", "full_name": "Nationalist Democratic Progressive Party", "aliases": [], "color": "#800080"},
    {"name": "SKM", "full_name": "Sikkim Krantikari Morcha", "aliases": [], "color": "#FF1493"},
    {"name": "ZPM", "full_name": "Zoram People's Movement", "aliases": [], "color": "#C71585"},
    {"name": "MNF", "full_name": "Mizo National Front", "aliases": [], "color": "#FF4500"},
    {"name": "UPPL", "full_name": "United People's Party Liberal", "aliases": [], "color": "#FFA500"},
    {"name": "BPF", "full_name": "Bodoland People's Front", "aliases": [], "color": "#FF8C00"},
    {"name": "RSP", "full_name": "Revolutionary Socialist Party", "aliases": [], "color": "#B22222"},
    {"name": "AIUDF", "full_name": "All India United Democratic Front", "aliases": [], "color": "#008000"},
    {"name": "AJSU", "full_name": "All Jharkhand Students Union", "aliases": [], "color": "#FF8C00"},
    {"name": "RLP", "full_name": "Rashtriya Loktantrik Party", "aliases": [], "color": "#FFFF00"},
    {"name": "TVK", "full_name": "Tamilaga Vettri Kazhagam", "aliases": [], "color": "#8B0000"},
    {"name": "JJP", "full_name": "Jannayak Janta Party", "aliases": [], "color": "#808000"},
    {"name": "SAD", "full_name": "Shiromani Akali Dal", "aliases": ["Akali Dal"], "color": "#FFD700"}
]

added_count = 0
for party in backfill_list:
    name = party["name"]
    if name.lower() not in existing_names:
        # Create full entry with default fields
        full_entry = {
            "name": name,
            "full_name": party["full_name"],
            "aliases": party["aliases"],
            "ideology": "Regionalism, state rights",
            "founded": "Unknown",
            "president": "Unknown",
            "coalition": "Unknown",
            "ruling_states": [],
            "wikipedia": "",
            "color": party["color"]
        }
        parties.append(full_entry)
        existing_names.add(name.lower())
        added_count += 1
        print(f"Added party: {name}")
    else:
        print(f"Party already exists: {name}")

if added_count > 0:
    data["india"]["parties"] = parties
    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Backfill completed. Added {added_count} parties.")
else:
    print("No parties needed to be added.")
