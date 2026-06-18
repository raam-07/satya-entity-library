import re
import sys

def normalize_text(t):
    if not t:
        return ""
    return re.sub(r'[^a-z0-9]', '', t.lower())

def backstop_role(name, full_name, role, role_evidence, context_sample):
    norm_evidence = normalize_text(role_evidence)
    norm_name = normalize_text(name)
    norm_full_name = normalize_text(full_name)
    norm_context = normalize_text(context_sample)

    is_evidence_valid = False
    if norm_evidence:
        # Must contain candidate's name or full_name
        contains_name = (norm_name in norm_evidence) or (norm_full_name in norm_evidence)
        # Must exist in context_sample
        in_context = norm_evidence in norm_context
        if contains_name and in_context:
            is_evidence_valid = True

    if not is_evidence_valid:
        role = ""
        role_evidence = ""

    # Drop former/ex roles
    if role:
        role_lower = role.lower()
        if "former" in role_lower or "ex-" in role_lower or "ex " in role_lower:
            role = ""
            role_evidence = ""

    return role

# Run test cases
test_cases = [
    # 1. The Pabitra Margherita Trap (mislabeled role of another person)
    {
        "name": "Pabitra Margherita",
        "full_name": "Pabitra Margherita",
        "role": "Former Chief Minister of Assam",
        "role_evidence": "former Chief Minister of Assam, Tarun Gogoi",
        "context": "BJP Union Minister Pabitra Margherita attacked the former Chief Minister of Assam, Tarun Gogoi.",
        "expected": ""
    },
    # 2. Pabitra Margherita correct case
    {
        "name": "Pabitra Margherita",
        "full_name": "Pabitra Margherita",
        "role": "Union Minister",
        "role_evidence": "BJP Union Minister Pabitra Margherita",
        "context": "BJP Union Minister Pabitra Margherita attacked the former Chief Minister of Assam, Tarun Gogoi.",
        "expected": "Union Minister"
    },
    # 3. Met with the Chief Minister (no name in evidence)
    {
        "name": "K. Shivam",
        "full_name": "K. Shivam",
        "role": "Chief Minister",
        "role_evidence": "met with the Chief Minister",
        "context": "K. Shivam met with the Chief Minister of Tamil Nadu to discuss the new budget.",
        "expected": ""
    },
    # 4. Succeeded the former Minister (contains "former" and no name)
    {
        "name": "Ramesh Singh",
        "full_name": "Ramesh Kumar Singh",
        "role": "former Minister",
        "role_evidence": "former Minister of state",
        "context": "Ramesh Kumar Singh succeeded the former Minister of state last Tuesday.",
        "expected": ""
    },
    # 5. Legitimate former role on self (dropped by former filter)
    {
        "name": "Suresh Prabhu",
        "full_name": "Suresh Prabhu",
        "role": "Former Railway Minister",
        "role_evidence": "Former Railway Minister Suresh Prabhu",
        "context": "Former Railway Minister Suresh Prabhu attended the seminar on high-speed rail.",
        "expected": ""
    },
    # 6. Valid current role with slight spelling/punctuation variation in evidence
    {
        "name": "A. Raja",
        "full_name": "Andimuthu Raja",
        "role": "Member of Parliament",
        "role_evidence": "Member of Parliament A. Raja,",
        "context": "The DMK leader and Member of Parliament A. Raja, addressed the press conference.",
        "expected": "Member of Parliament"
    }
]

failed = 0
for idx, tc in enumerate(test_cases, 1):
    result = backstop_role(tc["name"], tc["full_name"], tc["role"], tc["role_evidence"], tc["context"])
    if result != tc["expected"]:
        print(f"Test case {idx} FAILED!")
        print(f"  Name: {tc['name']}")
        print(f"  Context: {tc['context']}")
        print(f"  Extracted role: {tc['role']}")
        print(f"  Evidence: {tc['role_evidence']}")
        print(f"  Expected result: '{tc['expected']}', Got: '{result}'")
        failed += 1
    else:
        print(f"Test case {idx} PASSED.")

if failed > 0:
    print(f"\n{failed} test case(s) failed.")
    sys.exit(1)
else:
    print("\nAll role backstop unit tests passed successfully!")
    sys.exit(0)
