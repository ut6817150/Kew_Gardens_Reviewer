from checkers.formatting import FormattingChecker

checker = FormattingChecker()

# Test cases
test_texts = [
    "Species in the family rubiaceae",           # Should flag: rubiaceae → Rubiaceae
    "Species in the family Rubiaceae",           # Should pass (correct)
    "Members of orchidaceae",                    # Should flag: orchidaceae → Orchidaceae
    "Found in <i>Fabaceae</i>",                  # Should flag: italicized family name
    "Order rosales includes many species",       # Should flag: rosales → Rosales
    "The Felidae family contains cats",          # Should pass (correct)
]

for text in test_texts:
    violations = checker.check(text)
    print(f"\nText: '{text}'")
    if violations:
        for v in violations:
            if 'family' in v.message.lower() or 'taxonomy' in v.message.lower():
                print(f"  ✗ {v.message}")
                if v.suggested_fix:
                    print(f"    Fix: {v.suggested_fix}")
    else:
        print(f"No violations")