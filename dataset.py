import json

# Each sample: one input token sequence, one gloss output
SAMPLES = [
    {
        "input":  "<AGENT> I <ACTION> GO <LOCATION> COLLEGE <TIME> TOMORROW",
        "output": "TOMORROW I GO COLLEGE"
    },
    {
        "input":  "<AGENT> I <NEG> TRUE <ACTION> LIKE <PATIENT> FOOD",
        "output": "I LIKE FOOD NOT"
    },
    {
        "input":  "<TIME> YESTERDAY <AGENT> SHE <ACTION> BUY <PATIENT> BOOK",
        "output": "YESTERDAY SHE BUY BOOK"
    },
    # Add 200-500 such pairs for a miniproject
]

def save_dataset(path="dataset.json"):
    with open(path, "w") as f:
        json.dump(SAMPLES, f, indent=2)
        
if __name__ == "__main__":
    save_dataset()