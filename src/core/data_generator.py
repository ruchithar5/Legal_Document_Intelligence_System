# src/data_generator.py
import os
import random
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # project root
DATA_DIR = BASE / "data"

CLASSES = {
    "Agreement": [
        "This agreement is made on {date} between {party_a} and {party_b} for {purpose}. {terms}",
        "The parties hereto agree to the terms of this agreement, which shall be effective on {date}. {terms}",
        "This agreement outlines payment, confidentiality and liability terms between {party_a} and {party_b}. {terms}"
    ],
    "Contract": [
        "This contract is entered into by {party_a} and {party_b} on {date} and sets forth the obligations for supply of goods. {terms}",
        "The contract requires delivery of the goods within {days} days, with penalties for delay. {terms}",
        "Contract for services between {party_a} and {party_b} including scope, payment and termination. {terms}"
    ],
    "NDA": [
        "This Non-Disclosure Agreement (NDA) is entered into by {party_a} and {party_b} to protect confidential information. {terms}",
        "The recipient shall not disclose Confidential Information for {years} years and must return materials on request. {terms}",
        "Confidential Information excludes information already known or in the public domain. {terms}"
    ],
    "Lease": [
        "This lease agreement is made between landlord {party_a} and tenant {party_b} for property at {address}. Rent is {rent} per month. {terms}",
        "Lease term shall be {months} months commencing {date}. Tenant responsible for utilities and maintenance. {terms}",
        "Security deposit of {amount} will be held and returned subject to property condition. {terms}"
    ],
    "Employment": [
        "This employment agreement between {employer} and {employee} sets salary, benefits, and termination notice. {terms}",
        "Employee will perform duties as outlined and will be subject to company policies and confidentiality obligations. {terms}",
        "Employment commences on {date} and includes probation period and performance review schedules. {terms}"
    ]
}

PARTIES = [
    "Alpha Tech Pvt Ltd", "Beta Solutions", "John Doe", "Jane Smith",
    "Global Supplies Inc.", "City Rentals LLC", "Sunrise Technologies"
]

PURPOSES = [
    "software development services", "consulting engagement", "partnership in logistics",
    "distribution of products", "intellectual property licensing", "data analysis project"
]

TERMS_SNIPPETS = [
    "Payment to be made within 30 days. Governing law shall be the laws of India.",
    "Either party may terminate with 30 days' written notice.",
    "All disputes shall be resolved by arbitration in Mumbai.",
    "Parties will comply with data protection regulations and secure all confidential information.",
    "Indemnification clauses apply for breach or negligence as outlined below."
]

def make_text(template):
    t = template.format(
        date=f"{random.randint(1,28)} {random.choice(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])} 2024",
        party_a=random.choice(PARTIES),
        party_b=random.choice(PARTIES),
        employer=random.choice(PARTIES),
        employee=random.choice(["Alice Kumar", "Rohan Mehta", "Suresh Kumar", "Priya Patel"]),
        address=f"{random.randint(1,999)} MG Road, City",
        rent=f"₹{random.randint(5000,100000)}",
        amount=f"₹{random.randint(5000,500000)}",
        days=random.randint(7,120),
        months=random.choice([6,11,12,24]),
        years=random.choice([1,2,3,5]),
        purpose=random.choice(PURPOSES),  # ✅ Added missing variable
        terms=random.choice(TERMS_SNIPPETS)
    )
    return t

def generate(n_per_class=200):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for cls, templates in CLASSES.items():
        folder = DATA_DIR / cls
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            template = random.choice(templates)
            text = make_text(template)
            fname = folder / f"{cls.lower()}_{i+1}.txt"
            with open(fname, "w", encoding="utf-8") as f:
                f.write(text)
    print(f"\n✅ Generated {n_per_class} samples for each class.")
    print(f"📂 Dataset created under: {DATA_DIR.resolve()}")

if __name__ == "__main__":
    generate(n_per_class=200)
