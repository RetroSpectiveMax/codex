import csv
import pathlib
import random

random.seed(42)

makes_models = {
    "Toyota": ["Camry", "Corolla", "RAV4"],
    "Honda": ["Civic", "Accord", "CR-V"],
    "Ford": ["F-150", "Escape", "Fusion"],
    "Chevrolet": ["Silverado", "Equinox", "Malibu"],
    "Tesla": ["Model 3", "Model S", "Model Y"],
    "BMW": ["3 Series", "5 Series", "X3"],
    "Subaru": ["Outback", "Forester", "Impreza"],
    "Hyundai": ["Elantra", "Sonata", "Tucson"],
}

years = list(range(2010, 2023))
complaint_templates = [
    "{} experienced {} around {} miles causing {}",
    "{} owner reported {} leading to {} during {} driving",
    "Complaints about {} include {} and {} after {} maintenance",
    "{} suffers from {} with {} warning lights at {} mileage",
]

issues = [
    "transmission failure",
    "engine stalling",
    "battery degradation",
    "electrical system glitch",
    "brake wear",
    "suspension noise",
    "infotainment crash",
    "air conditioning fault",
]

impacts = [
    "sudden shutdown",
    "reduced acceleration",
    "loss of power steering",
    "dashboard malfunction",
    "increased stopping distance",
    "warning chimes",
]

contexts = [
    "highway",
    "city",
    "winter",
    "summer road trip",
    "daily commute",
]

maintenance_actions = [
    "oil change",
    "brake pad replacement",
    "software update",
    "battery inspection",
    "tire rotation",
]

sentiment_terms = {
    "positive": ["reliable", "smooth", "satisfied", "comfortable", "quiet"],
    "negative": ["frustrating", "disappointed", "unsafe", "annoying", "expensive"],
}


def bounded_normal(mu, sigma, minimum, maximum):
    value = random.gauss(mu, sigma)
    return min(max(value, minimum), maximum)


def poisson(lmbda):
    L = pow(2.718281828459045, -lmbda)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= random.random()
    return k - 1


rows = []
for _ in range(400):
    make = random.choice(list(makes_models))
    model = random.choice(makes_models[make])
    year = random.choice(years)
    mileage = int(bounded_normal(60000, 15000, 10000, 200000))
    avg_trip_length = round(bounded_normal(18, 6, 5, 60), 1)
    maintenance_events = poisson(2)
    past_failures = poisson(1)
    severity_score = round(bounded_normal(3.5, 1.7, 0, 10), 2)
    maintenance_cost_last_year = round(bounded_normal(650, 220, 120, 2400), 2)
    fuel_cost_last_year = round(bounded_normal(1200, 300, 320, 3600), 2)

    issue = random.choice(issues)
    impact = random.choice(impacts)
    context = random.choice(contexts)
    template = random.choice(complaint_templates)
    maintenance = random.choice(maintenance_actions)

    adjectives = []
    if random.random() < 0.6:
        adjectives.append(random.choice(sentiment_terms["negative"]))
    if random.random() < 0.4:
        adjectives.append(random.choice(sentiment_terms["positive"]))
    sentiment_phrase = " and ".join(adjectives) if adjectives else "average"

    complaint = template.format(
        f"{year} {make} {model}",
        issue,
        mileage,
        impact,
    )
    complaint = f"{complaint}. Owner felt {sentiment_phrase} after {maintenance}."

    risk_factor = (
        0.3 * (mileage / 100000)
        + 0.25 * (maintenance_events / 5)
        + 0.2 * (past_failures / 3)
        + 0.25 * (severity_score / 10)
    )
    probability = min(max(risk_factor, 0.05), 0.9)
    has_issue = 1 if random.random() < probability else 0

    rows.append(
        [
            make,
            model,
            year,
            mileage,
            avg_trip_length,
            maintenance_events,
            past_failures,
            severity_score,
            maintenance_cost_last_year,
            fuel_cost_last_year,
            complaint,
            maintenance,
            has_issue,
        ]
    )

pathlib.Path("data").mkdir(exist_ok=True)
with open("data/car_reliability_synthetic.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "make",
            "model",
            "year",
            "mileage",
            "avg_trip_length_miles",
            "maintenance_events",
            "past_failures",
            "severity_score",
            "maintenance_cost_last_year",
            "fuel_cost_last_year",
            "complaint_text",
            "maintenance_action",
            "has_mechanical_issue",
        ]
    )
    writer.writerows(rows)
