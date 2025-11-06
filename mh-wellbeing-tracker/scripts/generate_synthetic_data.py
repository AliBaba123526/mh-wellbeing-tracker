import os, random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker
from dateutil.relativedelta import relativedelta

# Output folder (relative to this script)
OUT_DIR = os.path.join("..", "data", "synthetic")
N_USERS = 25          # change if you want more/less demo users
DAYS_BACK = 90        # ~last 3 months
RNG = np.random.default_rng(42)
fake = Faker("en_GB")

def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

# ---------- Users ----------
users = []
for _ in range(N_USERS):
    name = fake.name()
    clean_name = name.lower().replace(' ', '.').replace("'", '')
    email = f"{clean_name}{RNG.integers(100,999)}@example.com"


    users.append({
        "FullName": name,
        "Email": email,
        "PasswordHash": "hashedpassword",
        "ConsentGiven": RNG.choice([0,1], p=[0.1,0.9]),
        "IsAnonymous": RNG.choice([0,1], p=[0.7,0.3]),
        "CreatedAt": fake.date_time_between(start_date="-120d", end_date="now").strftime("%Y-%m-%d %H:%M:%S"),
    })
users_df = pd.DataFrame(users)

today = datetime.today().date()
start_date = today - timedelta(days=DAYS_BACK)

mood_rows, sleep_rows, activity_rows, journal_rows, rec_rows, alert_rows = [], [], [], [], [], []

activity_types = ["Walk","Run","Gym","Yoga","Meditation","Tennis","Cycling","Stretching","Study Break"]
rec_pool = [
    "5-minute breathing exercise",
    "10-minute walk outside",
    "Digital detox for 30 minutes",
    "Drink water and stretch",
    "Short journaling prompt",
    "Early bedtime target +30 mins",
]

for u in users_df.itertuples(index=False):
    mood_base = RNG.integers(4, 7)         # personal baseline 4–6
    sleep_mean = RNG.uniform(6.0, 8.0)
    active_prob = RNG.uniform(0.3, 0.7)

    for d in daterange(start_date, today):
        noise = RNG.normal(0, 1.0)
        weekly = 0.7*np.sin(2*np.pi*(d.toordinal()%7)/7.0)
        mood = int(clamp(round(mood_base + noise + weekly), 1, 10))
        note = RNG.choice([None,"Busy day","Feeling ok","Stressed about uni","Had a good study session","Watched a film","Tired"],
                          p=[0.5,0.1,0.1,0.1,0.1,0.05,0.05])
        mood_rows.append({"UserEmail": u.Email, "MoodScore": mood, "Note": note, "LoggedAt": f"{d} 18:00:00"})

        sleep = clamp(RNG.normal(sleep_mean, 1.0), 4.0, 9.5)
        sleep_quality = RNG.choice(["Poor","Fair","Good","Very Good"], p=[0.15,0.25,0.4,0.2])
        sleep_rows.append({"UserEmail": u.Email, "SleepHours": round(float(sleep),1), "SleepQuality": sleep_quality, "LoggedAt": f"{d} 08:00:00"})

        if RNG.random() < active_prob:
            mins = int(clamp(RNG.normal(35, 15), 10, 120))
            atype = RNG.choice(activity_types)
            activity_rows.append({"UserEmail": u.Email, "ActivityType": atype, "DurationMinutes": mins, "LoggedAt": f"{d} 19:00:00"})

        if RNG.random() < 3/7:
            pos = ["Felt productive after study. Managed tasks well.","Nice chat with a friend. Mood lifted.","Exercise helped me relax today.","Slept okay and felt focused."]
            neg = ["Feeling stressed about deadlines.","Low energy and anxious this morning.","Couldn’t sleep properly and felt irritable.","Overwhelmed with coursework."]
            pool = pos if mood >= 6 else neg if mood <= 4 else pos+neg
            text = RNG.choice(pool)
            sscore = float((mood - 5)/5.0)  # quick proxy for demo
            slabel = "positive" if sscore > 0.2 else ("negative" if sscore < -0.2 else "neutral")
            cats = ";".join([c for c in ["sleep","study","exercise","social"] if RNG.random()<0.3]) or None
            journal_rows.append({"UserEmail": u.Email, "EntryText": text, "SentimentScore": round(sscore,3),
                                 "SentimentLabel": slabel, "DetectedCategories": cats, "LoggedAt": f"{d} 21:00:00"})

        if RNG.random() < 0.25:
            rec_rows.append({"UserEmail": u.Email, "RecommendationText": RNG.choice(rec_pool),
                             "GeneratedBy": RNG.choice(["rule","model"], p=[0.7,0.3]),
                             "Accepted": int(RNG.random() < 0.5), "CreatedAt": f"{d} 09:00:00"})

        if mood <= 2 and RNG.random() < 0.5:
            alert_rows.append({"UserEmail": u.Email, "AlertType": "LowMood",
                               "AlertMessage": "Detected sudden low mood; show support resources.",
                               "Acknowledged": int(RNG.random() < 0.6), "CreatedAt": f"{d} 22:00:00"})

# model metadata
model_runs = []
trained_dates = [datetime.today() - relativedelta(months=2), datetime.today() - relativedelta(weeks=3)]
for i, t in enumerate(trained_dates, start=1):
    model_runs.append({"ModelName": "SentimentBaseline" if i==1 else "MoodPredictorV1",
                       "Version": f"v{i}.0", "Accuracy": round(float(RNG.uniform(0.70, 0.90)),3),
                       "F1Score": round(float(RNG.uniform(0.68, 0.88)),3),
                       "TrainedOn": t.strftime("%Y-%m-%d %H:%M:%S"), "Notes": "Synthetic training; offline metrics only"})

# write files
os.makedirs(OUT_DIR, exist_ok=True)
pd.DataFrame(users).to_csv(os.path.join(OUT_DIR, "Users.csv"), index=False)
pd.DataFrame(model_runs).to_csv(os.path.join(OUT_DIR, "ModelRuns.csv"), index=False)
pd.DataFrame(mood_rows).to_csv(os.path.join(OUT_DIR, "MoodLogs_staging.csv"), index=False)
pd.DataFrame(sleep_rows).to_csv(os.path.join(OUT_DIR, "SleepLogs_staging.csv"), index=False)
pd.DataFrame(activity_rows).to_csv(os.path.join(OUT_DIR, "ActivityLogs_staging.csv"), index=False)
pd.DataFrame(journal_rows).to_csv(os.path.join(OUT_DIR, "JournalEntries_staging.csv"), index=False)
pd.DataFrame(rec_rows).to_csv(os.path.join(OUT_DIR, "Recommendations_staging.csv"), index=False)
pd.DataFrame(alert_rows).to_csv(os.path.join(OUT_DIR, "Alerts_staging.csv"), index=False)

print("Done! CSVs created in ../data/synthetic/")

