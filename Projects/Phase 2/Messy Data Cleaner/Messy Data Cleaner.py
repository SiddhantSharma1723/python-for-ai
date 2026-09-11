import pandas as pd
record = pd.read_csv("../../Data/students_records_messy.csv")
# print(record.shape)
# print(record.dtypes)
# print(record.info())
# print(record[record.duplicated()])

# print(record["age"].isna().sum())
# print(record["score"].isna().sum())
# print(record["city"].isna().sum())
# print(record["join_date"].isna().sum())
# print(record["gender"].isna().sum())
# print(record[record["score"]>0])

record["age"] = pd.to_numeric(record["age"],errors = "coerce")

correct_record = record.dropna(subset =["age"]).copy()

correct_record["score"] = pd.to_numeric(correct_record["score"],errors = "coerce")
correct_record = correct_record[correct_record["score"].notna() & (correct_record["score"]>=0)]

# correct_record = correct_record.rename[

# ]

# print(correct_record["gender"].unique())
correct_record["gender"] = correct_record["gender"].astype(str).str.strip().str.lower()

gender_map = {
    "nan":"Other",
    "female":"Female",
    "f":"Female",
    "male":"Male",
    "man":"Male",
    "other":"Other",
    "m":"Male"
}

correct_record["gender"] = correct_record["gender"].map(gender_map)
correct_record["gender"] = correct_record["gender"].fillna("Other")
# print(correct_record["gender"].unique())

print(correct_record["city"].unique())

correct_record["city"] = correct_record["city"].astype(str).str.strip().str.title()
correct_record["city"] = correct_record["city"].replace(
    ["Nan"],"Unknown"
)
correct_record["city"] = correct_record["city"].fillna("Unknown")

correct_record["join_date"] = pd.to_datetime(correct_record["join_date"],format = "mixed",errors = "coerce", dayfirst = True)

correct_record = correct_record.drop_duplicates()

print(correct_record)

def schema_check(df):
    assert df['age'].dtype != object, "age is not numeric!"
    assert df['score'].dtype != object, "score is not numeric!"
    assert df['student_id'].duplicated().sum() == 0, "Duplicate student_ids found!"
    assert df.duplicated().sum() == 0, "Duplicate rows still exist!"
    print("Schema check passed.")

schema_check(correct_record)