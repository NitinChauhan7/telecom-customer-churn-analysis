import pandas as pd

file_path = r"C:\Users\dell\Desktop\Teleco_customer_churn_analysis\Dataset\Telco_Customer_Churn_Dataset  (3).csv"

df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
print(df.head(10))


# Data cleaning
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

print("\n--- DESCRIPTIVE STATISTICS ---")
print("Monthly Charges Mean:", round(df["MonthlyCharges"].mean(), 2))
print("Monthly Charges Median:", round(df["MonthlyCharges"].median(), 2))
print("Monthly Charges Mode:", round(df["MonthlyCharges"].mode()[0], 2)) 

print("Duplicate Records:", df.duplicated().sum())
# 1. Overall churn
churn_rate = (df["Churn"] == "Yes").mean() * 100
print("\n--- OVERALL CHURN ---")
print("Total Customers:", len(df))
print("Churned Customers:", (df["Churn"] == "Yes").sum())
print("Churn Rate:", round(churn_rate, 2), "%")

# 2. Contract-wise churn
print("\n--- CONTRACT-WISE CHURN ---")
contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100
print(contract_churn.round(2))

# 3. Tenure-wise churn
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"]
)

print("\n--- TENURE-WISE CHURN ---")
tenure_churn = pd.crosstab(
    df["TenureGroup"],
    df["Churn"],
    normalize="index"
) * 100
print(tenure_churn.round(2))

# 4. Internet service
print("\n--- INTERNET SERVICE-WISE CHURN ---")
internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100
print(internet_churn.round(2))

# 5. Monthly charges
print("\n--- AVERAGE MONTHLY CHARGES ---")
print(df.groupby("Churn")["MonthlyCharges"].mean().round(2))

# 6. Payment method
print("\n--- PAYMENT METHOD-WISE CHURN ---")
payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100
print(payment_churn.round(2))

# 7. Tech support
print("\n--- TECH SUPPORT-WISE CHURN ---")
tech_churn = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
) * 100
print(tech_churn.round(2))

# 8. Online security
print("\n--- ONLINE SECURITY-WISE CHURN ---")
security_churn = pd.crosstab(
    df["OnlineSecurity"],
    df["Churn"],
    normalize="index"
) * 100
print(security_churn.round(2))
import matplotlib.pyplot as plt

contract_churn["Yes"].plot(kind="bar")

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(r"C:\Users\dell\Desktop\Teleco_customer_churn_analysis\Graphs\contract_churn.png")
plt.close()
tenure_churn["Yes"].plot(kind="bar")

plt.title("Churn Rate by Customer Tenure")
plt.xlabel("Tenure Group")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(r"C:\Users\dell\Desktop\Teleco_customer_churn_analysis\Graphs\tenure_churn.png")
plt.close()

internet_churn["Yes"].plot(kind="bar")

plt.title("Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(r"C:\Users\dell\Desktop\Teleco_customer_churn_analysis\Graphs\internet_service_churn.png")
plt.close()

payment_churn["Yes"].plot(kind="bar")

plt.title("Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(r"C:\Users\dell\Desktop\Teleco_customer_churn_analysis\Graphs\payment_method_churn.png")
plt.close()   # 9. Average Monthly Charges by Churn

avg_charges = df.groupby("Churn")["MonthlyCharges"].mean()

avg_charges.plot(kind="bar")

plt.title("Average Monthly Charges by Churn")
plt.xlabel("Churn")
plt.ylabel("Average Monthly Charges")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(r"C:\Users\dell\Desktop\Teleco_customer_churn_analysis\Graphs\average_monthly_charges_churn.png")

plt.close()

tech_churn["Yes"].plot(kind="bar")

plt.title("Churn Rate by Tech Support")
plt.xlabel("Tech Support")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(r"C:\Users\dell\Desktop\Teleco_customer_churn_analysis\Graphs\tech_support_churn.png")

plt.close()

security_churn["Yes"].plot(kind="bar")

plt.title("Churn Rate by Online Security")
plt.xlabel("Online Security")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(r"C:\Users\dell\Desktop\Teleco_customer_churn_analysis\Graphs\online_security_churn.png")

plt.close()
# Task 3 - Monthly Charges Distribution

plt.figure()
plt.hist(df["MonthlyCharges"], bins=20)
plt.title("Distribution of Monthly Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(r"C:\Users\dell\Desktop\Teleco_customer_churn_analysis\Graphs\monthly_charges_distribution.png")
plt.close()

print(df.dtypes)
print(df.isnull().sum())
# Task 3 - Monthly Charges Box Plot

plt.figure()
plt.boxplot(df["MonthlyCharges"])
plt.title("Box Plot of Monthly Charges")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig(r"C:\Users\dell\Desktop\Teleco_customer_churn_analysis\Graphs\monthly_charges_boxplot.png")
plt.close()
# Task 3 - Churn Distribution

plt.figure()
df["Churn"].value_counts().plot(kind="bar")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(r"C:\Users\dell\Desktop\Teleco_customer_churn_analysis\Graphs\churn_distribution.png")
plt.close()

# Task 4 - Customer Segmentation Visualization

# Tenure Segment Distribution
tenure_counts = df["TenureGroup"].value_counts().sort_index()

plt.figure()
tenure_counts.plot(kind="pie", autopct="%1.1f%%")
plt.title("Customer Distribution by Tenure Group")
plt.ylabel("")
plt.tight_layout()
plt.savefig(r"C:\Users\dell\Desktop\Teleco_customer_churn_analysis\Graphs\tenure_segment_distribution.png")
plt.close()

# Average Monthly Charges by Tenure Group
avg_charges_tenure = df.groupby("TenureGroup", observed=True)["MonthlyCharges"].mean()

plt.figure()
avg_charges_tenure.plot(kind="bar")
plt.title("Average Monthly Charges by Tenure Group")
plt.xlabel("Tenure Group")
plt.ylabel("Average Monthly Charges")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(r"C:\Users\dell\Desktop\Teleco_customer_churn_analysis\Graphs\avg_monthly_charges_by_tenure.png")
plt.close()

# Key Insight
highest_charge_group = avg_charges_tenure.idxmax()
highest_charge_value = avg_charges_tenure.max()

print("\n--- TASK 4 KEY INSIGHT ---")
print("Highest Average Monthly Charges:", highest_charge_group)
print("Average Monthly Charges:", round(highest_charge_value, 2))