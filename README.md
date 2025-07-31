# 📊 Is GenAI Stealing Our Jobs? | A SQL + Power BI Case Study

When I saw a reel saying *“GenAI will steal all your jobs soon”*, it sparked a question in my head — is that really happening?

So I rolled up my sleeves and built a full-stack data case study using:
- ✅ **SQL** for data cleaning and analysis  
- ✅ **Power BI** for interactive dashboards  
- ✅ A self-made dataset tracking GenAI adoption across industries and countries

---

## 🔍 Case Study Summary

This case study explores how companies across the globe are adopting GenAI tools and the resulting impact on:
- Productivity
- Job creation vs job loss
- Employee sentiment
- Training investment

---

## 📁 Dataset Structure

```sql
CREATE TABLE genai_adoption (
    company_name VARCHAR(255),
    industry VARCHAR(100),
    country VARCHAR(100),
    genai_tool VARCHAR(50),
    adoption_year INT,
    employees_impacted INT,
    new_roles_created INT,
    training_hours INT,
    productivity_change FLOAT,
    employee_sentiment TEXT,
    sentiment_category VARCHAR(100)
);
```
