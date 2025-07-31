# 📊 Is GenAI Stealing Our Jobs? | A SQL + Power BI Case Study

When I saw a reel saying *“GenAI will steal all your jobs soon”*, it sparked a question in my head — is that really happening?

So I rolled up my sleeves and built a full-stack data case study using:
- ✅ **Python** for data preparation
- ✅ **SQL** for analysis  
- ✅ **Power BI** for interactive dashboards  
- ✅ Dataset of 100000 Values from **Kaggle**

---

## 🔍 Case Study Summary

This case study explores how companies across the globe are adopting GenAI tools and the resulting impact on:
- Productivity
- Job creation vs job loss
- Employee sentiment
- Training investment

---
### Importing the Dataset
```python
import pandas as pd
df = pd.read_csv(r"C:\Users\Parth Yadav\Downloads\archive (8)\Enterprise_GenAI_Adoption_Impact.csv")
df.head(5)
```
### Data Cleaning
```python
print(df.isnull().sum())
print(df.duplicated().sum())
df.columns.str.strip()
df.replace({'â€”': '-', 'â€™': "'", 'â€œ': '"', 'â€': '"'}, regex=True, inplace=True)
df['GenAI Tool'] = df['GenAI Tool'].str.replace(r'\bgrok\b', 'Grok', case=False, regex=True)
df.head(20)
```
### Sentiment Categorization
```python
from sklearn.feature_extraction.text import CountVectorizer

sentiments = df['Employee Sentiment'].astype(str).str.lower()

vectorizer = CountVectorizer(stop_words='english', max_features=1000)  # top 1000 keywords
X = vectorizer.fit_transform(sentiments)
keywords = vectorizer.get_feature_names_out()

import numpy as np
word_freq = np.asarray(X.sum(axis=0)).flatten()
top_keywords = sorted(zip(keywords, word_freq), key=lambda x: x[1], reverse=True)

for word, freq in top_keywords[:50]:
    print(f"{word}: {freq}")

def map_sentiment_to_category(text):
    text = text.lower()
    positive_keywords = ['faster', 'efficient', 'improved', 'reduce', 'tasks', 'fun', 'ai']
    dynamic_keywords = ['roles', 'rotations', 'shifted', 'transition', 'new', 'smoother']
    negative_keywords = ['concern', 'replace', 'job', 'soon', 'scary', 'anxiety', 'struggling']
    learning_keywords = ['learning', 'curve', 'steep', 'adapted', 'helped', 'webinars']
    culture_keywords = ['collaboration', 'communicate', 'internal', 'hr', 'meetings', 'newsletter', 'townhalls']
    has_positive = any(word in text for word in positive_keywords)
    has_negative = any(word in text for word in negative_keywords)
    if has_positive and has_negative:
        return 'Mixed Impact'
    elif has_positive:
        return 'AI-Driven Productivity'
    elif any(word in text for word in dynamic_keywords):
        return 'Changing Job Dynamics'
    elif has_negative:
        return 'Job Security Anxiety'
    elif any(word in text for word in learning_keywords):
        return 'Learning & Adaptation'
    elif any(word in text for word in culture_keywords):
        return 'Communication & Culture Shift'
    else:
        return 'Uncategorized'

 df['Sentiment_Category'] = df['Employee Sentiment'].astype(str).apply(map_sentiment_to_category)


df[['Employee Sentiment', 'Sentiment_Category']].head()
```
### Final Checks
```python
uncategorized = df[df['Sentiment_Category'].isnull() | (df['Sentiment_Category'] == 'Uncategorized')]

print(f"Total Uncategorized Sentiments: {len(uncategorized)}")
uncategorized[['Employee Sentiment', 'Sentiment_Category']].head(10)
df.to_csv("cleaned_genai1.csv", index=False, encoding='utf-8')
```
### 📁 Dataset Structure
```sql
CREATE DATABASE Genai_db;

DROP TABLE IF EXISTS genai_adoption;
```
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
```sql
SELECT * FROM genai_adoption;
```

### 1. WHAT IS THE AVERAGE PRODUCTIVITY CHANGE PER COUNTRY BROKEM DOWN BY YEAR & SENTIMENT CATEGORY
```sql
WITH country_sentiment_productivity AS (
    SELECT
        country,
        adoption_year,
        sentiment_category,
        ROUND(AVG(Productivity_Change), 2) AS avg_productivity_change
    FROM
        genai_adoption
    GROUP BY
        country,
        adoption_year,
        sentiment_category
)
SELECT *
FROM country_sentiment_productivity
ORDER BY country, adoption_year, sentiment_category;
```
### 2. Which GenAI tools are used in the highest number of unique countries?
```sql
SELECT 
    genai_tool,
    COUNT(DISTINCT country) AS unique_countries
FROM 
    genai_adoption
GROUP BY 
    genai_tool
ORDER BY 
    unique_countries DESC;
```
### 3. Which are the top 3 companies (per adoption year) with the highest number of employees impacted?
```sql
WITH ranked_companies AS (
    SELECT 
        company_name,
        adoption_year,
        employees_impacted,
        RANK() OVER (PARTITION BY adoption_year ORDER BY employees_impacted DESC) AS rnk
    FROM 
        genai_adoption
)
SELECT 
    company_name,
    adoption_year,
    employees_impacted,
    rnk
FROM 
    ranked_companies
WHERE 
rnk <= 3
ORDER BY 
    adoption_year, rnk;
```
### 4. Which industries show consistent sentiment across all tools used?
```sql
WITH tool_sentiment_per_industry AS (
  SELECT
    industry,
    genai_tool,
    COUNT(DISTINCT sentiment_category) AS sentiment_variety
  FROM genai_adoption
  GROUP BY industry, genai_tool
)
SELECT
    industry,
    genai_tool,
    sentiment_variety
FROM tool_sentiment_per_industry;
```
### 5. Find tools where average productivity is below overall average
```sql
WITH 
avg_productivity AS (
    SELECT AVG(productivity_change) AS overall_avg
    FROM genai_adoption
),
tool_avg_productivity AS (
    SELECT genai_tool, AVG(productivity_change) AS tool_avg
    FROM genai_adoption
    GROUP BY genai_tool
)
SELECT genai_tool, tool_avg
FROM tool_avg_productivity, avg_productivity
WHERE tool_avg < overall_avg;
```
### 6. Which industry-country combinations have the highest role creation efficiency (roles created per 1000 employees impacted)?
```sql
WITH role_creation_efficiency AS (
    SELECT 
        industry,
	    country,
		SUM(new_roles_created) AS total_roles,
		SUM(employees_impacted) AS total_employees,
		ROUND(((SUM(new_roles_created) / SUM(employees_impacted))*1000), 2) AS efficiency_per_1000
    FROM
        genai_adoption
	GROUP BY
        industry, country
)
SELECT *
FROM role_creation_efficiency
ORDER BY efficiency_per_1000 DESC
LIMIT 10;
```
### 7. Which companies exceed both the industry and tool average in productivity change?
```sql
WITH industry_avg AS (
    SELECT 
        industry,
        AVG(productivity_change) AS industry_avg_productivity
    FROM genai_adoption
    GROUP BY industry
),
tool_avg AS (
    SELECT 
        genai_tool,
        AVG(productivity_change) AS tool_avg_productivity
    FROM genai_adoption
    GROUP BY genai_tool
)
SELECT 
    g.company_name,
    g.industry,
    g.genai_tool,
    g.productivity_change,
    i.industry_avg_productivity,
    t.tool_avg_productivity
FROM 
    genai_adoption g
JOIN industry_avg i ON g.industry = i.industry
JOIN tool_avg t ON g.genai_tool = t.genai_tool
WHERE 
    g.productivity_change > i.industry_avg_productivity
    AND g.productivity_change > t.tool_avg_productivity
ORDER BY 
    g.productivity_change DESC;
```
### 8. What is the average sentiment category distribution per country and adoption year?
```sql
WITH sentiment_counts AS (
    SELECT
        country,
        adoption_year,
        sentiment_category,
        COUNT(*) AS sentiment_count
    FROM genai_adoption
    GROUP BY country, adoption_year, sentiment_category
),
total_counts AS (
    SELECT
        country,
        adoption_year,
        COUNT(*) AS total_companies
    FROM genai_adoption
    GROUP BY country, adoption_year
)
SELECT 
    sc.country,
    sc.adoption_year,
    sc.sentiment_category,
    sc.sentiment_count,
    tc.total_companies,
    ROUND((sc.sentiment_count * 100.0) / tc.total_companies, 2) AS percentage_distribution
FROM sentiment_counts sc
JOIN total_counts tc
  ON sc.country = tc.country AND sc.adoption_year = tc.adoption_year
ORDER BY sc.country, sc.adoption_year, sc.sentiment_category;
```
### 9. Compare average productivity between early adopters (2022) vs late adopters (2023–2024)
```sql
SELECT
    CASE 
        WHEN adoption_year = 2022 THEN 'Early Adopter'
        WHEN adoption_year IN (2023, 2024) THEN 'Late Adopter'
        ELSE 'Other'
    END AS adoption_group,
    ROUND(AVG(productivity_change), 2) AS avg_productivity_change,
    COUNT(*) AS total_companies
FROM genai_adoption
WHERE adoption_year IN (2022, 2023, 2024)
GROUP BY adoption_group;
```
### 10. Which tools are used in at least 3 companies across 2 or more industries with positive sentiment categories?
```sql
WITH Positive_Sentiments AS(
    SELECT *
    FROM genai_adoption
    WHERE sentiment_category IN(
        'AI-Driven Productivity',
        'Changing Job Dynamics',
        'Communication & Culture Shift'		
    )
),
tool_company_industry AS (
    SELECT 
        genai_tool,
        company_name,
        industry
    FROM Positive_Sentiments
    GROUP BY genai_tool, company_name, industry
),
tool_summary AS (
    SELECT 
        genai_tool,
        COUNT(DISTINCT company_name) AS Company_Count,
        COUNT(DISTINCT industry) AS Industry_Count
    FROM tool_company_industry
    GROUP BY genai_tool
)
SELECT *
FROM tool_summary
WHERE Company_Count >= 3 AND Industry_Count >= 2;

```
### 11. Identify companies with many new roles but low productivity outcomes.
```sql
SELECT
    company_name,
    industry,
    country,
    genai_tool,
    new_roles_created,
    productivity_change
FROM
    genai_adoption
WHERE
    new_roles_created > 20
    AND productivity_change < 10
ORDER BY
    new_roles_created DESC, productivity_change ASC;
```
### 12. Which tools show the highest variance in productivity across industries?
```sql
WITH tool_industry_productivity AS (
    SELECT 
        genai_tool,
        industry,
        AVG(productivity_change) AS avg_productivity
    FROM genai_adoption
    GROUP BY genai_tool, industry
),
Productivity_Variance AS (
    SELECT 
        genai_tool,
        STDDEV_POP(avg_productivity) AS productivity_variance
    FROM tool_industry_productivity
    GROUP BY genai_tool
)
SELECT *
FROM Productivity_Variance
ORDER BY productivity_variance DESC;
```
### 13. Find industries where at least 3 tools are adopted, and productivity is above the dataset average.
```sql
WITH overall_avg_productivity_change AS (
    SELECT 
        AVG(productivity_change) AS overall_avg
    FROM genai_adoption
),
industry_tool_stats AS (
    SELECT
        industry,
        COUNT(DISTINCT genai_tool) AS tool_count,
        AVG(productivity_change) AS industry_avg
    FROM genai_adoption
    GROUP BY industry
)
SELECT i.*
FROM industry_tool_stats i
JOIN overall_avg_productivity_change o
  ON i.industry_avg > o.overall_avg
WHERE i.tool_count >= 3;
```
### 14. Cluster GenAI tools by behavioral patterns based on usage metrics
```sql
WITH tool_metrics AS (
  SELECT
    genai_tool,
    ROUND(AVG(productivity_change), 2) AS avg_productivity,
    ROUND(AVG(training_hours), 2) AS avg_training_hours,
    ROUND(AVG(new_roles_created), 0) AS avg_roles_created,
    ROUND(AVG(employees_impacted), 0) AS avg_employees_impacted
    FROM genai_adoption
  GROUP BY genai_tool
),
clustered_tools AS (
  SELECT *,
    -- Productivity Cluster
    CASE
      WHEN avg_productivity >= 18.5 THEN 'High Impact'
      WHEN avg_productivity BETWEEN 15 AND 18.5 THEN 'Moderate Impact'
      ELSE 'Low Impact'
    END AS productivity_cluster,

    -- Training Hours Cluster
    CASE
      WHEN avg_training_hours >= 12750 THEN 'High Training'
      WHEN avg_training_hours BETWEEN 11500 AND 12750 THEN 'Moderate Training'
      ELSE 'Low Training'
    END AS training_cluster,

    -- New Roles Cluster
    CASE
      WHEN avg_roles_created >= 15 THEN 'High Role Creation'
      WHEN avg_roles_created BETWEEN 10 AND 15 THEN 'Moderate Role Creation'
      ELSE 'Low Role Creation'
    END AS role_cluster
  FROM tool_metrics
)
SELECT 
  genai_tool,
  avg_productivity,
  avg_training_hours,
  avg_roles_created,
  avg_employees_impacted,
  CONCAT_WS(', ', productivity_cluster, training_cluster, role_cluster) AS behavior_cluster
FROM clustered_tools
ORDER BY genai_tool;
```
### 15. How does sentiment category impact productivity?
```sql
WITH sentiment_productivity AS (
  SELECT 
    sentiment_category, 
    ROUND(AVG(productivity_change), 2) AS avg_productivity,
    COUNT(*) AS company_count
  FROM 
    genai_adoption
  GROUP BY 
    sentiment_category
)
SELECT 
  sentiment_category, 
  avg_productivity,
  company_count
FROM 
  sentiment_productivity
ORDER BY 
  avg_productivity DESC;
```
### 16. Create a Workforce Transformation Score per company
```sql
WITH company_metrics AS (
  SELECT 
    company_name,
    ROUND(AVG(productivity_change), 2) AS avg_productivity,
    ROUND(SUM(new_roles_created), 2) AS total_roles,
    ROUND(SUM(employees_impacted), 2) AS total_employees,
    ROUND(SUM(training_hours), 2) AS total_training
  FROM 
    genai_adoption
  GROUP BY 
    company_name
),
transformation_score AS (
  SELECT 
    company_name,
    avg_productivity,
    total_roles,
    total_employees,
    total_training,
    ROUND(
      0.4 * avg_productivity +
      0.3 * (total_roles / NULLIF(total_employees, 0)) +
      0.3 * (total_training / NULLIF(total_roles, 0)),
      2
    ) AS workforce_transformation_score
  FROM 
    company_metrics
),
segmented_companies_ratings AS (
  SELECT *,
    CASE
      WHEN workforce_transformation_score >= 5000 THEN '*****'
      WHEN workforce_transformation_score >= 2000 THEN '****'
      WHEN workforce_transformation_score >= 500 THEN '***'
      ELSE '*'
END AS transformation_segment_ratings
  FROM transformation_score
)
SELECT *
FROM segmented_companies_ratings
ORDER BY workforce_transformation_score DESC;
```
### 17. Find contradictory cases: High role creation but high job security anxiety
```sql
SELECT 
    company_name,
    industry,
    country,
    genai_tool,
    new_roles_created,
    sentiment_category,
    productivity_change
FROM 
    genai_adoption
WHERE 
    new_roles_created > 20
    AND sentiment_category = 'Job Security Anxiety'
ORDER BY 
    new_roles_created DESC;
```
### 18. Tool impact across multiple dimensions
```sql
WITH 
productivity_cte AS (
  SELECT 
    genai_tool,
    AVG(productivity_change) AS avg_productivity
  FROM genai_adoption
  GROUP BY genai_tool
),
sentiment_var_cte AS (
  SELECT 
    genai_tool,
    STDDEV_POP(CASE 
                  WHEN sentiment_category IS NOT NULL THEN 
                      CASE 
                          WHEN sentiment_category = 'AI-Driven Productivity' THEN 4
                          WHEN sentiment_category = 'Changing Job Dynamics' THEN 3
                          WHEN sentiment_category = 'Mixed Impact' THEN 2
                          WHEN sentiment_category = 'Job Security Anxiety' THEN 1
                          ELSE 0
                      END
              END) AS sentiment_variance
  FROM genai_adoption
  GROUP BY genai_tool
),
tool_freq_cte AS (
  SELECT 
    genai_tool,
    COUNT(DISTINCT company_name) AS tool_frequency
  FROM genai_adoption
  GROUP BY genai_tool
),
combined_cte AS (
  SELECT 
    p.genai_tool,
    p.avg_productivity,
    s.sentiment_variance,
    f.tool_frequency
  FROM productivity_cte p
  JOIN sentiment_var_cte s ON p.genai_tool = s.genai_tool
  JOIN tool_freq_cte f ON p.genai_tool = f.genai_tool
),
stats AS (
  SELECT 
    MIN(avg_productivity) AS min_prod,
    MAX(avg_productivity) AS max_prod,
    MIN(sentiment_variance) AS min_var,
    MAX(sentiment_variance) AS max_var,
    MIN(tool_frequency) AS min_freq,
    MAX(tool_frequency) AS max_freq
  FROM combined_cte
)
SELECT 
  c.genai_tool,
  ROUND((c.avg_productivity - s.min_prod) / NULLIF(s.max_prod - s.min_prod, 0), 2) AS norm_productivity,
  ROUND((c.sentiment_variance - s.min_var) / NULLIF(s.max_var - s.min_var, 0), 2) AS norm_sentiment_variance,
  ROUND((c.tool_frequency - s.min_freq) / NULLIF(s.max_freq - s.min_freq, 0), 2) AS norm_tool_frequency
FROM 
  combined_cte c, stats s
ORDER BY genai_tool;
```
### 19. Sentiment shift analysis: How sentiments changed over years by industry
```sql
WITH total_per_industry_year AS (
  SELECT
    industry,
    adoption_year,
    COUNT(*) AS total_companies
  FROM genai_adoption
  GROUP BY industry, adoption_year
),
sentiment_counts AS (
  SELECT
    industry,
    adoption_year,
    sentiment_category,
    COUNT(*) AS sentiment_count
  FROM genai_adoption
  GROUP BY industry, adoption_year, sentiment_category
)
SELECT
  s.industry,
  s.adoption_year,
  s.sentiment_category,
  s.sentiment_count,
  t.total_companies,
  ROUND((s.sentiment_count / t.total_companies) * 100, 2) AS sentiment_percentage
FROM sentiment_counts s
JOIN total_per_industry_year t
  ON s.industry = t.industry AND s.adoption_year = t.adoption_year
ORDER BY s.industry, s.adoption_year, s.sentiment_category;
```
### 20. Do companies that impact more employees also invest more in training each person?
```sql
WITH company_training AS (
  SELECT 
    company_name,
    SUM(employees_impacted) AS total_employees_impacted,
    SUM(training_hours) AS total_training_hours,
    ROUND(SUM(training_hours) / NULLIF(SUM(employees_impacted), 0), 2) AS training_per_employee
  FROM genai_adoption
  GROUP BY company_name
),
top_impact_threshold AS (
  SELECT 
    MAX(total_employees_impacted) * 0.8 AS threshold
  FROM company_training
),
tagged_companies AS (
  SELECT 
    c.*,
    CASE 
      WHEN c.total_employees_impacted >= t.threshold THEN 'High Impact'
      ELSE 'Others'
    END AS impact_segment
  FROM company_training c
  CROSS JOIN top_impact_threshold t
)
SELECT 
  impact_segment,
  ROUND(AVG(training_per_employee), 2) AS avg_training_per_employee,
  COUNT(*) AS number_of_companies
FROM tagged_companies
GROUP BY impact_segment
ORDER BY impact_segment DESC;
```
