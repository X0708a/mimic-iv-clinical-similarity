-- Step 0: Prescriptions with noise filtering
CREATE OR REPLACE TABLE rx_cleaned AS
SELECT 
    subject_id,
    hadm_id,
    drug,
    drug_type,
    formulary_drug_cd,
    prod_strength,
    doses_per_24_hrs,
    route,
    starttime,
    stoptime
FROM prescriptions
WHERE drug IS NOT NULL 
  AND drug != ''
  AND NOT REGEXP_MATCHES(drug, '(?i)(saline|sodium chloride|ns|dextrose|flush|heparin lock|water|diluent)');

-- Step 1: Procedures - RAW nested structure (simplified without MAP_AGG)
CREATE OR REPLACE TABLE procedures_raw_agg AS
SELECT 
    p.subject_id,
    p.hadm_id,
    -- Store procedures as JSON string grouped by version
    -- ICD-9 procedures
    ARRAY_AGG(
        STRUCT_PACK(
            icd_code := p.icd_code,
            long_title := COALESCE(d.long_title, 'Unknown')
        )
    ) FILTER (WHERE p.icd_version = 9) as procedures_icd9,
    
    -- ICD-10 procedures
    ARRAY_AGG(
        STRUCT_PACK(
            icd_code := p.icd_code,
            long_title := COALESCE(d.long_title, 'Unknown')
        )
    ) FILTER (WHERE p.icd_version = 10) as procedures_icd10
    
FROM procedures_icd p
LEFT JOIN d_icd_procedures d 
    ON p.icd_code = d.icd_code 
    AND p.icd_version = d.icd_version
GROUP BY p.subject_id, p.hadm_id;

-- Step 2: Procedures - ENGINEERED features
CREATE OR REPLACE TABLE procedures_features AS
SELECT 
    p.subject_id,
    p.hadm_id,
    
    -- Full procedure codes (for recommendations)
    ARRAY_AGG(DISTINCT p.icd_code) as procedure_list,
    
    -- 3-digit grouping (for similarity)
    ARRAY_AGG(DISTINCT SUBSTRING(p.icd_code, 1, 3)) as proc_icd_list,
    
    -- ICD versions present
    STRING_AGG(DISTINCT CAST(p.icd_version AS VARCHAR), ',') as proc_icd_versions,
    
    -- Counts
    COUNT(*) as proc_count,
    COUNT(DISTINCT SUBSTRING(p.icd_code, 1, 3)) as proc_group_count,
    
    -- REFINED: Surgical procedure detection (3-digit groups)
    -- ICD-9: 01-86 (use SUBSTRING to check)
    COUNT(*) FILTER (WHERE 
        (p.icd_version = 9 AND SUBSTRING(p.icd_code, 1, 2) IN (
            '01','02','03','04','05','06','07','08','09','10','11','12','13','14','15',
            '16','17','18','19','20','21','22','23','24','25','26','27','28','29','30',
            '31','32','33','34','35','36','37','38','39','40','41','42','43','44','45',
            '46','47','48','49','50','51','52','53','54','55','56','57','58','59','60',
            '61','62','63','64','65','66','67','68','69','70','71','72','73','74','75',
            '76','77','78','79','80','81','82','83','84','85','86'
        ))
        OR (p.icd_version = 10 AND SUBSTRING(p.icd_code, 1, 1) = '0')
    ) as surgery_count
FROM procedures_icd p
LEFT JOIN d_icd_procedures d 
    ON p.icd_code = d.icd_code 
    AND p.icd_version = d.icd_version
GROUP BY p.subject_id, p.hadm_id;

-- Step 3: Prescriptions - RAW nested structure
CREATE OR REPLACE TABLE prescriptions_raw_agg AS
SELECT 
    subject_id,
    hadm_id,
    ARRAY_AGG(
        STRUCT_PACK(
            drug := drug,
            drug_type := drug_type,
            formulary_drug_cd := formulary_drug_cd,
            prod_strength := prod_strength,
            doses_per_24_hrs := doses_per_24_hrs,
            route := route,
            starttime := starttime,
            stoptime := stoptime
        )
    ) as prescriptions_raw
FROM rx_cleaned
WHERE starttime IS NOT NULL
GROUP BY subject_id, hadm_id;

-- Step 4: Prescriptions - ENGINEERED features
CREATE OR REPLACE TABLE prescriptions_features AS
SELECT 
    subject_id,
    hadm_id,
    
    -- Drug list (for recommendations)
    ARRAY_AGG(DISTINCT LOWER(drug)) as rx_drug_list,
    
    -- Route diversity
    ARRAY_AGG(DISTINCT route) as rx_routes_list,
    COUNT(DISTINCT route) as route_diversity,
    
    -- Drug types
    ARRAY_AGG(DISTINCT drug_type) as rx_drug_types,
    
    -- Counts
    COUNT(*) as rx_total_count,
    COUNT(DISTINCT LOWER(drug)) as rx_unique_drugs,
    COUNT(DISTINCT formulary_drug_cd) as rx_unique_formulary_drugs,
    
    -- REFINED: IV detection (case-insensitive with LIKE)
    COUNT(*) FILTER (WHERE UPPER(route) LIKE '%IV%') as rx_iv_count,
    COUNT(*) FILTER (WHERE UPPER(route) LIKE '%PO%' OR UPPER(route) LIKE '%ORAL%') as rx_oral_count,
    
    -- IV ratio
    ROUND(
        COUNT(*) FILTER (WHERE UPPER(route) LIKE '%IV%')::FLOAT 
        / NULLIF(COUNT(*), 0), 
        2
    ) as rx_iv_ratio,
    
    -- REFINED: Treatment duration (days on medication)
    COUNT(DISTINCT DATE(starttime)) as treatment_days,
    
    -- Min/Max duration per drug
    ROUND(AVG(EXTRACT(EPOCH FROM (stoptime - starttime)) / 86400.0), 1) as avg_rx_duration_days

FROM rx_cleaned
WHERE starttime IS NOT NULL
GROUP BY subject_id, hadm_id;

-- Step 5: FINAL TABLE - Both raw + engineered (REFINED join pattern)
CREATE OR REPLACE TABLE cdss_treatment AS
SELECT 
    COALESCE(proc.subject_id, rx.subject_id) as subject_id,
    COALESCE(proc.hadm_id, rx.hadm_id) as hadm_id,
    
    -- ========== RAW DATA (for ML exploration) ==========
    proc_raw.procedures_icd9,
    proc_raw.procedures_icd10,
    rx_raw.prescriptions_raw,
    
    -- ========== ENGINEERED FEATURES (for CBR + Recommendation) ==========
    -- Procedures
    COALESCE(proc.procedure_list, []) as procedure_list,
    COALESCE(proc.proc_icd_list, []) as proc_icd_list,
    COALESCE(proc.proc_icd_versions, '') as proc_icd_versions,
    COALESCE(proc.proc_count, 0) as proc_count,
    COALESCE(proc.proc_group_count, 0) as proc_group_count,
    COALESCE(proc.surgery_count, 0) as surgery_count,
    
    -- Prescriptions
    COALESCE(rx.rx_drug_list, []) as rx_drug_list,
    COALESCE(rx.rx_routes_list, []) as rx_routes_list,
    COALESCE(rx.route_diversity, 0) as route_diversity,
    COALESCE(rx.rx_drug_types, []) as rx_drug_types,
    COALESCE(rx.rx_total_count, 0) as rx_total_count,
    COALESCE(rx.rx_unique_drugs, 0) as rx_unique_drugs,
    COALESCE(rx.rx_unique_formulary_drugs, 0) as rx_unique_formulary_drugs,
    COALESCE(rx.rx_iv_count, 0) as rx_iv_count,
    COALESCE(rx.rx_oral_count, 0) as rx_oral_count,
    COALESCE(rx.rx_iv_ratio, 0.0) as rx_iv_ratio,
    COALESCE(rx.treatment_days, 0) as treatment_days,
    COALESCE(rx.avg_rx_duration_days, 0.0) as avg_rx_duration_days,
    
    -- ========== TREATMENT INTENSITY (REFINED: numeric + label) ==========
    -- Numeric score (sum of normalized signals)
    ROUND(
        LEAST(1.0, COALESCE(rx.rx_unique_drugs, 0) / 15.0) * 0.35 +
        LEAST(1.0, COALESCE(proc.proc_count, 0) / 5.0) * 0.30 +
        LEAST(1.0, COALESCE(rx.rx_iv_count, 0) / 10.0) * 0.20 +
        LEAST(1.0, COALESCE(rx.treatment_days, 0) / 14.0) * 0.15
    , 3) as treatment_complexity_score,
    
    -- Categorical label (optional, based on score)
    CASE 
        WHEN COALESCE(rx.rx_unique_drugs, 0) > 15 
          OR COALESCE(proc.surgery_count, 0) > 0 
        THEN 'High'
        WHEN COALESCE(rx.rx_unique_drugs, 0) > 5 
          OR COALESCE(proc.proc_count, 0) > 1 
        THEN 'Medium'
        ELSE 'Low'
    END as treatment_intensity_label
    
FROM procedures_features proc
FULL OUTER JOIN prescriptions_features rx 
    ON proc.hadm_id = rx.hadm_id
    AND proc.subject_id = rx.subject_id
-- REFINED: Join pattern (no correlated subqueries)
LEFT JOIN procedures_raw_agg proc_raw
    ON proc.hadm_id = proc_raw.hadm_id
    AND proc.subject_id = proc_raw.subject_id
LEFT JOIN prescriptions_raw_agg rx_raw
    ON rx.hadm_id = rx_raw.hadm_id
    AND rx.subject_id = rx_raw.subject_id
ORDER BY COALESCE(proc.hadm_id, rx.hadm_id);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_cdss_treatment_subject 
    ON cdss_treatment(subject_id);

CREATE INDEX IF NOT EXISTS idx_cdss_treatment_complexity_score 
    ON cdss_treatment(treatment_complexity_score);

CREATE INDEX IF NOT EXISTS idx_cdss_treatment_intensity_label 
    ON cdss_treatment(treatment_intensity_label);

-- Verification query
SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT subject_id) as unique_subjects,
    COUNT(DISTINCT hadm_id) as unique_admissions,
    MIN(treatment_complexity_score) as min_complexity,
    MAX(treatment_complexity_score) as max_complexity,
    COUNT(*) FILTER (WHERE treatment_intensity_label = 'High') as high_intensity,
    COUNT(*) FILTER (WHERE treatment_intensity_label = 'Medium') as medium_intensity,
    COUNT(*) FILTER (WHERE treatment_intensity_label = 'Low') as low_intensity
FROM cdss_treatment;