# Download DuckDB

Go here: https://duckdb.org/docs/installation/

Download: duckdb_cli-windows-amd64.zip (do differently for MAC)
Step 2 — Extract the zip in "AI CDSS" folder

You'll get: duckdb.exe

# Database file in the main folder (not to be uploaded to github)

make a mimic.db file in the AI CDSS folder
Now in the **load_tables.py** correct the folder path as per your device and load the data by running the following script in command terminal (make sure you are in the "AI CDSS" folder) : python load_tables.py
then run : python -c "import duckdb; print(duckdb.connect('mimic.db').execute('SHOW TABLES').fetchall())"
Now you'll be able to see all of these tables (if not, ask ChatGPT) : "patients.csv.gz", "admissions.csv.gz", "omr.csv.gz" "diagnoses_icd.csv.gz", "d_icd_diagnoses.csv.gz", "prescriptions.csv.gz", "procedures_icd.csv.gz", "d_icd_procedures.csv.gz", "labevents.csv.gz", "d_labitems.csv.gz", "transfers.csv.gz",

# If you want preview of these tables :

run : python preview_mimic.py
output : 10-25 rows of each table with each column on your browser

# Analysis of base-info for CDSS :

Correct the BASE_PATH and DB_PATH as per your device in 01_cdss_base.py but you don't need to run it

# for analysis, just attach the preprocessing_log.txt and the following prompt :

remember the structure of these 3 files :

MIMIC-IV preview: patients.csv.gz

Generated: 2026-03-13 15:08:46 • Columns: 6

subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod

10000032, F, 52, 2180, 2014 - 2016, 2180-09-09

10000048, F, 23, 2126, 2008 - 2010, NaT

10000058, F, 33, 2168, 2020 - 2022, NaT

10000068, F, 19, 2160, 2008 - 2010, NaT

10000084, M, 72, 2160, 2017 - 2019, 2161-02-13

MIMIC-IV preview: admissions.csv.gz

Generated: 2026-03-13 15:08:48 • Columns: 16

subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, admit_provider_id, admission_location, discharge_location, insurance, language, marital_status, race, edregtime, edouttime, hospital_expire_flag

10000032, 22595853, 2180-05-06 22:23:00, 2180-05-07 17:15:00, NaT, URGENT, P49AFC, TRANSFER FROM HOSPITAL, HOME, Medicaid, English, WIDOWED, WHITE, 2180-05-06 19:17:00, 2180-05-06 23:30:00, 0

10000032, 22841357, 2180-06-26 18:27:00, 2180-06-27 18:49:00, NaT, EW EMER., P784FA, EMERGENCY ROOM, HOME, Medicaid, English, WIDOWED, WHITE, 2180-06-26 15:54:00, 2180-06-26 21:31:00, 0

10000032, 25742920, 2180-08-05 23:44:00, 2180-08-07 17:50:00, NaT, EW EMER., P19UTS, EMERGENCY ROOM, HOSPICE, Medicaid, English, WIDOWED, WHITE, 2180-08-05 20:58:00, 2180-08-06 01:44:00, 0

10000032, 29079034, 2180-07-23 12:35:00, 2180-07-25 17:55:00, NaT, EW EMER., P06OTX, EMERGENCY ROOM, HOME, Medicaid, English, WIDOWED, WHITE, 2180-07-23 05:54:00, 2180-07-23 14:00:00, 0

MIMIC-IV preview: omr.csv.gz

Generated: 2026-03-15 19:36:36 • Columns: 5

subject_id, chartdate, seq_num, result_name, result_value

10000032, 2180-04-27, 1, Blood Pressure, 110/65

10000032, 2180-04-27, 1, Weight (Lbs), 94

10000032, 2180-05-07, 1, BMI (kg/m2), 18.0

10000032, 2180-05-07, 1, Height (Inches), 60

Also remember these rules we decided :
For each (subject_id, hadm_id):

1. Find all OMR records where:
   - subject_id matches

   - chartdate BETWEEN admittime AND dischtime

   - result_name IN ('Height', 'Weight', 'BMI (kg/m2)', 'Blood Pressure')

2. Get the max(chartdate) → latest_date

3. Filter to:
   - chartdate = latest_date

   - seq_num = 1

4. Pivot result_name → columns (Height, Weight, BMI, BloodPressure)

5. If no match found → NULL

Now give the detailed analysis as seen in the preprocessing_log.txt

You'll get exact detailed info of what **base personal info** you'll be working with

# Make 01_cdss_diagnoses in the database

correct the paths in **01_cdss_diagnoses.py** file as per your device
run "python 01_cdss_diagnoses.py"
now you have 01_cdss_diagnoses table in mimic.db

# Make 01_cdss_treatment in the databse

make sure you're in "AI CDSS" folder in the CLI
now run : "./duckdb.exe mimic.db"
then you'll have something like : mimic D
now just run : ".read preprocessing/01_cdss_treatment.sql"
