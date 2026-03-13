CREATE TABLE mservices(
Patient_id VARCHAR,
Services VARCHAR,
Overweight VARCHAR,
Arthritis VARCHAR,Diabetes VARCHAR,
Hyperlipidemia VARCHAR,
BackPain VARCHAR,
Anxiety VARCHAR,
Allergic_rhinitis VARCHAR,
Reflux_esophagitis VARCHAR,
Asthma VARCHAR,
PRIMARY KEY(patient_id)

COPY public.mservices (patient_id, services, overweight, arthritis, diabetes, hyperlipidemia,
backpain, anxiety, allergic_rhinitis, reflux_esophagitis, asthma)
FROM ‘C:\LabFiles\Medical\Mservices.CSV’
DELIMITER ‘,’
CSV HEADER ;

CREATE TABLE data_analysis
AS (select patient_id, soft_drink
FROM patient)

CREATE TABLE calc1
AS
SELECT data_analysis.patient_id, m.overweight, data_analysis.soft_drink
FROM mservices AS m
INNER JOIN data_analysis ON data_analysis.patient_id = m.patient_id ;

CREATE TABLE final_table
(total_over INT,
Overandsoftdrink INT) ;

INSERT INTO final_table (total_over)
SELECT COUNT(*)
FROM calc1
WHERE overweight =’Yes’ ;

INSERT INTO final_table(overandsoftdrink)
SELECT COUNT(*)
FROM calc1
WHERE overweight =’Yes’ AND soft_drink = ‘Yes’