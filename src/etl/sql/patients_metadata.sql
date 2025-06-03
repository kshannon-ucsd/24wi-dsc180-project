DROP VIEW IF EXISTS second_feature;

-- Create view to export data
CREATE VIEW second_feature AS(
    -- Obtain necessary vital signs for sepsis diagnosis
    WITH vitals AS (
        SELECT v.subject_id, v.stay_id, v.charttime, v.heart_rate, v.sbp, v.sbp_ni, v.mbp, v.mbp_ni, v.resp_rate, v.temperature
        FROM mimiciv_derived.vitalsign AS v
        WHERE v.subject_id IS NOT NULL 
    ),

    --- Obtain platelet and white blood cell count
    blood AS (
        SELECT b.subject_id, b.hadm_id, b.charttime, b.platelet, b.wbc
        FROM mimiciv_derived.complete_blood_count AS b
        WHERE b.subject_id IS NOT NULL AND b.hadm_id IS NOT NULL
    ),

    --- Obtain bands measure for white blood cells
    white_blood AS (
        SELECT b.subject_id, b.hadm_id, b.charttime, b.wbc, b.bands
        FROM mimiciv_derived.blood_differential AS b
        WHERE b.subject_id IS NOT NULL AND b.hadm_id IS NOT NULL
    ),

    --- Obtain lactate measurement
    lactate AS (
        SELECT b.subject_id, b.hadm_id, b.charttime, b.lactate
        FROM mimiciv_derived.bg AS b
        WHERE b.subject_id IS NOT NULL AND b.hadm_id IS NOT NULL
    ),

    --- Obtain inr and ptt measurement
    inr_ptt AS (
        SELECT c.subject_id, c.hadm_id, c.charttime, c.inr, c.ptt
        FROM mimiciv_derived.coagulation AS c
        WHERE c.subject_id IS NOT NULL AND c.hadm_id IS NOT NULL
    ),

    --- Obtain creatinine measurement
    creatinine AS (
        SELECT c.subject_id, c.hadm_id, c.charttime, c.creatinine
        FROM mimiciv_derived.chemistry AS c
        WHERE c.subject_id IS NOT NULL AND c.hadm_id IS NOT NULL
    ),

    --- Obtain ICD codes related to bilirubin
    bilirubin_label AS (
        SELECT l2.itemid, l2.label
        FROM mimiciv_hosp.d_labitems AS l2
        WHERE LOWER(label) LIKE '%bilirubin%'
    ), 

    --- Obtain patients lab records
    bilirubin_merge AS (
        SELECT l1.itemid, l1.subject_id, l1.hadm_id, l1.charttime, l1.valuenum, l1.valueuom
        FROM mimiciv_hosp.labevents AS l1
        WHERE l1.hadm_id IS NOT NULL AND l1.valuenum IS NOT NULL
    ),

    --- Obtain patients' bilirubin measurement
    bilirubin AS (
        SELECT b1.itemid, b1.subject_id, b1.hadm_id, b1.charttime, b1.valuenum AS bilirubin
        FROM bilirubin_merge AS b1
        JOIN bilirubin_label AS b2 ON b1.itemid = b2.itemid
        WHERE b1.subject_id IS NOT NULL AND b1.hadm_id IS NOT NULL
    ),

    --- Obtain pneumonia indicator
	pneumonia AS (
		SELECT d.subject_id, d.hadm_id, 1 AS pneumonia
		FROM mimiciv_hosp.diagnoses_icd AS d
		JOIN (
			SELECT i.icd_code
			FROM mimiciv_hosp.d_icd_diagnoses AS i
			WHERE LOWER(long_title) LIKE '%pneumonia%'
		) AS p ON d.icd_code = p.icd_code
		WHERE d.subject_id IS NOT NULL AND d.hadm_id IS NOT NULL
	),

    --- Merge vitals and blood measurements together
    vitals_blood AS (
        SELECT v.subject_id, b.hadm_id, v.stay_id, v.charttime, v.heart_rate, v.sbp, v.sbp_ni, v.mbp, v.mbp_ni, v.resp_rate, v.temperature, b.platelet, b.wbc
        FROM vitals AS v
        FULL JOIN blood AS b ON v.subject_id = b.subject_id AND v.charttime = b.charttime
    ),

    --- Merge previous with white blood cell bands
    complete_vitals_blood AS (
        SELECT v.subject_id, v.hadm_id, v.stay_id, v.charttime, v.heart_rate, v.sbp, v.sbp_ni, v.mbp, v.mbp_ni, v.resp_rate, v.temperature, v.platelet, v.wbc, w.bands
        FROM vitals_blood AS v
        FULL JOIN white_blood AS w ON v.subject_id = w.subject_id AND v.hadm_id = w.hadm_id AND v.charttime = w.charttime
    ),

    --- Merge previous with lactate measurement
    merge_lactate AS (
        SELECT v.subject_id, v.hadm_id, v.stay_id, v.charttime, v.heart_rate, v.sbp, v.sbp_ni, v.mbp, v.mbp_ni, v.resp_rate, v.temperature, v.platelet, v.wbc, v.bands, l.lactate
        FROM complete_vitals_blood AS v
        FULL JOIN lactate AS l ON v.subject_id = l.subject_id AND v.hadm_id = l.hadm_id AND v.charttime = l.charttime
    ),

    --- Merge previous with inr and ptt measurement
    merge_inr AS (
        SELECT v.subject_id, v.hadm_id, v.stay_id, v.charttime, v.heart_rate, v.sbp, v.sbp_ni, v.mbp, v.mbp_ni, v.resp_rate, v.temperature, v.platelet, v.wbc, v.bands, v.lactate, i.inr, i.ptt
        FROM merge_lactate AS v
        FULL JOIN inr_ptt AS i ON v.subject_id = i.subject_id AND v.hadm_id = i.hadm_id AND v.charttime = i.charttime
    ),

    --- Merge previous with creatinine measurement
    merge_creatinine AS (
        SELECT v.subject_id, v.hadm_id, v.stay_id, v.charttime, v.heart_rate, v.sbp, v.sbp_ni, v.mbp, v.mbp_ni, v.resp_rate, v.temperature, v.platelet, v.wbc, v.bands, v.lactate, v.inr, v.ptt, c.creatinine
        FROM merge_inr AS v
        FULL JOIN creatinine AS c ON v.subject_id = c.subject_id AND v.hadm_id = c.hadm_id AND v.charttime = c.charttime
    ),

    --- Merge previous with bilirubin measurement
    merge_bilirubin AS (
        SELECT v.subject_id, v.hadm_id, v.stay_id, v.charttime, v.heart_rate, v.sbp, v.sbp_ni, v.mbp, v.mbp_ni, v.resp_rate, v.temperature, v.platelet, v.wbc, v.bands, v.lactate, v.inr, v.ptt, v.creatinine, b.bilirubin
        FROM merge_creatinine AS v
        FULL JOIN bilirubin AS b ON v.subject_id = b.subject_id AND v.hadm_id = b.hadm_id AND v.charttime = b.charttime
    ),

    --- Merge previous with pneumonia indicator
	merge_pneumonia AS (
        SELECT v.subject_id, v.hadm_id, v.stay_id, v.charttime, v.heart_rate, v.sbp, v.sbp_ni, v.mbp, v.mbp_ni, v.resp_rate, v.temperature, v.platelet, v.wbc, v.bands, v.lactate, v.inr, v.ptt, v.creatinine, v.bilirubin, COALESCE(p.pneumonia, 0) AS pneumonia 
        FROM merge_bilirubin AS v
        FULL JOIN pneumonia AS p ON v.subject_id = p.subject_id AND v.hadm_id = p.hadm_id
    )

    SELECT *
    FROM merge_pneumonia
    WHERE hadm_id IS NOT NULL
);

--- Store data into csv
\copy (SELECT * FROM second_feature) TO 'data/sql-data/patients_metadata.csv' WITH DELIMITER ',' CSV HEADER