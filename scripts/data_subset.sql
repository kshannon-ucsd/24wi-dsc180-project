DROP VIEW IF EXISTS subset;

-- Create view to export data
CREATE VIEW subset AS(
	-- Identify if patient has chronic pulmonary disease
	WITH pulmonary AS (
		SELECT c.subject_id, c.hadm_id, c.chronic_pulmonary_disease
		FROM mimiciv_derived.charlson AS c
	),

	-- Create mapping for subject id, hospital admission id, and stay id
	id_map AS (
		SELECT i.subject_id, i.hadm_id, i.stay_id
		FROM mimiciv_icu.icustays AS i
	),

	-- Maps the patients
	id_pulmonary AS (
		SELECT p.subject_id, p.hadm_id, i.stay_id, p.chronic_pulmonary_disease
		FROM pulmonary AS p
		JOIN id_map AS i ON i.subject_id = p.subject_id AND i.hadm_id = p.hadm_id
	),

	-- Determine if patients has sepsis or not
	has_sepsis AS (
		SELECT i.subject_id, i.hadm_id, i.stay_id, i.chronic_pulmonary_disease, COALESCE(s.sepsis3, false) AS sepsis3
		FROM id_pulmonary AS i
		LEFT JOIN mimiciv_derived.sepsis3 AS s ON i.subject_id = s.subject_id AND i.stay_id = s.stay_id
	),

	-- Determine time patients were admitted and discharged
	admission_time AS (
		SELECT h.subject_id, h.hadm_id, h.admittime, h.dischtime
		FROM mimiciv_hosp.admissions AS h
	),

	-- Map the admission and discharge time to patients
	sepsis_admission AS (
		SELECT s.subject_id, s.hadm_id, s.stay_id, a.admittime, a.dischtime, (a.dischtime - a.admittime) AS los, s.chronic_pulmonary_disease, s.sepsis3
		FROM has_sepsis AS s
		JOIN admission_time AS a ON s.subject_id = a.subject_id AND s.hadm_id = a.hadm_id
	),

	-- Determine suspected time patient got sepsis and merge with subset
	infection_time AS (
		SELECT s1.subject_id, s1.hadm_id, s1.stay_id, s1.admittime, s1.dischtime, s1.los, s1.chronic_pulmonary_disease, s2.suspected_infection_time, s1.sepsis3
		FROM sepsis_admission AS s1
		LEFT JOIN (
			SELECT s.subject_id, s.stay_id, s.suspected_infection_time
			FROM mimiciv_derived.sepsis3 AS s
		) AS s2 ON s1.subject_id = s2.subject_id AND s1.stay_id = s2.stay_id
	)

	SELECT *
	FROM infection_time
);

\copy (SELECT * FROM subset) TO '../data/interim/subset.csv' WITH DELIMITER ',' CSV HEADER