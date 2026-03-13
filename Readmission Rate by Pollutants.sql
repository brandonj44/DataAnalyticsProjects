SELECT "Patient (CusSQL)"."admis_id" AS "admis_id",
        "Patient (CusSQL)"."age" AS "age",
        CAST("Patient (CusSQL)"."city" AS TEXT) AS "city (Custom SQL Query)",
        CAST("location"."city" AS TEXT) AS "city (location)",
        "Patient (CusSQL)"."compl_id" AS "compl_id",
        CAST("location"."county" AS TEXT) AS "county (location)",
        CAST("Patient (CusSQL)"."gender" AS TEXT) AS "gender",
        CAST("Patient (CusSQL)"."highblood" AS TEXT) AS "highblood",
        "Patient (CusSQL)"."initial_days" AS "initial_days",
        "Patient (CusSQL)"."lat" AS "lat",
        "Patient (CusSQL)"."lng" AS "lng",
        "location"."location_id" AS "location_id (location) #1",
        "Patient (CusSQL)"."location_id" AS "location_id",
        CAST("Patient (CusSQL)"."patient_id" AS TEXT) AS "patient_id",
        "Patient (CusSQL)"."population" AS "population",
        CAST("Patient (CusSQL)"."readmis" AS TEXT) AS "readmis",
        CAST("Patient (CusSQL)"."soft_drink" AS TEXT) AS "soft_drink",
        CAST("location"."state" AS TEXT) AS "state (location)",
        CAST("Patient (CusSQL)"."stroke" AS TEXT) AS "stroke",
        "location"."zip" AS "zip (location)"
 FROM (
          SELECT "patient"."admis_id" AS "admis_id",
                 "patient"."age" AS "age",
                 "patient"."compl_id" AS "compl_id",
                 CAST("patient"."gender" AS TEXT) AS "gender",
                 CAST("patient"."hignblood" AS TEXT) AS "highblood",
                 "patient"."initial_days" AS "initial_days",
                 "patient"."lat" AS "lat",
                 "patient"."lng" AS "lng",
                 "patient"."location_id" AS "location_id",
                 CAST("patient"."patient_id" AS TEXT) AS "patient_id",
                 "patient"."population" AS "population",
                 CAST("patient"."readmis" AS TEXT) AS "readmis",
                 CAST("patient"."soft_drink" AS TEXT) AS "soft_drink",
                 CAST("patient"."stroke" AS TEXT) AS "stroke",
                 "location"."city" AS "city"
          FROM "public"."patient" "patient"
                   LEFT JOIN "public"."location" "location"
                             ON  "patient"."location_id" = "location"."location_id"
      ) "Patient (CusSQL)"
          LEFT JOIN "public"."location" "location" ON ("Patient (CusSQL)"."location_id" = "location"."location_id")