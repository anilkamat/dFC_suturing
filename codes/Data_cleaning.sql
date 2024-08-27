
# 1. Remove Duplicates 
# 2. Standarize the Data
# 3. Null values or blank values 
# 4. Remove any columns


# SETTING up 

-- CREATE TABLE layoffs_staging 
-- LIKE layoffs; 

-- INSERT layoffs_staging
-- SELECT *
-- FROM layoffs;


# 1. Remove Duplicates 
SELECT *
FROM layoffs_staging;

SELECT *,
ROW_NUMBER() OVER(
PARTITION BY company, industry, total_laid_off, percentage_laid_off, 'date') AS row_num #stage, country, funds_raised_millions
FROM layoffs_staging;

WITH duplicate_CTE AS (
SELECT *,
ROW_NUMBER() OVER(
PARTITION BY company,location, industry, total_laid_off, percentage_laid_off, 'date', stage, country, funds_raised_millions) AS row_num #stage, country, funds_raised_millions
FROM layoffs_staging
)
SELECT * 
FROM duplicate_cte
WHERE row_num >1;

SELECT *
FROM layoffs_staging
WHERE company = "Casper";

WITH duplicate_CTE AS (
SELECT *,
ROW_NUMBER() OVER(
PARTITION BY company,location, industry, total_laid_off, percentage_laid_off, 'date', stage, country, funds_raised_millions) AS row_num #stage, country, funds_raised_millions
FROM layoffs_staging
)
DELETE
FROM duplicate_cte
WHERE row_num >1;



CREATE TABLE `layoffs_staging2` (
  `company` text,
  `location` text,
  `industry` text,
  `total_laid_off` int DEFAULT NULL,
  `percentage_laid_off` text,
  `date` text,
  `stage` text,
  `country` text,
  `funds_raised_millions` int DEFAULT NULL, 
  `row_num` INT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

SELECT *
FROM layoffs_staging2;

INSERT INTO layoffs_staging2
SELECT *,
ROW_NUMBER() OVER(
PARTITION BY company,location, industry, total_laid_off, percentage_laid_off, 'date', stage, country, funds_raised_millions) AS row_num 
FROM layoffs_staging;

SET SQL_SAFE_UPDATES = 0;  # Turn offs the safe mode for deletion in the subsequent step

DELETE
FROM layoffs_staging2
WHERE row_num > 1;

SELECT *
FROM layoffs_staging2;


# 2. Standarize the Data

SELECT company, TRIM(company)
FROM layoffs_staging2;

UPDATE layoffs_staging2
SET company = TRIM(company);

SELECT DISTINCT industry
FROM layoffs_staging2
ORDER BY 1;

SELECT *
FROM layoffs_staging2
WHERE industry LIKE "Crypto%";

UPDATE layoffs_staging2
SET industry = 'Crypto'
WHERE industry LIKE 'Crypto%';

SELECT *
FROM layoffs_staging2
WHERE industry LIKE "Crypto";

SELECT DISTINCT country
FROM layoffs_staging2
ORDER BY 1;
#WHERE country LIKE "united States%";

UPDATE layoffs_staging2
SET country = 'United States'
WHERE country LIKE 'United States.';

SELECT *
FROM layoffs_staging2;

SELECT date,  # NOTE date here is used as column despite it beinga keyword. It was automatically prompted to do so with list of options. 
STR_TO_DATE(date , '%m/%d/%Y')
FROM layoffs_staging2;

UPDATE layoffs_staging2
SET date = STR_TO_DATE(date , '%m/%d/%Y');

ALTER TABLE layoffs_staging2
MODIFY COLUMN date DATE;


# 3. Null values or blank values
SELECT *
FROM layoffs_staging2
WHERE total_laid_off IS NULL
AND percentage_laid_off IS NULL;

SELECT *
FROM layoffs_staging2
WHERE industry IS NULL
OR industry = "";

UPDATE layoffs_staging2
SET industry = NULL
WHERE industry = '' ;

SELECT *
FROM layoffs_staging2
WHERE company = "Airbnb"
OR industry = "";

SELECT *
FROM layoffs_staging2 t1
JOIN layoffs_staging2 t2
	ON t1.company = t2.company 
WHERE (t1.industry IS NULL OR t1.industry = '')
AND t2.industry IS NOT NULL;

UPDATE layoffs_staging2 t1 
JOIN layoffs_staging2 t2
	ON t1.company = t2.company
SET t1.industry = t2.industry
WHERE t1.industry IS NULL 
AND t2.industry IS NOT NULL ;

# 4. Remove any columns
SELECT *
FROM layoffs_staging2
WHERE total_laid_off IS NULL;

DELETE 
FROM layoffs_staging2
WHERE total_laid_off IS NULL
AND percentage_laid_off IS NULL;

SELECT *
FROM layoffs_staging2;

ALTER TABLE layoffs_staging2
DROP COLUMN row_num;

SELECT *
FROM layoffs_staging2;

















