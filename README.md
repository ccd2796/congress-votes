# congress-votes
OCR applied to detect votes in the Congress of the Republic of Peru

Steps:
  -PDF files with scanned acts of votations were downloaded from https://www.congreso.gob.pe/AsistenciasVotacionesPleno/
  -Manually labeled pages with votations to be converted to images.
  -Used OpenCV to detect the title and the table with votes.
  -Sliced the table in three columns and used PyTesseract to detect the number of votes in each column.
  -Used PyTesseract to read the text in sliced images, stored output in preliminar file.
  -Curated the results in two steps: name of congresspeople was detected using Jaro-Winkler distance, political party and vote were curated manually.
  
 Steps for community detection:
  -Create table with consolidated votes.
  -Recoded votes: 1 for "YES", -1 for "NO" and 0 for "Absent", null for others.
  -Removed votations with 87% or more "YES" votes.
  -Removed congresspeople that didn't vote at least 80 times during the year.
  -Constructed the distance measure: the % of votations that two congresspeople voted equally.
  -Applied HCA to detect the 10 more prominent groups within congress.
  
 
