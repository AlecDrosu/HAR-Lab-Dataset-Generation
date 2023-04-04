1. ~~Fix Pre Processing~~
   - ~~Investigate issue with random activities and activity statuses~~
   - ~~Address and correct any issues found~~
2. ~~Move Processing Files~~
   - ~~Move pre processing, post processing, and intermediate processing files to separate Jupyter notebook~~
3. Implement RNN network or similar
   - Train VAE model on event sequences
   - Ensure the following sequences are recognized:
     1. Device IDs starting with 'M' must be followed by 'ON' or 'OFF' status
     2. Device IDs starting with 'T' must be followed by degrees in Celsius
     3. Device IDs starting with 'D' must be followed by 'OPEN' or 'CLOSE' status
     4. Activities must be followed by a 'begin' or 'end' status
         - Meal_Preparation
         - Relax
         - Eating
         - Work
         - Sleeping
         - Wash_Dishes
         - Bed_to_Toilet
         - Enter_Home
         - Leave_Home
         - Housekeeping
         - Respirate
     5. Specific Activities only occuring with specific Device IDs, but hopefully the RNN will be able to figure this out