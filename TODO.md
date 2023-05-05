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
5. Specific Activities only occuring with specific Device IDs
6. All IDs must be within a certain range

combine the datasets which means fix sliding window

try implementing the other validation forms:
   Leave one subject out cross validation (maybe not?)
   one to one
   many to one
   stratified three fold cross validation

fix the constraints
(do the constraints account for normalizing the data?)
cite him as a source (not urgent)

fix the error of undoing the label encoding, the datasets are coming out too small.